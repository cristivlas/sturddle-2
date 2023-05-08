#!/usr/bin/env python3
'''
Trainer for the Sturddle Chess 2.0 engine's neural network.
Copyright (c) 2023 Cristian Vlasceanu.

Expects memmapped numpy arrays or H5 files as inputs.
'''
import argparse
import logging
import os
import sys
from contextlib import redirect_stdout

import h5py
import numpy as np

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Quantization range: use int16_t with QSCALE of 1024, and need to add 18 values
# (16 weights, 1 bias, 1 residual) w/o overflow, max representable value is 32767 / 18 / 1024
Q_MAX = 1.7777
Q_MIN = -Q_MAX

def configure_logging(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename=args.logfile,
        format='%(asctime)s;%(levelname)s;%(message)s',
        level=log_level,
    )
    # silence off annoying logging, see https://github.com/abseil/abseil-py/issues/99
    logging.getLogger('absl').addFilter(lambda *_:False)
    return log_level


def loss_function(args):
    @tf.function
    def _clipped_mae(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, -args.clip, args.clip)
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)
    if args.clip:
        # tf.keras.utils.get_custom_objects().update({'_clipped_mae': _clipped_mae})
        return _clipped_mae
    else:
        return MeanAbsoluteError()

'''
*****************************************************************************
The deep model is used as a "teacher" for online distillation training.

*****************************************************************************
'''
def make_deep_model(args, starting_units=4096):
    def create_dense_layer(inputs, units, activation, name):
        x = Dense(units, activation=activation, name=name)(inputs)
        x = BatchNormalization(name=f'bn_{name}')(x)
        x = Dropout(0.2, name=f'dropout_{name}')(x)
        return x

    input_layer = Input(shape=(args.hot_encoding,), name='input')

    x = input_layer
    i = 1
    units = starting_units
    while units >= 16:
        x = create_dense_layer(x, units, 'relu', f'dense_{i}')
        units //= 2
        i += 1

    output_layer = Dense(1, name='output')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name=args.name)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learn_rate,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=args.optimizer == 'amsgrad',
    )
    model.compile(loss=loss_function(args), optimizer=optimizer, metrics=[])

    total_params = sum([tf.reduce_prod(variable.shape).numpy() for variable in model.trainable_variables])
    print(f'Trainable parameters in deep teacher model: {total_params:,}')

    return model


'''
*****************************************************************************
This is the "student" model.

*****************************************************************************
'''
def make_model(args):
    activation = tf.keras.activations.relu

    # Define the input layer
    input_layer = Input(shape=(args.hot_encoding,), name='input')

    def black_occupied_mask(x):
        mask = tf.zeros_like(x[:, :64])
        for i in range(0, 12, 2):
            mask = tf.math.add(mask, x[:, i*64:(i+1)*64])
        return mask

    def white_occupied_mask(x):
        mask = tf.zeros_like(x[:, :64])
        for i in range(1, 12, 2):
            mask = tf.math.add(mask, x[:, i*64:(i+1)*64])
        return mask

    # Extracting black occupation mask (summing black pieces' bitboards)
    black_occupied = Lambda(black_occupied_mask)(input_layer)

    # Extracting white occupation mask (summing white pieces' bitboards)
    white_occupied = Lambda(white_occupied_mask)(input_layer)

    concat = Concatenate()([input_layer, black_occupied, white_occupied])

    # Define layer 1a
    hidden_1a = Dense(
        512,
        activation=activation,
        name='hidden_1a',
        kernel_constraint=MinMaxNorm(min_value=Q_MIN, max_value=Q_MAX),
        bias_constraint=MinMaxNorm(min_value=Q_MIN, max_value=Q_MAX),
    )(concat)

    # Add 2nd hidden layer
    hidden_2 = Dense(16, activation=activation, name='hidden_2')(hidden_1a)

    # Define hidden layer 1b (use kings and pawns to compute dynamic weights)
    input_1b = Lambda(lambda x: x[:, :256], name='slice_input_1b')(input_layer)
    hidden_1b = Dense(
        64,
        activation=activation,
        name='hidden_1b',
        kernel_constraint=MinMaxNorm(min_value=Q_MIN, max_value=Q_MAX),
        bias_constraint=MinMaxNorm(min_value=Q_MIN, max_value=Q_MAX),
    )(input_1b)

    # Compute dynamic weights based on hidden_1b
    dynamic_weights = Dense(
        16,
        activation=None,
        name='dynamic_weights',
        #kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        #bias_regularizer=L1L2(l1=1e-5, l2=1e-4),
    )(hidden_1b)

    # Apply dynamic weights to hidden_2
    weighted_hidden_2 = Multiply(name='weighted_hidden_2')([hidden_2, dynamic_weights])

    # Define the output layer
    output_layer = Dense(1, name='out', dtype='float32')(weighted_hidden_2)

    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name=args.name)

    if args.optimizer in ['adam', 'amsgrad']:
        optimizer=tf.keras.optimizers.Adam(
            amsgrad=args.optimizer=='amsgrad',
            beta_1=0.99,
            beta_2=0.995,
            learning_rate=args.learn_rate,
            use_ema=args.ema,
            weight_decay=args.decay if args.decay else None)
    elif args.optimizer == 'sgd':
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=args.learn_rate,
            nesterov=args.nesterov,
            use_ema=args.ema,
            weight_decay=args.decay if args.decay else None)
    else:
        assert False

    model.compile(loss=loss_function(args), optimizer=optimizer, metrics=[])
    return model


'''
*****************************************************************************
Export weights as C++ code snippet.

The numpy.float32 data type in NumPy uses the 32-bit floating-point
format defined by the IEEE 754 standard. According to the standard,
the maximum number of decimal digits of precision that can be
represented by a numpy.float32 value is 7.

*****************************************************************************
'''
def write_weigths(args, model, indent):
    for layer in model.layers:
        params = layer.get_weights()
        if not params:
            continue
        weights, biases = params
        rows, cols = weights.shape
        print(f'constexpr float {layer.name}_w[{rows}][{cols}] = {{')
        for i in range(rows):
            print(f'{" " * indent}{{', end='')
            for j in range(cols):
                if j % 8 == 0:
                    if j:
                        print(f'\n{" " * 2 * indent}', end='')
                    else:
                        print(f'{" " * (indent - 1)}', end='')
                print(f'{weights[i][j]:12.8f},', end='')
            if cols > 1:
                print()
            print(f'{" " * indent}}}, /* {i} */')
        print('};')

        assert len(biases.shape) == 1, biases.shape
        assert cols == biases.shape[0], biases.shape
        print(f'constexpr float {layer.name}_b[{cols}] = {{')
        for i in range(cols):
            if i % 8 == 0:
                if i:
                    print()
                print(f'{" " * 2 *indent}', end='')
            print(f'{biases[i]:12.8f},', end='')
        print('\n};')


def export_weights(args, model, indent=2):
    if args.export == sys.stdout:
        write_weigths(args, model, indent)
    else:
        with open(args.export, 'w+') as f:
            with redirect_stdout(f):
                write_weigths(args, model, indent)


'''
*****************************************************************************
Create TF dataset object.

*****************************************************************************
'''
def dataset_from_file(args, filepath, clip, strategy, callbacks):
    @tf.function
    def filter(x, y):
        lower_bound = tf.greater(y[0], -args.filter)
        upper_bound = tf.less(y[0], args.filter)
        condition = tf.logical_and(lower_bound, upper_bound)
        return tf.reduce_all(condition)

    '''
    Batch generator.
    '''
    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, x, y):
            assert len(x) == len(y)
            self.x, self.y = x, y
            self.batch_size = args.batch_size if args.batch_size else 1
            self.len = int(np.ceil(len(self.x) / self.batch_size))
            self.indices = np.arange(self.len)
            np.random.shuffle(self.indices)
            logging.info(f'using {self.len} batches.')

        def __call__(self):
            return self

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            i = self.indices[index]
            # assert 0 <= i < self.len
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x = self.x[start:end]
            y = self.y[start:end]
            return x, y

        def on_epoch_end(self):
            np.random.shuffle(self.indices)

    def make_dataset(x, y):
        class CallbackOnEpochEnd(tf.keras.callbacks.Callback):
            def __init__(self, generator):
                super(CallbackOnEpochEnd, self).__init__()
                self.generator = generator

            def on_epoch_end(self, epoch, logs=None):
                self.generator.on_epoch_end()

        generator = DataGenerator(x, y)
        if callbacks is None:
            return generator, None

        callbacks.append(CallbackOnEpochEnd(generator))
        steps_per_epoch = len(generator)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(dtype, dtype),
            output_shapes=((None, args.hot_encoding), (None, 1)),
        )

        if args.filter:
            dataset = dataset.filter(filter)
        dataset = dataset.prefetch(tf.data.AUTOTUNE).repeat()

        return dataset, steps_per_epoch

    print(f'Loading dataset {filepath}')
    dtype = np.float16 if args.half else np.float32

    if os.path.splitext(filepath)[1].lower() == '.h5':
        f = h5py.File(filepath)
        data = f['data']
        dtype = data.dtype
        row_count = data.shape[0]
        assert data.shape[1] == args.hot_encoding + 1, data.shape[1]
        print(f'{row_count:,} rows.')

        class LazyView:
            def __init__(self, data, slice_, rows):
                self.data = data
                self.slice_ = slice_
                self.len = rows

            def __getitem__(self, key):
                return self.data[key, self.slice_]

            def __len__(self):
                return self.len

        x = LazyView(data, slice(0, args.hot_encoding), row_count)
        y = LazyView(data, slice(args.hot_encoding, args.hot_encoding + 1), row_count)
    else:
        data = np.memmap(filepath, dtype=dtype, mode='r')
        row_count = data.shape[0] // (args.hot_encoding + 1)
        data = data.reshape((row_count, (args.hot_encoding + 1)))
        x = data[:,:args.hot_encoding]
        y = data[:,args.hot_encoding:]
        print(x.shape, y.shape)
    steps_per_epoch = None

    if args.distribute and callbacks is not None:
        dataset, steps_per_epoch = make_dataset(x, y)
        # distribute data accross several GPUs
        dataset = strategy.experimental_distribute_dataset(dataset)
    else:
        dataset, steps_per_epoch = make_dataset(x, y)
    return dataset, steps_per_epoch


'''
*****************************************************************************


*****************************************************************************
'''
def main(args):

    @tf.function
    def train_step(inputs, labels, student, teacher, student_loss_fn, alpha):
        mse_loss = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as student_tape, tf.GradientTape() as teacher_tape:
            # Get model outputs
            student_outputs = student(inputs, training=True)
            teacher_outputs = teacher(inputs, training=True)

            # Compute distillation loss
            distillation_loss = mse_loss(teacher_outputs, student_outputs) * alpha

            # Compute total loss as a weighted sum of the student loss and the distillation loss
            student_loss_weight = 1 - alpha
            distillation_loss_weight = alpha
            student_loss = student_loss_weight * student_loss_fn(labels, student_outputs)
            total_loss = student_loss + distillation_loss_weight * distillation_loss

        # Compute gradients and update student model weights
        student_gradients = student_tape.gradient(total_loss, student.trainable_variables)
        student.optimizer.apply_gradients(zip(student_gradients, student.trainable_variables))

        # Compute gradients and update teacher model weights
        teacher_gradients = teacher_tape.gradient(distillation_loss, teacher.trainable_variables)
        teacher.optimizer.apply_gradients(zip(teacher_gradients, teacher.trainable_variables))

        return total_loss, student_loss, distillation_loss

    def train_distillation(
            student,
            teacher,
            train_dataset,
            teacher_model_path,
            student_model_path,
            epochs,
            steps_per_epoch,
            callbacks,
            alpha):
        '''
        Train a student model using knowledge distillation with a teacher model.
        Both student and teacher models are trained simultaneously.
        '''
        # Checkpoints
        if teacher_model_path is not None:
            os.makedirs(os.path.dirname(teacher_model_path), exist_ok=True)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(teacher_model_path)
            checkpoint.model = teacher
            callbacks.append(checkpoint)

        if student_model_path is not None:
            os.makedirs(os.path.dirname(student_model_path), exist_ok=True)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(student_model_path)
            checkpoint.model = student
            callbacks.append(checkpoint)

        progbar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=['loss', 'student', 'distil'])
        for epoch in range(epochs):
            print (f'Epoch {epoch+1}/{epochs}')
            for batch, (inputs, labels) in enumerate(train_dataset):
                # Stop training after the specified number of steps per epoch, if provided
                if steps_per_epoch is not None and batch >= steps_per_epoch:
                    break

                # Call the train_step function
                total_loss, student_loss, distillation_loss = train_step(
                    inputs, labels, student, teacher, student.loss, alpha)

                # Update progress bar with loss information
                progress_values = [
                    ('loss', total_loss), ('student', student_loss), ('distil', distillation_loss)]
                progbar.update(batch+1, progress_values)

            # Call custom callbacks and save teacher and student model checkpoints at the end of each epoch
            for callback in callbacks:
                callback.on_epoch_end(epoch, {})

    if args.gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    # Load or initialize models.
    teacher_model = f'{args.model}-deep'
    with strategy.scope():
        model = make_model(args)
        deep_model = make_deep_model(args)

    if args.model and os.path.exists(args.model) and os.path.exists(teacher_model):
        custom_objects={'_clipped_mae' : None}
        # Load the student model
        saved_model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)
        print(f'Loaded model {os.path.abspath(args.model)}.')
        model.set_weights(saved_model.get_weights())

        # Load the teacher (deep) model
        saved_model = tf.keras.models.load_model(teacher_model, custom_objects=custom_objects)
        print(f'Loaded model {os.path.abspath(teacher_model)}.')
        deep_model.set_weights(saved_model.get_weights())

    if args.export:
        export_weights(args, model)
    else:
        callbacks = []
        dataset, steps_per_epoch = dataset_from_file(args, args.input[0], args.clip, strategy, callbacks)

        if args.schedule:
            from keras.callbacks import ReduceLROnPlateau
            lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-9)
            callbacks.append(lr)

        model.summary()
        if not args.model:
            print('*****************************************************************')
            print(' WARNING: checkpoint path not provided, model WILL NOT BE SAVED! ')
            print('*****************************************************************')

        if args.tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=args.logdir, profile_batch=(1, steps_per_epoch))
            callbacks.append(tensorboard_callback)

        # Alpha is a hyperparameter that controls the trade-off between the student loss
        # (i.e., the loss computed using the ground truth labels) and the distillation loss
        # Default is 0.1; with this value, the distillation loss will have a smaller weight
        # (10%) in the total loss compared to the student loss, which will have a weight of
        # 90% (1 - alpha). This means that the student model will focus more on fitting the
        # ground truth labels while still learning from the teacher model to some extent.
        args.alpha = max(0, min(1, args.alpha))
        train_distillation(model, deep_model, dataset, teacher_model,
                           args.model, args.epochs, steps_per_epoch, callbacks,
                           args.alpha)


if __name__ == '__main__':
    try:
        class CustomFormatter(
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter
        ):
            pass
        parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
        parser.add_argument('input', nargs=1, help='memmap-ed numpy, or h5, input data file path')
        parser.add_argument('-a', '--alpha', type=float, default=0.1, help='hyperparameter for distillation')
        parser.add_argument('-b', '--batch-size', type=int, default=8192, help='batch size')
        parser.add_argument('-c', '--clip', type=int)
        parser.add_argument('-d', '--decay', type=float, help='weight decay')
        parser.add_argument('-D', '--distribute', action='store_true', help='distribute dataset across GPUs')
        parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
        parser.add_argument('-E', '--ema', action='store_true', help='use Exponential Moving Average')
        parser.add_argument('-f', '--save-freq', type=int, help='frequency for saving model')
        parser.add_argument('-F', '--filter', type=int)
        parser.add_argument('-H', '--half', action='store_true', help='treat input data as half-precision (float16)')
        parser.add_argument('-L', '--logfile', default='train.log', help='log filename')
        parser.add_argument('-m', '--model', help='model checkpoint path')
        parser.add_argument('-r', '--learn-rate', type=float, default=1e-4, help='learning rate')
        parser.add_argument('-v', '--debug', action='store_true', help='verbose logging (DEBUG level)')
        parser.add_argument('-o', '--export', help='filename to export weights to, as C++ code')

        parser.add_argument('--gpu', dest='gpu', action='store_true', default=True, help='train on GPU')
        parser.add_argument('--no-gpu', dest='gpu', action='store_false')

        #for future support of other hot-encoding schemes
        parser.add_argument('--hot-encoding', choices=(769,), type=int, default=769, help=argparse.SUPPRESS)

        parser.add_argument('--logdir', default='/tmp/logs', help='tensorboard log dir')
        parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True, help='enable mixed precision')
        parser.add_argument('--name', help='optional model name')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true', default=False, help='use Nesterov momentum (SGD only)')
        parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
        parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
        parser.add_argument('--optimizer', choices=['adam', 'amsgrad', 'sgd'], default='amsgrad', help='optimization algorithm')
        parser.add_argument('--tensorboard', '-t', action='store_true', help='enable TensorBoard')
        parser.add_argument('--schedule', action='store_true', help='use learning rate schedule')

        args = parser.parse_args()
        if args.input[0] == 'export' and not args.export:
            args.export = sys.stdout

        log_level = configure_logging(args)

        # delay tensorflow import so that --help does not have to wait
        print('Importing TensorFlow')

        import tensorflow as tf
        tf.get_logger().setLevel(log_level)

        from tensorflow.keras.callbacks import ProgbarLogger
        from tensorflow.keras.constraints import MinMaxNorm
        from tensorflow.keras.layers import *
        from tensorflow.keras.losses import MeanAbsoluteError
        from tensorflow.keras.regularizers import L1L2

        # Detect GPU presence and compute capability.
        compute = 0

        if not args.gpu:
            # Force TensorFlow to place all operations on the CPU
            tf.config.set_soft_device_placement(True)
        else:
            if gpus := tf.config.list_physical_devices('GPU'):
                for gpu in gpus:
                    print(gpu)
                    cap = tf.config.experimental.get_device_details(gpu).get('compute_capability', None)
                    if cap != None:
                        logging.info(f'{gpu}: {cap}')
                        compute = max(compute, cap[0])
            else:
                args.gpu = False

        # The mixed_float16 policy specifies that TensorFlow should use a mix of float16 and float32
        # data types during training, with float32 being used for the activations and the parameters
        # of the model, and float16 being used for the intermediate computations. !!!! GPU Only !!!!
        if args.gpu and args.mixed_precision:
            from tensorflow.keras import mixed_precision
            if compute >= 7:
                mixed_precision.set_global_policy('mixed_float16')
                logging.info('Using mixed_float16 policy')
            else:
                mixed_precision.set_global_policy('float32')

        main(args)

    except KeyboardInterrupt:
        print()
        os._exit(0)
