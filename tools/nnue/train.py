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
import h5py
from contextlib import redirect_stdout
import numpy as np

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Quantization range: use int16_t with QSCALE of 1024, and need to add 18 values
(16 weights, 1 bias, 1 residual) without overflow, max representable value is 32767 / 18 / 1024
'''
MAX_QVAL = 1.7777
MIN_QVAL = -MAX_QVAL

def _configure_logging(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename=args.logfile,
        format='%(asctime)s;%(levelname)s;%(message)s',
        level=log_level,
    )
    # silence off annoying logging, see https://github.com/abseil/abseil-py/issues/99
    logging.getLogger('absl').addFilter(lambda *_:False)
    return log_level


def _make_model(args, strategy):
    @tf.function
    def _clipped_mae(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, -args.clip, args.clip)
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    tf.keras.utils.get_custom_objects().update({'_clipped_mae': _clipped_mae})

    with strategy.scope():
        activation = tf.keras.activations.relu

        # Define the input layer
        input_layer = Input(shape=(args.hot_encoding,), name='input')

        # Define layer 1a
        hidden_1a = Dense(
            512,
            activation=activation,
            name='hidden_1a',
            kernel_constraint=MinMaxNorm(min_value=MIN_QVAL, max_value=MAX_QVAL),
            bias_constraint=MinMaxNorm(min_value=MIN_QVAL, max_value=MAX_QVAL),
        )(input_layer)

        # Add 2nd hidden layer
        hidden_2 = Dense(16, activation=activation, name='hidden_2')(hidden_1a)

        # Define hidden layer 1b (use kings and pawns to compute dynamic weights)
        input_1b = Lambda(lambda x: x[:, :256], name='slice_input_1b')(input_layer)
        hidden_1b = Dense(
            64,
            activation=activation,
            name='hidden_1b',
            kernel_constraint=MinMaxNorm(min_value=MIN_QVAL, max_value=MAX_QVAL),
            bias_constraint=MinMaxNorm(min_value=MIN_QVAL, max_value=MAX_QVAL),
        )(input_1b)

        # Compute dynamic weights based on hidden_1b
        dynamic_weights = Dense(
            16,
            activation=None,
            name='dynamic_weights',
            #kernel_regularizer=L1L2(l1=2.5*1e-5, l2=5*1e-4),
            #bias_regularizer=L1L2(l1=2.5*1e-5, l2=5*1e-4),
        )(hidden_1b)

        # Apply dynamic weights to hidden_2
        weighted_hidden_2 = Multiply(name='weighted_hidden_2')([hidden_2, dynamic_weights])

        # Define the output layer
        output_layer = Dense(1, name='out', dtype='float32')(weighted_hidden_2)

        # Create the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name=args.name)

        if args.clip:
            loss = _clipped_mae
        else:
            loss = MeanAbsoluteError()

        if args.optimizer == 'adam':
            optimizer=tf.keras.optimizers.Adam(
                amsgrad=args.amsgrad,
                beta_1=0.995,
                beta_2=0.9995,
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

        model.compile(loss=loss, optimizer=optimizer, metrics=[])

    return model


'''
Export weights as C++ code snippet.

The numpy.float32 data type in NumPy uses the 32-bit floating-point
format defined by the IEEE 754 standard. According to the standard,
the maximum number of decimal digits of precision that can be
represented by a numpy.float32 value is 7.
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


def dataset_from_file(args, filepath, clip, strategy, callbacks):
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

    '''
    Split training data into macro batches (chunks).
    Using macro batching may reduce I/O latency.
    '''
    class MacroBatchGenerator(tf.keras.utils.Sequence):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.macro_batch_size = args.macro_batch_size
            self.len = int(np.ceil(len(self.x) / self.macro_batch_size))
            logging.info(f'using {self.len} macro-batches.')

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            x_macro_batch = self.x[index * self.macro_batch_size:(index + 1) * self.macro_batch_size]
            y_macro_batch = self.y[index * self.macro_batch_size:(index + 1) * self.macro_batch_size]
            return DataGenerator(x_macro_batch, y_macro_batch)

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
        prefetch_batches = tf.data.AUTOTUNE

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(dtype, dtype),
            output_shapes=((None, args.hot_encoding), (None, 1)),
        ).prefetch(prefetch_batches).repeat()

        return dataset, steps_per_epoch


    print(f'Loading dataset {filepath}')
    dtype = np.float16 if args.half else np.float32

    if os.path.splitext(filepath)[1].lower() == '.h5':
        f = h5py.File(filepath)
        data = f['eval']
        row_count = data.shape[0]
        assert data.shape[1] == args.hot_encoding + 1, data.shape[1]
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
    elif args.macro_batch_size > 0:
        # use macro-batching (chunking) to reduce I/O latency
        dataset = MacroBatchGenerator(x, y)
    else:
        dataset, steps_per_epoch = make_dataset(x, y)

    return dataset, steps_per_epoch


def main(args):
    if args.gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    if args.model and os.path.exists(args.model):
        saved_model = tf.keras.models.load_model(args.model, custom_objects={'_clipped_mae' : None})
        if not args.name:
            args.name = saved_model.name
        model = _make_model(args, strategy)
        model.set_weights(saved_model.get_weights())
        print(f'Loaded model {os.path.abspath(args.model)}.')
    else:
        model = _make_model(args, strategy)

    if args.export:
        export_weights(args, model)
    else:
        callbacks = []
        dataset, steps_per_epoch = dataset_from_file(args, args.input[0], args.clip, strategy, callbacks)

        if args.schedule_lr:
            from keras.callbacks import ReduceLROnPlateau
            lr = ReduceLROnPlateau(
                monitor='loss', factor=0.2, patience=5, min_lr=0.000001)
            callbacks.append(lr)

        if args.model is not None:
            assert os.path.exists(os.path.dirname(args.model))

            if args.macro_batch_size > 0:
                save_best_only = False
            else:
                save_best_only = not bool(args.save_freq)

            # https://keras.io/api/callbacks/model_checkpoint/
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                args.model,
                monitor='loss',
                mode='min',
                save_best_only=save_best_only,
                save_freq=args.save_freq if args.save_freq else 'epoch',
            )
            callbacks.append(model_checkpoint_callback)

        if args.infer > 0:
            # Test inference.
            x_test, y_test = x[:args.infer], y[:args.infer]
            for u, v in zip(model(x_test, training=False), y_test):
                print(u.numpy(), v)

            print(x_test.shape)
            predictions = model.predict(x_test)
            # Compute the confidence matrix
            confidence_matrix = np.zeros((args.hot_encoding, args.hot_encoding))
            for i in range(len(x_test)):
                true_value = y_test[i]
                predicted_value = predictions[i][0]
                error = true_value - predicted_value
                confidence_matrix += np.outer(error, error)

            # Normalize the confidence matrix
            confidence_matrix /= len(x_test)

            # Print the confidence matrix
            print(confidence_matrix)
        else:
            model.summary()
            if not args.model:
                print('*****************************************************************')
                print(' WARNING: checkpoint path not provided, model WILL NOT BE SAVED! ')
                print('*****************************************************************')

            if args.tensorboard:
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=args.logdir,
                    update_freq=args.save_freq if args.save_freq else 'epoch',
                    profile_batch=(1, steps_per_epoch)
                )
                callbacks.append(tensorboard_callback)

            assert dataset

            if args.macro_batch_size:
                # Validation data is not supported in chunk mode.
                # H5 files not well supported either (may run out of memory).
                for era in range(args.epochs // args.macro_epochs):
                    logging.info(f'===== Era: {era} =====')
                    indices = np.arange(len(dataset))
                    np.random.shuffle(indices)
                    logging.info(f'indices: {indices}')
                    for i,j in enumerate(indices):
                        logging.info(f'MacroBatch: {i + 1}/{len(indices)}: {j}')
                        model.fit(
                            dataset[j],
                            callbacks=callbacks,
                            epochs=args.macro_epochs,
                            max_queue_size=args.max_queue_size,
                            workers=args.workers,
                            use_multiprocessing=args.use_multiprocessing,
                        )

            elif args.validation:
                validation_data, _ = dataset_from_file(args, args.validation, None, strategy, None)
                model.fit(
                    dataset,
                    callbacks=callbacks,
                    epochs=args.epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_data,
                    validation_freq=args.vfreq,
                    max_queue_size=args.max_queue_size,
                    workers=args.workers,
                    use_multiprocessing=args.use_multiprocessing,
                )
            else:
                # https://www.tensorflow.org/api_docs/python/tf/keras/Model
                model.fit(
                    dataset,
                    callbacks=callbacks,
                    epochs=args.epochs,
                    steps_per_epoch=steps_per_epoch,
                    max_queue_size=args.max_queue_size,
                    workers=args.workers,
                    use_multiprocessing=args.use_multiprocessing,
                )

if __name__ == '__main__':
    try:
        class CustomFormatter(
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter
        ):
            pass
        parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
        parser.add_argument('input', nargs=1, help='memmap-ed numpy, or h5, input data file path')
        parser.add_argument('-b', '--batch-size', type=int, default=8192, help='batch size')
        parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
        parser.add_argument('-f', '--save_freq', type=int, help='frequency for saving model')
        parser.add_argument('-i', '--infer', type=int, default=0, help='test inference on specified number of examples')
        parser.add_argument('-L', '--logfile', default='train.log', help='log filename')
        parser.add_argument('-m', '--model', help='model checkpoint path')
        parser.add_argument('-r', '--learn-rate', type=float, default=0.0001, help='learning rate')
        parser.add_argument('-v', '--debug', action='store_true', help='use verbose (DEBUG level) logging')
        parser.add_argument('-o', '--export', help='filename to export weights to, as C++ code')
        parser.add_argument('--amsgrad', action='store_true', help='use amsgrad (ignored when not using adam)')
        parser.add_argument('--clip', type=int)
        parser.add_argument('--decay', type=float, help='weight decay')
        parser.add_argument('--distribute', action='store_true', help='distribute dataset between GPUs')
        parser.add_argument('--ema', action='store_true', help='use Exponential Moving Average')
        parser.add_argument('--half', action='store_true', help='read half-precision (float16) input')
        parser.add_argument('--gpu', dest='gpu', action='store_true', default=True, help='train on GPU')
        parser.add_argument('--no-gpu', dest='gpu', action='store_false')

        #for future support of other hot-encoding schemes
        parser.add_argument('--hot-encoding', choices=(769,), type=int, default=769, help=argparse.SUPPRESS)

        parser.add_argument('--logdir', default='/tmp/logs', help='tensorboard log dir')
        parser.add_argument('--macro-batch-size', type=int, default=0)
        parser.add_argument('--macro-epochs', type=int, default=1, help='epochs per macro-batch')
        parser.add_argument('--max-queue-size', type=int, default=10000, help='max size for queue that holds batches')
        parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True, help='enable mixed precision')
        parser.add_argument('--name', help='optional model name')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true', default=False, help='use Nesterov momentum (SGD only)')
        parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
        parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
        parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
        parser.add_argument('--tensorboard', '-tb', action='store_true', help='enable TensorBoard')
        parser.add_argument('--schedule-lr', action='store_true', help='use learning rate schedule')
        parser.add_argument('--validation', help='validation data filepath')
        parser.add_argument('--vfreq', type=int, default=1, help='validation frequency')
        parser.add_argument('--use-multiprocessing', action='store_true')
        parser.add_argument('--workers', type=int, default=16)

        args = parser.parse_args()
        if args.input[0] == 'export' and not args.export:
            args.export = sys.stdout

        log_level = _configure_logging(args)

        # delay tensorflow import so that --help does not have to wait
        print('Importing TensorFlow')

        import tensorflow as tf
        tf.get_logger().setLevel(log_level)

        from tensorflow.keras.constraints import MinMaxNorm
        from tensorflow.keras.layers import *
        from tensorflow.keras.losses import Huber, MeanAbsoluteError, MeanSquaredError
        from tensorflow.keras.regularizers import L1L2

        '''
        Detect GPU presence and compute capability.
        '''
        compute = 0

        if not args.gpu:
            # force TensorFlow to place all operations on the CPU
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

        '''
        The mixed_float16 policy specifies that TensorFlow should use a mix of float16 and float32
        data types during training, with float32 being used for the activations and the parameters
        of the model, and float16 being used for the intermediate computations. !!!! GPU Only !!!!
        '''
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
