#! /usr/bin/env python3
'''
Trainer for the Sturddle Chess 2.0 engine's neural network.
Copyright (c) 2023 Cristian Vlasceanu.

Expects memmapped numpy arrays as inputs.
'''
import argparse
import logging
import os
import sys
from contextlib import redirect_stdout

import numpy as np

 # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    def _clipped_relu(x):
        return tf.keras.activations.relu(x, max_value=1.)

    activation = tf.keras.activations.relu if args.activation == 'relu' else _clipped_relu
    with strategy.scope():
        model = tf.keras.models.Sequential([
            Dense(128, input_shape=(args.hot_encoding,), activation=activation, name='hidden'),
            Dense(1, name='out', dtype='float32')
        ])

        if args.loss == 'mse':
            loss = MeanSquaredError()
        elif args.loss == 'huber':
            loss = Huber(delta=args.delta)
        elif args.loss == 'rmse':
            @tf.function
            def rmse(y_true, y_pred):
                return tf.keras.losses.mean_squared_error(y_true, y_pred)

            loss = rmse
        else:
            assert False

        if args.optimizer == 'adam':
            optimizer=tf.keras.optimizers.Adam(
                amsgrad=args.amsgrad,
                learning_rate=args.learn_rate,
                use_ema=args.ema,
                weight_decay=args.decay if args.decay else None)
        elif args.optimizer == 'sgd':
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=args.learn_rate,
                nesterov=True,
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
        weights, biases = layer.get_weights()
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


def main(args):
    '''
    Batch generator.
    '''
    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.batch_size = args.batch_size if args.batch_size else 1
            self.len = int(np.ceil(len(self.x) / self.batch_size))
            logging.info(f'using {self.len} batches.')

        def __call__(self):
            return self

        def __len__(self):
            return self.len

        def __getitem__(self, i):
            x = self.x[i * self.batch_size:(i + 1) * self.batch_size]
            y = self.y[i * self.batch_size:(i + 1) * self.batch_size]
            if args.clip:
                y = np.clip(y, -args.clip, args.clip)
            return x, y

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
            logging.info(f'using {self.len} batches.')

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            x_macro_batch = self.x[index * self.macro_batch_size:(index + 1) * self.macro_batch_size]
            y_macro_batch = self.y[index * self.macro_batch_size:(index + 1) * self.macro_batch_size]
            return DataGenerator(x_macro_batch, y_macro_batch)


    if args.gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    if not args.export:
        print('Loading dataset')
        dtype = np.float16 if args.half else np.float32
        data = np.memmap(*args.input, dtype=dtype, mode='r')
        row_count = data.shape[0] // (args.hot_encoding + 1)
        data = data.reshape((row_count, (args.hot_encoding + 1)))
        x = data[:,:args.hot_encoding]
        y = data[:,args.hot_encoding:]
        print(x.shape, y.shape)

        steps_per_epoch = None

        if args.whole_dataset:
            # attempt to fit whole training set in memory
            #dataset = tf.data.Dataset.from_tensor_slices((x, y))
            #if args.batch_size:
            #    dataset = dataset.batch(args.batch_size)
            dataset = None

        elif args.distribute:
            # distribute data accross several GPUs
            dataset = DataGenerator(x, y)
            steps_per_epoch = len(dataset)
            dataset = tf.data.Dataset.from_generator(
                dataset,
                output_types=(dtype, dtype),
                output_shapes=((None, args.hot_encoding), (None, 1)),
            ).prefetch(tf.data.AUTOTUNE).repeat()

            dataset = strategy.experimental_distribute_dataset(dataset)

        elif args.macro_batch_size > 0:
            # use macro-batching (chunking) to reduce I/O latency
            dataset = MacroBatchGenerator(x, y)

        else:
            dataset = DataGenerator(x, y)

    if args.model and os.path.exists(args.model):
        model = _make_model(args, strategy)
        model.set_weights(tf.keras.models.load_model(args.model).get_weights())
        print(f'Loaded model {os.path.abspath(args.model)}.')
    else:
        model = _make_model(args, strategy)

    if args.export:
        export_weights(args, model)
    else:
        if args.model is None:
            callbacks = []
        else:
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
            callbacks = [model_checkpoint_callback]

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
            print(f'Training with {row_count} rows of data.')

            if args.profile_batches:
                log_dir = '/tmp/logs'
                profile = (1, args.profile_batches)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=profile)
                callbacks = callbacks + [tensorboard_callback]

            if dataset:
                if isinstance(dataset, MacroBatchGenerator):
                    loss = []
                    for era in range(args.epochs // args.macro_epochs):
                        logging.info(f'===== Era: {era} =====')

                        indices = np.arange(len(dataset))

                        np.random.shuffle(indices)
                        logging.info(f'indices: {indices}')

                        for i,j in enumerate(indices):
                            logging.info(f'MacroBatch: {i + 1}/{len(indices)}: {j}')
                            model.fit(dataset[j], callbacks=callbacks, epochs=args.macro_epochs)

                else:
                    # https://www.tensorflow.org/api_docs/python/tf/keras/Model
                    model.fit(dataset,
                        callbacks=callbacks,
                        epochs=args.epochs,
                        steps_per_epoch=steps_per_epoch)
            else:
                model.fit(x, y, callbacks=callbacks, epochs=args.epochs)


if __name__ == '__main__':
    try:
        class CustomFormatter(
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter
        ):
            pass
        parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
        parser.add_argument('input', nargs=1, help='memmap-ed numpy input data')
        parser.add_argument('-a', '--arch', type=int, default=6, help='network architecture')
        parser.add_argument('-b', '--batch-size', type=int, default=8192, help='batch size')
        parser.add_argument('-d', '--delta', type=float, default=1.35, help='Huber delta')
        parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
        parser.add_argument('-f', '--save_freq', type=int, help='frequency for saving model')
        parser.add_argument('-i', '--infer', type=int, default=0)
        parser.add_argument('-l', '--loss', choices=['huber', 'mse', 'rmse'], default='mse', help='loss function')
        parser.add_argument('-L', '--logfile', default='train.log', help='log filename')
        parser.add_argument('-m', '--model', help='model checkpoint path')
        parser.add_argument('-r', '--learn-rate', type=float, default=0.0001, help='learning rate')
        parser.add_argument('-v', '--debug', action='store_true', help='use verbose (DEBUG level) logging')
        parser.add_argument('-w', '--export', help='filename to export weights to, as C++ code')
        parser.add_argument('--activation', choices=['clipped-relu', 'relu'], default='relu', help='activation function')
        parser.add_argument('--amsgrad', action='store_true', help='use amsgrad (ignored when not using adam)')
        parser.add_argument('--clip', type=int)
        parser.add_argument('--decay', type=float, help='weight decay')
        parser.add_argument('--distribute', action='store_true', help='distribute dataset between GPUs')
        parser.add_argument('--ema', action='store_true', help='use Exponential Moving Average')
        parser.add_argument('--half', action='store_true', help='read half-precision (float16) input')
        parser.add_argument('--gpu', dest='gpu', action='store_true', default=True, help='train on GPU')
        parser.add_argument('--no-gpu', dest='gpu', action='store_false')
        parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd')

        #for future support of other hot-encoding schemes
        parser.add_argument('--hot-encoding', choices=(769,), type=int, default=769, help=argparse.SUPPRESS)

        parser.add_argument('--macro-batch-size', type=int, default=0)
        parser.add_argument('--macro-epochs', type=int, default=20, help='epochs per macro-batch')
        parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True, help='enable mixed precision')
        parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
        parser.add_argument('--profile-batches', type=int, default=0, help='enable TensorBoard to profile range of batches')
        parser.add_argument('--whole-dataset', action='store_true', help='attempt to fit whole dataset in memory')

        args = parser.parse_args()
        if args.input[0] == 'export' and not args.export:
            args.export = sys.stdout

        log_level = _configure_logging(args)

        # delay tensorflow import so that --help does not have to wait
        print('Importing tensorflow')

        import tensorflow as tf
        tf.get_logger().setLevel(log_level)

        from tensorflow.keras.layers import *
        from tensorflow.keras.losses import Huber, MeanSquaredError

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
