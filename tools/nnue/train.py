#!/usr/bin/env python3
'''
**********************************************************************
Trainer for the Sturddle Chess 2.X engine's neural net.
NNUE-style architecture with king perspectives.
Copyright (c) 2023 - 2025 Cristian Vlasceanu.
**********************************************************************
'''
import argparse
import logging
import math
import os
import sys
from contextlib import redirect_stdout

import h5py
import numpy as np

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# uncomment (or set in environment) for newer TF versions (> 2.15.1 ?) that use Keras 3
# (and pip install tf-keras if needed)
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

ACCUMULATOR_SIZE = 256

Q16_SCALE = 255

# Quantization range: use int16_t with Q16_SCALE, prevent overflow
# 32 pieces + 1 bias == 33
Q16_MAX = 32767 / Q16_SCALE / 33

# clipped relu max value
Q16_CLIP = 127 / Q16_SCALE

Q8_SCALE = 64
Q8_MAX = 127 / Q8_SCALE

assert 1/Q16_SCALE < Q16_MAX
assert 1/Q8_SCALE < Q8_MAX

SCALE = 16.0  # Eval scale


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


def make_model(args, strategy):
    class QConstraint(tf.keras.constraints.Constraint):
        def __init__(self, qmax, qscale, quantize_round=args.quantize_round):
            self.qmin = -qmax
            self.qmax = qmax
            self.qscale = qscale
            self.quantize_round = quantize_round
            assert self.qmin < self.qmax
            assert 1 / self.qscale < self.qmax

        def __call__(self, w):
            w = tf.clip_by_value(w, self.qmin, self.qmax)

            if self.quantize_round:
                w = tf.round(w * self.qscale) / self.qscale

            return w

    @tf.function
    def combined_loss(y_true, y_pred):
        """Combine eval with game outcome (WDL) losses"""

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        eval_target = y_true[:, 0:1]
        outcome_target = y_true[:, 1:2]
        sigmoid_scale = tf.constant(args.outcome_scale, dtype=tf.float32)

        # Convert predicted and expected (target) eval scores to Win/Draw/Loss prob. scores
        wdl_eval_pred = tf.sigmoid(y_pred * SCALE / sigmoid_scale)
        wdl_eval_target = tf.sigmoid(eval_target * SCALE / sigmoid_scale)

        if args.mae:
            loss_eval = tf.reduce_mean(tf.abs(wdl_eval_pred - wdl_eval_target))
            loss_outcome = tf.reduce_mean(tf.abs(wdl_eval_pred - outcome_target))
        elif args.mse:
            loss_eval = tf.reduce_mean(tf.square(wdl_eval_pred - wdl_eval_target))
            loss_outcome = tf.reduce_mean(tf.square(wdl_eval_pred - outcome_target))
        elif args.bce:
            loss_eval = tf.keras.losses.binary_crossentropy(wdl_eval_target, wdl_eval_pred)
            loss_outcome = tf.keras.losses.binary_crossentropy(outcome_target, wdl_eval_pred)
        elif args.focal_bce:
            loss_outcome = tf.keras.losses.binary_focal_crossentropy(
                outcome_target, wdl_eval_pred, gamma=args.focal_gamma
            )
            loss_outcome = tf.reduce_mean(loss_outcome)
            # And now for something completely different: mix up eval space with WDL space:
            loss_eval = tf.keras.losses.huber(eval_target, y_pred, delta=args.huber_delta)
            loss_eval = tf.reduce_mean(loss_eval)
        else:
            assert False, "Loss function expected to be one of: --bce, --focal-bce, --mae, --mse"

        # Blend the losses
        eval_weight = tf.constant(1.0 - args.outcome_weight, dtype=tf.float32)
        outcome_weight = tf.constant(args.outcome_weight, dtype=tf.float32)

        loss = loss_eval * eval_weight + loss_outcome * outcome_weight
        return loss

    @tf.function
    def tf_unpack_bits(bitboards):
        # Create a tensor containing bit positions [63, 62, ..., 0]
        bit_positions = tf.constant(list(range(63, -1, -1)), dtype=tf.uint64)

        # Expand dimensions to make it broadcastable with bitboards
        bit_positions_exp = tf.reshape(bit_positions, [1, 1, 64])

        # Expand bitboards dimensions to [batch_size, tf.shape(bitboards)[1], 1]
        bitboards_exp = tf.expand_dims(bitboards, axis=-1)

        # Right shift bitboards by bit positions
        shifted = tf.bitwise.right_shift(bitboards_exp, bit_positions_exp)

        # Apply bitwise AND with 1 to isolate each bit
        isolated_bits = tf.bitwise.bitwise_and(shifted, 1)

        # Return shape (batch, 12, 64)
        return tf.cast(tf.reshape(isolated_bits, [tf.shape(bitboards)[0], 12, 64]), tf.float32)

    class Unpack(tf.keras.layers.Layer):
        def __init__(self, num_outputs, **kwargs):
            super(Unpack, self).__init__(**kwargs)
            self.num_outputs = num_outputs
            self.num_buckets = 32

            # Vertical mirror indices (rank flip: i ^ 56)
            self.vertical_mirror_indices = tf.constant([i ^ 56 for i in range(64)], dtype=tf.int32)

            # Horizontal mirror indices (file flip: i ^ 7)
            self.horizontal_mirror_indices = tf.constant([i ^ 7 for i in range(64)], dtype=tf.int32)

            # Color swap indices: swap pairs (0,1), (2,3), (4,5), etc.
            self.color_swap_indices = tf.constant([1,0, 3,2, 5,4, 7,6, 9,8, 11,10], dtype=tf.int32)

        @tf.function
        def call(self, packed):
            bitboards = packed[:, :12]
            unpacked = tf_unpack_bits(bitboards)

            black_king = unpacked[:, 0, :]
            white_king = unpacked[:, 1, :]

            black_king_sq = 63 - tf.argmax(black_king, axis=1, output_type=tf.int32)
            white_king_sq = 63 - tf.argmax(white_king, axis=1, output_type=tf.int32)

            pieces = unpacked[:, :12, :]
            batch_size = tf.shape(packed)[0]

            # === Black's perspective ===
            # Vertical mirror only (black already in friendly slots 0,2,4...)
            black_king_sq = black_king_sq ^ 56
            black_pieces = tf.gather(pieces, self.vertical_mirror_indices, axis=2)

            # === White's perspective ===
            # Color swap only (move white into friendly slots 0,2,4...)
            white_pieces = tf.gather(pieces, self.color_swap_indices, axis=1)

            # === Horizontal bucketing for black ===
            black_file = black_king_sq % 8
            black_rank = black_king_sq // 8
            black_needs_mirror = black_file < 4

            black_pieces = tf.where(
                tf.reshape(black_needs_mirror, [-1, 1, 1]),
                tf.gather(black_pieces, self.horizontal_mirror_indices, axis=2),
                black_pieces
            )

            black_file = tf.where(black_file < 4, 3 - black_file, black_file - 4)
            black_king_bucket = black_rank * 4 + black_file

            # === Horizontal bucketing for white ===
            white_file = white_king_sq % 8
            white_rank = white_king_sq // 8
            white_needs_mirror = white_file < 4

            white_pieces = tf.where(
                tf.reshape(white_needs_mirror, [-1, 1, 1]),
                tf.gather(white_pieces, self.horizontal_mirror_indices, axis=2),
                white_pieces
            )

            white_file = tf.where(white_file < 4, 3 - white_file, white_file - 4)
            white_king_bucket = white_rank * 4 + white_file

            # Flatten and compute perspectives
            black_pieces_flat = tf.reshape(black_pieces, [batch_size, -1])
            white_pieces_flat = tf.reshape(white_pieces, [batch_size, -1])

            black_king_onehot = tf.one_hot(black_king_bucket, depth=32, dtype=tf.float32)
            white_king_onehot = tf.one_hot(white_king_bucket, depth=32, dtype=tf.float32)

            black_pieces_exp = tf.expand_dims(black_pieces_flat, axis=1)
            white_pieces_exp = tf.expand_dims(white_pieces_flat, axis=1)

            black_king_exp = tf.expand_dims(black_king_onehot, axis=2)
            white_king_exp = tf.expand_dims(white_king_onehot, axis=2)

            output_size = self.num_buckets * 768
            black_perspective = tf.reshape(black_king_exp * black_pieces_exp, [batch_size, output_size])
            white_perspective = tf.reshape(white_king_exp * white_pieces_exp, [batch_size, output_size])

            return black_perspective, white_perspective

    class SelectPerspective(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(SelectPerspective, self).__init__(**kwargs)
            self.concat = Concatenate()

        @tf.function
        def call(self, inputs):
            stm_input, black_perspective, white_perspective = inputs

            # Extract side-to-move bit (last bit of input)
            stm = tf.bitwise.bitwise_and(stm_input[:, -1:], tf.constant(1, dtype=tf.uint64))

            # Stack both perspectives
            both_perspectives = tf.stack([black_perspective, white_perspective], axis=1)

            # Create indices based on stm
            stm_int = tf.cast(stm, tf.int32)
            indices_stm = tf.squeeze(stm_int, axis=-1)
            indices_opp = 1 - indices_stm

            # Gather
            batch_size = tf.shape(black_perspective)[0]
            batch_indices = tf.range(batch_size)

            stm_perspective = tf.gather_nd(both_perspectives, tf.stack([batch_indices, indices_stm], axis=1))
            opponent_perspective = tf.gather_nd(both_perspectives, tf.stack([batch_indices, indices_opp], axis=1))

            return self.concat([stm_perspective, opponent_perspective])

    with strategy.scope():
        input_layer = Input(shape=(13,), dtype=tf.uint64, name='input')
        unpack_layer = Unpack(args.hot_encoding, name='unpack')
        black_perspective_input, white_perspective_input = unpack_layer(input_layer)

        q_constr_1 = QConstraint(Q16_MAX, Q16_SCALE)
        K_INIT = tf.keras.initializers.HeNormal

        # Feature transformers
        black_perspective = Dense(
            ACCUMULATOR_SIZE,
            name='black_perspective',
            activation=tf.keras.layers.ReLU(max_value=Q16_CLIP, name='clip_relu_b'),
            kernel_initializer=K_INIT,
            kernel_constraint=q_constr_1,
            bias_constraint=q_constr_1,
        )(black_perspective_input)

        white_perspective = Dense(
            ACCUMULATOR_SIZE,
            name='white_perspective',
            activation=tf.keras.layers.ReLU(max_value=Q16_CLIP, name='clip_relu_w'),
            kernel_initializer=K_INIT,
            kernel_constraint=q_constr_1,
            bias_constraint=q_constr_1,
        )(white_perspective_input)

        concat = SelectPerspective(name='select_perspective')([input_layer, black_perspective, white_perspective])

        q_constr_2 = QConstraint(Q8_MAX, Q8_SCALE)
        hidden_2 = Dense(
            32,
            name='hidden_2',
            activation=tf.keras.activations.relu,
            kernel_initializer=K_INIT,
            kernel_constraint=q_constr_2,
            bias_constraint=q_constr_2,
        )(concat)

        hidden_3 = Dense(
            8,
            name='hidden_3',
            activation=tf.keras.activations.relu,
            kernel_initializer=K_INIT,
        )(hidden_2)

        eval_output = Dense(1, name='eval', dtype='float32')(hidden_3)

        outputs = [eval_output]

        # Create the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs, name=args.name)

        if args.optimizer in ['adam', 'amsgrad']:
            optimizer=tf.keras.optimizers.Adam(
                amsgrad=args.optimizer=='amsgrad',
                beta_1=0.99,
                beta_2=0.995,
                clipnorm=args.clip_norm,  # Gradient clipping
                learning_rate=args.learn_rate,
                use_ema=args.ema,
                weight_decay=args.decay if args.decay else None)
        elif args.optimizer == 'sgd':
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=args.clip_norm,  # Gradient clipping
                learning_rate=args.learn_rate,
                momentum=args.momentum,
                nesterov=args.nesterov,
                use_ema=args.ema,
                weight_decay=args.decay if args.decay else None)
        else:
            assert False

        if args.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        @tf.function
        def accuracy(y_true, y_pred):
            outcome_target = y_true[:, 1:2]  # Extract outcome component
            centipawns = y_pred * SCALE
            scale = tf.constant(args.outcome_scale, dtype=tf.float32)
            logits = centipawns / scale
            probs = tf.sigmoid(logits)
            mae = tf.reduce_mean(tf.abs(probs - outcome_target))
            accuracy_score = 1.0 - mae  # Convert to accuracy (higher is better)
            return accuracy_score

        @tf.function
        def mae(y_true, y_pred):
            eval_target = y_true[:, 0:1]  # Extract eval component
            return tf.keras.metrics.mean_absolute_error(eval_target, y_pred) * SCALE / 100

        losses = {'eval': combined_loss}
        metrics = {'eval': [accuracy, mae]}
        loss_weights = {'eval': 1.0}

        model.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics
        )

        # Log momentum
        optimizer = model.optimizer
        if hasattr(optimizer, '_optimizer'):
            optimizer = optimizer._optimizer
        if hasattr(optimizer, 'momentum'):
            logging.info(f'momentum: {optimizer.momentum}')

    return model


def get_layer_weights(layer):
    """Get layer weights, applying constraints if present."""
    params = layer.get_weights()
    if len(params) != 2:
        return None
    weights, biases = params
    if layer.kernel_constraint:
        weights = layer.kernel_constraint(weights).numpy()
    if layer.bias_constraint:
        biases = layer.bias_constraint(biases).numpy()
    return weights, biases


'''
Export weights as C++ code snippet.
'''
def write_weigths(args, model, indent=2):
    for layer in model.layers:
        params = get_layer_weights(layer)
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
                if args.hex:
                    print(f'{float(weights[i][j]).hex()}f,', end='')
                else:
                    print(f'{weights[i][j]:12.9f},', end='')
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
            if args.hex:
                print(f'{float(biases[i]).hex()}f,', end='')
            else:
                print(f'{biases[i]:12.9f},', end='')
        print('\n};')


def write_binary_weights(args, model, file):
    for layer in model.layers:
        weights = get_layer_weights(layer)
        if weights:
            kernel, bias = weights
            print(layer.name, kernel.shape, bias.shape)
            kernel.astype(np.float32).tofile(file)
            bias.astype(np.float32).tofile(file)


def export_weights(args, model):
    if args.bin:
        if args.export == sys.stdout:
            filename = f'{model.name}.bin'
        else:
            filename = args.export
        print(f'Exporting weights to: {filename}')
        with open(filename, 'wb') as file:
            write_binary_weights(args, model, file)

    elif args.export == sys.stdout:
        write_weigths(args, model)
    else:
        with open(args.export, 'w+') as f:
            with redirect_stdout(f):
                print('#pragma once')
                print(f'// Generated from {args.model}')
                write_weigths(args, model)


def load_binary_weights(args, model, file):
    """Load weights from a binary file into the model."""
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) == 2:
            kernel, bias = weights
            print(f"Loading {layer.name}: kernel {kernel.shape}, bias {bias.shape}")

            # Read kernel weights
            kernel_size = np.prod(kernel.shape)
            kernel_data = np.fromfile(file, dtype=np.float32, count=kernel_size)
            kernel_data = kernel_data.reshape(kernel.shape)

            # Read bias weights
            bias_size = np.prod(bias.shape)
            bias_data = np.fromfile(file, dtype=np.float32, count=bias_size)
            bias_data = bias_data.reshape(bias.shape)

            # Set the weights back to the layer
            layer.set_weights([kernel_data, bias_data])


def dataset_from_file(args, filepath, strategy, callbacks):
    # Features are packed as np.uint64
    packed_feature_count = int(np.ceil(args.hot_encoding / 64))

    class BatchGenerator(tf.keras.utils.Sequence):
        def __init__(self, filepath, feature_count, batch_size):
            self.hf = h5py.File(filepath, 'r')
            self.data = self.hf['data']

            expected_cols = feature_count + 4  # eval, outcome, from_square, to_square
            # Check data shape
            if self.data.shape[1] != expected_cols:
                raise ValueError("Invalid data format")

            self.feature_count = feature_count
            self.batch_size = batch_size
            self._num_batches = int(np.floor(len(self.data) / self.batch_size))  # drop incomplete batch
            if args.sample:
                self.sample_batches()
            else:
                self.indices = np.arange(self.num_batches)
                np.random.shuffle(self.indices)

            logging.info(f'using {len(self.indices)} batches.')

        @property
        def num_batches(self):
            return self._num_batches

        def __call__(self):
            return self

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            i = self.indices[index]
            start, end = i * self.batch_size, (i + 1) * self.batch_size

            # Get input features (bitboards)
            x = self.data[start:end, :self.feature_count]

            # Get both evaluation and outcome data
            y_eval = self.data[start:end, self.feature_count:self.feature_count+1]
            y_eval = tf.cast(y_eval, tf.int64)  # Cast from unsigned to signed
            y_eval = tf.cast(y_eval, tf.float32) / SCALE  # Convert to float, and scale

            y_outcome_np = self.data[start:end, self.feature_count+1:self.feature_count+2]  # Keep raw NumPy
            y_outcome = tf.cast(y_outcome_np, tf.float32) / 2.0  # Convert 0,1,2 -> 0.0,0.5,1.0
            y_outcome = y_outcome * (1 - args.outcome_smoothing) + 0.5 * args.outcome_smoothing

            mask = None

            # Apply capture filter first
            if args.no_capture and self.data.shape[1] > self.feature_count + 1:
                to_square = self.data[start:end, self.feature_count+3]

                # Compute occupied bitboards
                black_occupied = np.bitwise_or.reduce(x[:, 0:12:2], axis=1)
                white_occupied = np.bitwise_or.reduce(x[:, 1:12:2], axis=1)

                # Select opponent based on side to move
                stm = x[:, -1]

                opponent_occupied = np.where(stm == 1, black_occupied, white_occupied)

                # Check if to_square is occupied by opponent
                to_square_mask = np.left_shift(np.uint64(1), to_square)
                is_capture = (np.bitwise_and(opponent_occupied, to_square_mask) != 0)

                mask = ~is_capture

                x = x[mask]
                y_eval = y_eval[mask]
                y_outcome = y_outcome[mask]

            # Combine both targets into a single tensor
            y_combined = tf.concat([y_eval, y_outcome], axis=1)  # Shape: (batch_size, 2)

            return x, y_combined

        def rows(self):
            return self.data.shape[0]

        def on_epoch_end(self):
            if args.sample:
                self.sample_batches()
            else:
                np.random.shuffle(self.indices)

        def sample_batches(self):
            k = int(self.num_batches * args.sample)  # Round to integer
            self.indices = np.random.choice(self.num_batches, k, replace=False)

    print(f'Loading dataset {filepath}')  # begin reading the H5 file.

    generator = BatchGenerator(filepath, packed_feature_count, args.batch_size)
    print(f'{generator.rows():,} rows.')

    def make_dataset():
        if callbacks is not None:  # wire up the generator-defined callback
            class CallbackOnEpochEnd(tf.keras.callbacks.Callback):
                def __init__(self, generator):
                    super(CallbackOnEpochEnd, self).__init__()
                    self.generator = generator

                def on_epoch_end(self, epoch, logs=None):
                    self.generator.on_epoch_end()

                    # Log hyper-parameters
                    hyperparam = {
                        'batch size': args.batch_size,
                        'clip_norm': args.clip_norm,
                        'dataset size': f'{generator.rows():,}',
                        'filter': args.filter,
                        'learn rate': f'{self.model.optimizer.lr.read_value():.2e}',
                        'outcome_weight': args.outcome_weight,
                        'model': self.model.name,
                        'outcome_scale': args.outcome_scale,
                        'sampling ratio': args.sample,
                    }

                    # Log main loss if available
                    loss = logs.get('loss', math.nan) if logs else math.nan
                    logging.info(f'epoch={epoch} loss={loss:.6f} hyperparam={hyperparam}')

                    # Log additional metrics if available
                    if logs:
                        for key, value in logs.items():
                            if key != 'loss':
                                logging.info(f'epoch={epoch} {key}={value:.6f}')

            callbacks.append(CallbackOnEpochEnd(generator))

        output_types = (np.uint64, np.float32)
        output_shapes = ((None, packed_feature_count), (None, 2))

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes,
        )

        if args.filter:
            @tf.function
            def filter_data(x, y):
                eval_y = y[:, 0:1]
                outcome_y = y[:, 1:2]

                bound = args.filter / SCALE
                lower_bound = tf.greater(eval_y, -bound)
                upper_bound = tf.less(eval_y, bound)
                condition = tf.logical_and(lower_bound, upper_bound)

                if args.no_draw:
                    not_draw = tf.not_equal(outcome_y, 0.5)
                    condition = tf.logical_and(condition, not_draw)

                condition = tf.reshape(condition, [-1])  # Flatten to 1D

                # Apply mask to both input and all outputs
                filtered_x = tf.boolean_mask(x, condition)
                filtered_y = tf.boolean_mask(y, condition)

                return filtered_x, filtered_y

            dataset = dataset.map(filter_data, num_parallel_calls=tf.data.AUTOTUNE)

        if args.gpu:
            dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))

        dataset = dataset.prefetch(tf.data.AUTOTUNE).repeat()

        if args.distribute:
            # distribute data accross several GPUs
            dataset = strategy.experimental_distribute_dataset(dataset)

        return dataset

    return make_dataset(), len(generator)


def load_model(path):
    custom_objects = {
        'combined_loss': None,
    }

    return tf.keras.models.load_model(path, custom_objects=custom_objects)


def set_weights(from_model, to_model):
    for layer in from_model.layers:
        params = layer.get_weights()
        if not params:
            continue
        name = layer.name
        try:
            to_layer = to_model.get_layer(name)
        except ValueError:
            # Layer doesn't exist in target model
            logging.warning(f"Layer {name} not found in target model, skipping")
            continue

        if len(to_layer.get_weights()):
            to_layer.set_weights(params)


def main(args):
    if args.gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    alt_model = None
    if args.alt_model and os.path.exists(args.alt_model):
        alt_model = load_model(args.alt_model)

    if args.model and os.path.exists(args.model):
        saved_model = load_model(args.model)
        if not args.name:
            args.name = saved_model.name
        model = make_model(args, strategy)
        set_weights(saved_model, model)
        print(f'Loaded model {os.path.abspath(args.model)}.')
    else:
        model = make_model(args, strategy)

    if args.import_file:
        with open(args.import_file, 'rb') as file:
            load_binary_weights(args, model, file)

    if alt_model:
        set_weights(alt_model, model)
        print(f'Applied alternate weights from {os.path.abspath(args.alt_model)}.')

    if args.save_model:
        tf.keras.models.save_model(model, args.model)

    if args.plot_file:  # Display the model architecture
        tf.keras.utils.plot_model(
            model,
            to_file=args.plot_file,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
        )
    elif args.export:
        export_weights(args, model)
    else:
        callbacks = []
        dataset, steps_per_epoch = dataset_from_file(args, args.input[0], strategy, callbacks)

        if args.schedule:
            if os.environ.get('TF_USE_LEGACY_KERAS'):
                from tf_keras.callbacks import ReduceLROnPlateau
            else:
                from keras.callbacks import ReduceLROnPlateau

            lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-9)
            callbacks.append(lr)

        if args.model is not None:
            assert os.path.exists(os.path.dirname(args.model))

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

        model.summary(line_length=140)
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

        if args.validation:
            validation_data, _ = dataset_from_file(args, args.validation, strategy, None)
            model.fit(
                dataset,
                callbacks=callbacks,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                validation_freq=args.vfreq,
                max_queue_size=args.max_queue_size,
                workers=args.workers,
                use_multiprocessing=args.use_multiprocessing)
        else:
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model
            model.fit(
                dataset,
                callbacks=callbacks,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                max_queue_size=args.max_queue_size,
                workers=args.workers,
                use_multiprocessing=args.use_multiprocessing)

if __name__ == '__main__':
    try:
        class CustomFormatter(
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter
        ):
            pass

        parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
        parser.add_argument('input', nargs=1, help='memmap-ed numpy, or h5, input data file path')
        parser.add_argument('-b', '--batch-size', type=int, default=16384, help='batch size')
        parser.add_argument('-d', '--decay', type=float, help='weight decay')
        parser.add_argument('-D', '--distribute', action='store_true', help='distribute dataset across GPUs')
        parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('-E', '--ema', action='store_true', help='use Exponential Moving Average')
        parser.add_argument('-f', '--save-freq', type=int, help='frequency for saving model')
        parser.add_argument('-F', '--filter', type=int, help='filter out positions with absolute score above this value')
        parser.add_argument('-L', '--logfile', default='train.log', help='log filename')
        parser.add_argument('-m', '--model', help='model checkpoint path')
        parser.add_argument('-r', '--learn-rate', type=float, default=1e-4, help='learning rate')
        parser.add_argument('-v', '--debug', action='store_true', help='verbose logging (DEBUG level)')
        parser.add_argument('-o', '--export', help='filename to export weights to (in C++ header file format)')
        parser.add_argument('-q', '--quantize-round', action='store_true')
        parser.add_argument('-s', '--outcome-smoothing', type=float, default=0.025)

        parser.add_argument('--bce', action='store_true', default=False, help='use binary cross-entropy loss instead of MAE')
        parser.add_argument('--bin', action='store_true', default=True, help='export weights as binary file')
        parser.add_argument('--no-bin', action='store_false', dest='bin')

        parser.add_argument('--clip-norm', type=float, default=1.0, help='gradient clipping norm')

        # TODO: remove support for exporting weights as C++ text -- small nets legacy.
        parser.add_argument('--hex', action='store_true', help='export weights in hex format (no effect with --bin)')

        parser.add_argument('--import-file', help='import weights from binary file')
        parser.add_argument('--save-model', action='store_true')

        parser.add_argument('--no-capture', action='store_true', help='exclude captures from training')
        parser.add_argument('--no-draw', action='store_true', help='exclude draws from training')

        parser.add_argument('--outcome-weight', type=float, default=0.1, help='weight for outcome loss vs eval loss, 0: train on evals only, 1: outcome only')
        parser.add_argument('--outcome-scale', type=float, default=400.0, help='scale factor for converting centipawns to win probability (sigmoid scaling)')

        """Hack for overlaying and re-using compatible layers from alternate model when making architectual changes."""
        parser.add_argument('--alt-model', help='Path to another model to load (import) weights from')

        parser.add_argument('--focal-bce', action='store_true', help='use focal cross-entropy instead of MAE loss')
        parser.add_argument('--focal-gamma', type=float, default=2.0)
        parser.add_argument('--huber-delta', type=float, default=0.2)

        parser.add_argument('--gpu', dest='gpu', action='store_true', default=True, help='train on GPU')
        parser.add_argument('--no-gpu', dest='gpu', action='store_false')

        # For future support of other hot-encoding schemes
        parser.add_argument('--hot-encoding', choices=(769,), type=int, default=769, help=argparse.SUPPRESS)

        parser.add_argument('--logdir', default='/tmp/logs', help='tensorboard log dir')
        parser.add_argument('--max-queue-size', type=int, default=10000, help='max size for queue that holds batches')
        parser.add_argument('--mem-growth', action='store_true')
        parser.add_argument('--mem-limit', type=int, default=0, help='GPU memory limit in MB')
        parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True, help='enable mixed precision')
        parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
        parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
        parser.add_argument('--mae', action='store_true', default=True, help='use MAE loss')
        parser.add_argument('--mse', action='store_true', default=False, help='use MSE loss, instead of MAE')
        parser.add_argument('--name', help='optional model name')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true', default=False, help='use Nesterov momentum (SGD only)')
        parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
        parser.add_argument('--optimizer', choices=['adam', 'amsgrad', 'sgd'], default='amsgrad', help='optimization algorithm')
        parser.add_argument('--plot-file', help='plot model architecture to file')
        parser.add_argument('--sample', type=float, help='sampling ratio')
        parser.add_argument('--tensorboard', '-t', action='store_true', help='enable TensorBoard logging callback')
        parser.add_argument('--schedule', action='store_true', help='use learning rate schedule')
        parser.add_argument('--validation', help='validation data filepath')
        parser.add_argument('--vfreq', type=int, default=1, help='validation frequency')
        parser.add_argument('--use-multiprocessing', action='store_true', help='enable multiprocessing for data loading')
        parser.add_argument('--workers', '-w', type=int, default=4, help='the number of worker threads for data loading')

        args = parser.parse_args()

        if args.outcome_weight < 0 or args.outcome_weight > 1:
            parser.error("--outcome-weight must be between 0 and 1 (inclusive)")

        # Validate outcome scale parameter
        if args.outcome_scale <= 0:
            parser.error("--outcome-scale must be positive")

        if args.input[0] == 'export' and not args.export:
            args.export = sys.stdout

        if args.sample:
            args.sample = max(1e-3, min(1.0, args.sample))
            print(f'Sampling ratio={args.sample}')

        log_level = configure_logging(args)

        # delay tensorflow import so that --help does not have to wait
        print('Importing TensorFlow')

        import tensorflow as tf
        tf.get_logger().setLevel(log_level)

        print(f'TensorFlow version: {tf.__version__}')
        tf_ver = [int(v) for v in tf.__version__.split('.')]
        if tf_ver[0] >= 2 and tf_ver[1] > 12:
            from keras.src.saving.serialization_lib import SafeModeScope
        else:
            class SafeModeScope:  # fake, for compat with Tensorflow < 2.13.0
                def __init__(self, safe_mode=True):
                    pass
                def __enter__(self):
                    pass
                def __exit__(self, *_):
                    pass

        from tensorflow.keras.layers import *

        # Detect GPU presence and GPU compute capability.
        compute = 0

        if not args.gpu:
            # Force TensorFlow to place all operations on the CPU
            tf.config.set_soft_device_placement(True)
        else:
            if gpus := tf.config.list_physical_devices('GPU'):
                for gpu in gpus:
                    print(gpu)
                    if args.mem_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    if args.mem_limit > 0:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(memory_limit=args.mem_limit)]
                        )
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
                args.mixed_precision = False
                mixed_precision.set_global_policy('float32')
        else:
            args.mixed_precision = False

        main(args)

    except KeyboardInterrupt:
        print()
        os._exit(0)
