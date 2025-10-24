#!/usr/bin/env python3
'''
**********************************************************************
Trainer for the Sturddle Chess 2.X engine's neural net.

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
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

ACCUMULATOR_SIZE = 1280
ATTN_FAN_OUT = 32
POOL_SIZE = 8

Q_SCALE = 1024

# Quantization range: use int16_t with Q_SCALE, prevent overflow
# 32 pieces + (16 + 16) occupancy + 1 side-to-move + 1 bias == 66
Q_MAX_A = 32767 / Q_SCALE / 66
Q_MIN_A = -Q_MAX_A

# (8 pawns + 1 king) x 2 + 1 bias == 19
Q_MAX_B = 32767  / Q_SCALE / 19
Q_MIN_B = -Q_MAX_B

SCALE = 100.0


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
        def __init__(self, qmin, qmax, quantize_round=args.quantize_round):
            self.qmin = qmin
            self.qmax = qmax
            self.quantize_round = quantize_round

        def __call__(self, w):
            w = tf.clip_by_value(w, self.qmin, self.qmax)

            if self.quantize_round:
                mask = tf.abs(w) < 1 / Q_SCALE
                w = tf.where(mask, tf.sign(w) / Q_SCALE, w)
                w = tf.round(w * Q_SCALE) / Q_SCALE

            return w

    @tf.function
    def soft_clip(x, clip_value, alpha=0.1):
        return (2 * tf.sigmoid(.5 * x) - 1) * clip_value + x * alpha

    @tf.function
    def combined_loss(y_true, y_pred):
        """Combine eval with game outcome (WDL) losses"""
        eval_target = y_true[:, 0:1]
        outcome_target = y_true[:, 1:2]
        sigmoid_scale = tf.constant(args.outcome_scale, dtype=tf.float32)

        # convert predicted and expected (target) eval scores to Win/Draw/Loss prob. scores
        wdl_eval_pred = tf.sigmoid(y_pred * SCALE / sigmoid_scale)
        # wdl_eval_target = tf.sigmoid(eval_target * SCALE / sigmoid_scale)

        # loss_eval = tf.reduce_mean(tf.square(wdl_eval_pred - wdl_eval_target))
        # loss_outcome = tf.reduce_mean(tf.square(wdl_eval_pred - outcome_target))

        # loss_eval = tf.reduce_mean(tf.abs(wdl_eval_pred - wdl_eval_target))
        # loss_outcome = tf.reduce_mean(tf.abs(wdl_eval_pred - outcome_target))

        # loss_eval = tf.keras.losses.binary_crossentropy(wdl_eval_target, wdl_eval_pred)
        loss_eval = tf.reduce_mean(tf.abs(y_pred - eval_target))
        loss_outcome = tf.keras.losses.binary_crossentropy(outcome_target, wdl_eval_pred)

        # blend the losses
        eval_weight = tf.constant(1.0 - args.outcome_weight, dtype=tf.float32)
        outcome_weight = tf.constant(args.outcome_weight, dtype=tf.float32)

        loss = loss_eval * eval_weight + loss_outcome * outcome_weight
        return loss


    class UnpackLayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs, **kwargs):
            super(UnpackLayer, self).__init__(**kwargs)
            self.num_outputs = num_outputs

        def call(self, packed):
            bitboards, turn = packed[:, :12], packed[:,-1:]

            f = tf.concat([tf_unpack_bits(bitboards), turn], axis=1)
            return tf.cast(f, tf.float32)

    with strategy.scope():
        ACTIVATION = tf.keras.activations.relu
        K_INIT = tf.keras.initializers.HeNormal

        # Define the input layer
        input_layer = Input(shape=(13,), dtype=tf.uint64, name='input')
        unpack_layer = UnpackLayer(args.hot_encoding, name='unpack')(input_layer)

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

        # Extract black occupation mask (summing black pieces' bitboards)
        black_occupied = Lambda(black_occupied_mask, name='black')(unpack_layer)
        # Extract white occupation mask (summing white pieces' bitboards)
        white_occupied = Lambda(white_occupied_mask, name='white')(unpack_layer)

        concat = Concatenate(name='features')([unpack_layer, black_occupied, white_occupied])

        constr_a = QConstraint(Q_MIN_A, Q_MAX_A)
        hidden_1a = Dense(
            ACCUMULATOR_SIZE,
            activation=ACTIVATION,
            name='hidden_1a',
            kernel_initializer=K_INIT,
            kernel_constraint=constr_a,
            bias_constraint=constr_a,
        )(concat)

        constr_b = QConstraint(Q_MIN_B, Q_MAX_B)

        # Define hidden layer 1b (use kings and pawns to modulate "main" network path)
        # hidden_1b_layer: selects the pawns and kings features.
        input_1b = Lambda(lambda x: x[:, :256], name='kings_and_pawns')(unpack_layer)
        hidden_1b = Dense(
            64,
            activation=ACTIVATION,
            name='hidden_1b',
            kernel_initializer=K_INIT,
            kernel_constraint=constr_b,
            bias_constraint=constr_b,
        )(input_1b)

        spatial_attn = Dense(ATTN_FAN_OUT, activation=None, name='spatial_attn')(hidden_1b)

        def custom_pooling(x):
            reshaped = tf.reshape(x, (-1, tf.shape(x)[1] // POOL_SIZE, POOL_SIZE))
            # Take the mean over the last dimension
            return tf.reduce_mean(reshaped, axis=-1)

        pooled = Lambda(custom_pooling, name='pool')(hidden_1a)

        # The "reshaping" layer repeats or tiles the dynamic weights to match the output shape of pooled
        attn_reshape_layer = Lambda(lambda x: tf.tile(x, tf.constant([1, ACCUMULATOR_SIZE // POOL_SIZE // ATTN_FAN_OUT])))

        # Compute spatial attention modulation
        modulation = Multiply(name='modulation')([pooled, attn_reshape_layer(spatial_attn)])

        # Add residual connection: pooled + pooled * attention_weights
        residual = Add(name='residual')([pooled, modulation])

        hidden_2 = Dense(16, activation=ACTIVATION, kernel_initializer=K_INIT, name='hidden_2')(residual)
        hidden_3 = Dense(16, activation=ACTIVATION, kernel_initializer=K_INIT, name='hidden_3')(hidden_2)

        # Define the position evaluation output
        eval_output = Dense(1, name='out', dtype='float32')(hidden_3)

        # Add move prediction heads if enabled
        outputs = [eval_output]

        if args.predict_moves:
            stop_grad = tf.stop_gradient(concat)

            # Output layer: 4096 logits for all possible moves (64x64)
            move_logits = Dense(
                4096,
                activation=None,  # Raw logits, no softmax
                # Use smaller initialization to prevent gradient explosion
                kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_constraint=constr_a,
                bias_constraint=constr_a,
                name='move',
                dtype='float32'
            )(stop_grad)

            outputs.append(move_logits)

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

        losses = {'out': combined_loss}
        metrics = {'out': [accuracy, mae]}
        loss_weights = {'out': 1.0}

        if args.predict_moves:
            """Experimental"""
            @tf.function
            def scaled_sparse_categorical_crossentropy(y_true, y_pred):
                """
                Scaled cross-entropy loss to prevent gradient explosion.
                Uses label smoothing and temperature scaling.
                """
                # y_true: move_indices

                # Apply temperature scaling to logits to reduce magnitude
                temperature = tf.constant(args.move_temperature, dtype=tf.float32)
                scaled_logits = y_pred / temperature

                # Clip to logits to prevent extreme values
                max_logit = tf.constant(args.move_logit_clip, dtype=tf.float32)
                clipped_logits = soft_clip(scaled_logits, -max_logit, max_logit)

                # Compute cross-entropy with label smoothing
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, clipped_logits, from_logits=True)

                # Scale down the loss to balance with position evaluation
                return loss * args.move_loss_scale

            @tf.function
            def top(y_true, y_pred, k=1):
                """Top-k accuracy for move prediction."""
                move_indices = tf.cast(y_true, tf.int32)
                return tf.keras.metrics.sparse_top_k_categorical_accuracy(
                    move_indices, y_pred, k=k
                )

            @tf.function
            def top_3(y_true, y_pred):
                return top(y_true, y_pred, k=3)

            @tf.function
            def top_5(y_true, y_pred):
                return top(y_true, y_pred, k=5)

            # Set up move prediction loss and metrics
            loss_weights['move'] = args.move_weight
            loss_weights['out'] = 1 - args.move_weight

            losses['move'] = scaled_sparse_categorical_crossentropy
            metrics['move'] = [top, top_3, top_5]

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


'''
Export weights as C++ code snippet.
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


def export_weights(args, model, indent=2):
    if args.export == sys.stdout:
        write_weigths(args, model, indent)
    else:
        with open(args.export, 'w+') as f:
            with redirect_stdout(f):
                print('#pragma once')
                print(f'// Generated from {args.model}')
                write_weigths(args, model, indent)


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

    # Flatten the isolated bits tensor
    # return tf.reshape(isolated_bits, [tf.shape(bitboards)[0], -1])
    return tf.reshape(isolated_bits, [-1, 12 * 64])


def dataset_from_file(args, filepath, strategy, callbacks):
    # Features are packed as np.uint64
    packed_feature_count = int(np.ceil(args.hot_encoding / 64))

    class BatchGenerator(tf.keras.utils.Sequence):
        def __init__(self, filepath, feature_count, batch_size):
            self.hf = h5py.File(filepath, 'r')
            self.data = self.hf['data']

            # Calculate the expected columns based on whether move prediction is enabled
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

            white_to_move = tf.equal(x[:,-1:], 1)  # Training data is from side-to-move POV

            # Get both evaluation and outcome data
            y_eval = self.data[start:end, self.feature_count:self.feature_count+1]
            y_eval = tf.cast(y_eval, tf.int64)  # Cast from unsigned to signed
            y_eval = tf.cast(y_eval, tf.float32) / SCALE  # Convert to float, and scale
            if args.clip_eval:
                y_eval = tf.clip_by_value(y_eval, -args.clip_eval/SCALE, args.clip_eval/SCALE)

            y_eval = tf.where(white_to_move, y_eval, -y_eval)  # Convert to White's perspective

            y_outcome = self.data[start:end, self.feature_count+1:self.feature_count+2]
            y_outcome = tf.cast(y_outcome, tf.float32) - 1.0  # Convert 0,1,2 -> -1,0,1
            # Convert from STM perspective to white's perspective
            y_outcome_white_pov = tf.where(white_to_move, y_outcome, -y_outcome)
            # Convert to win probability: -1->0.0, 0->0.5, 1->1.0
            y_outcome = (y_outcome_white_pov + 1.0) / 2.0

            y_outcome = y_outcome * (1 - args.outcome_smoothing) + 0.5 * args.outcome_smoothing

            # Combine both targets into a single tensor
            y_combined = tf.concat([y_eval, y_outcome], axis=1)  # Shape: (batch_size, 2)

            mask = None
            if args.balance:
                white_wins = np.where(y_outcome_white_pov > 0.5)[0]
                black_wins = np.where(y_outcome_white_pov < -0.5)[0]

                min_wins = min(len(white_wins), len(black_wins))

                mask = np.ones(end - start, dtype=bool)

                if len(white_wins) > min_wins:
                    keep = np.random.choice(white_wins, min_wins, replace=False)
                    drop = np.setdiff1d(white_wins, keep)
                    mask[drop] = False

                if len(black_wins) > min_wins:
                   keep = np.random.choice(black_wins, min_wins, replace=False)
                   drop = np.setdiff1d(black_wins, keep)
                   mask[drop] = False

            if args.no_capture and self.data.shape[1] > self.feature_count + 1:
                to_square = self.data[start:end, self.feature_count+3]

                # Compute occupied bitboards
                black_occupied = np.bitwise_or.reduce(x[:, 0:12:2], axis=1)
                white_occupied = np.bitwise_or.reduce(x[:, 1:12:2], axis=1)

                # Select opponent based on side to move
                stm = x[:, -1]
                # tf.assert_equal(stm, tf.cast(tf.squeeze(white_to_move, axis=1), tf.uint64))
                opponent_occupied = np.where(stm == 1, black_occupied, white_occupied)

                # Check if to_square is occupied by opponent
                to_square_mask = np.left_shift(np.uint64(1), to_square)
                is_capture = (np.bitwise_and(opponent_occupied, to_square_mask) != 0)

                mask = ~is_capture if mask is None else (mask & ~is_capture)

            if mask is not None:
                x = tf.boolean_mask(x, mask)
                y_combined = tf.boolean_mask(y_combined, mask)

            # Prepare outputs based on whether move prediction is enabled
            if args.predict_moves and self.data.shape[1] > self.feature_count + 1:
                # Get move coordinates (from_square, to_square) as indices
                from_square = self.data[start:end, self.feature_count+2]
                to_square = self.data[start:end, self.feature_count+3]

                # Convert from/to squares to move index (from_square * 64 + to_square)
                move_indices = from_square * 64 + to_square

                if mask is not None:
                    move_indices = tf.boolean_mask(move_indices, mask)

                # Reshape to match expected output shape
                move_indices = tf.reshape(move_indices, (-1, 1))

                # Return as tuple
                return x, (y_combined, move_indices)
            else:
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

                    # Add move prediction parameters if enabled
                    if args.predict_moves:
                        hyperparam.update({
                            'move_weight': args.move_weight,
                            'move_temperature': args.move_temperature,
                            'move_logit_clip': args.move_logit_clip,
                            'move_loss_scale': args.move_loss_scale,
                        })

                    # Log main loss if available
                    loss = logs.get('loss', math.nan) if logs else math.nan
                    logging.info(f'epoch={epoch} loss={loss:.6f} hyperparam={hyperparam}')

                    # Log additional metrics if available
                    if logs:
                        for key, value in logs.items():
                            if key != 'loss':
                                logging.info(f'epoch={epoch} {key}={value:.6f}')

            callbacks.append(CallbackOnEpochEnd(generator))

        # Determine output types and shapes based on whether move prediction is enabled
        if args.predict_moves:
            output_types = (
                np.uint64,
                (np.float32, np.float32)
            )
            output_shapes = (
                (None, packed_feature_count),
                ((None, 2), (None, 1))
            )
        else:
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
                if args.predict_moves:
                    combined_y = y[0]
                    eval_y = combined_y[:, 0:1]
                    outcome_y = combined_y[:, 1:2]
                else:
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
                if args.predict_moves:
                    filtered_y = tuple(tf.boolean_mask(y_item, condition) for y_item in y)
                else:
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
        'scaled_sparse_categorical_crossentropy': None,
        'top': None,
        'top_3': None,
        'top_5': None,
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
            # Layer doesn't exist in target model (e.g., move prediction layers)
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

    if alt_model:
        set_weights(alt_model, model)
        print(f'Applied alternate weights from {os.path.abspath(args.alt_model)}.')

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
        parser.add_argument('-c', '--clip-eval', type=int, help='clip eval target values [-CLIP,CLIP]')
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

        parser.add_argument('--balance', action='store_true', help='balance white / black wins inside batches')
        parser.add_argument('--hex', action='store_true', help='export weights in hex format')
        parser.add_argument('--no-capture', action='store_true', help='exclude captures from training')
        parser.add_argument('--no-draw', action='store_true', help='exclude draws from training')

        parser.add_argument('--outcome-weight', type=float, default=0.85, help='weight for outcome loss vs eval loss')
        parser.add_argument('--outcome-scale', type=float, default=400.0, help='scale factor for converting centipawns to win probability (sigmoid scaling)')

        # Move prediction related arguments
        parser.add_argument('--predict-moves', action='store_true', help='enable move prediction')
        parser.add_argument('--move-weight', type=float, default=0.3, help='blending weight for move prediction loss')

        # Arguments for move prediction stability
        parser.add_argument('--move-temperature', type=float, default=1.0, help='temperature scaling for move logits')
        parser.add_argument('--move-logit-clip', type=float, default=10.0, help='clip move logits to prevent extreme values')
        parser.add_argument('--move-loss-scale', type=float, default=0.1, help='scale factor for move prediction loss')
        parser.add_argument('--clip-norm', type=float, default=1.0, help='gradient clipping norm')

        parser.add_argument('--alt-model', help='Path to another model to load weights from')

        parser.add_argument('--gpu', dest='gpu', action='store_true', default=True, help='train on GPU')
        parser.add_argument('--no-gpu', dest='gpu', action='store_false')

        # For future support of other hot-encoding schemes
        parser.add_argument('--hot-encoding', choices=(769,), type=int, default=769, help=argparse.SUPPRESS)

        parser.add_argument('--logdir', default='/tmp/logs', help='tensorboard log dir')
        parser.add_argument('--max-queue-size', type=int, default=10000, help='max size for queue that holds batches')
        parser.add_argument('--mem-growth', action='store_true')
        parser.add_argument('--mem-limit', type=int, default=0, help='GPU memory limit in MB')
        parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True, help='enable mixed precision')
        parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
        parser.add_argument('--name', help='optional model name')
        parser.add_argument('--nesterov', dest='nesterov', action='store_true', default=False, help='use Nesterov momentum (SGD only)')
        parser.add_argument('--no-nesterov', dest='nesterov', action='store_false')
        parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false')
        parser.add_argument('--optimizer', choices=['adam', 'amsgrad', 'sgd'], default='amsgrad', help='optimization algorithm')
        parser.add_argument('--plot-file', help='plot model architecture to file')
        parser.add_argument('--sample', type=float, help='sampling ratio')
        parser.add_argument('--soft-alpha', type=float, default=0.01, help='alpha for soft_round operation')
        parser.add_argument('--tensorboard', '-t', action='store_true', help='enable TensorBoard logging callback')
        parser.add_argument('--schedule', action='store_true', help='use learning rate schedule')
        parser.add_argument('--validation', help='validation data filepath')
        parser.add_argument('--vfreq', type=int, default=1, help='validation frequency')
        parser.add_argument('--use-multiprocessing', action='store_true', help='enable multiprocessing for data loading')
        parser.add_argument('--workers', '-w', type=int, default=4, help='the number of worker threads for data loading')

        args = parser.parse_args()

        if args.outcome_weight < 0 or args.outcome_weight > 1:
            parser.error("--outcome-weight must be between 0 and 1 (inclusive)")

        # Validate move_weight
        if args.predict_moves and (args.move_weight <= 0 or args.move_weight >= 1):
            parser.error("--move-weight must be between 0 and 1 (exclusive)")

        # Validate new move prediction parameters
        if args.predict_moves:
            if args.move_temperature <= 0:
                parser.error("--move-temperature must be positive")
            if args.move_logit_clip <= 0:
                parser.error("--move-logit-clip must be positive")
            if args.move_loss_scale <= 0:
                parser.error("--move-loss-scale must be positive")

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
