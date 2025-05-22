#!/usr/bin/env python3
'''
**********************************************************************
Trainer for the Sturddle Chess 2.0 engine's neural net.

Copyright (c) 2023 - 2025 Cristian Vlasceanu.

Expects H5 files as inputs (produced by toh5.py)
Uses a custom layer to unpack features, which allows unpacking on GPU.
**********************************************************************
'''
import argparse
import logging
import math
import os
import re
import sys
from contextlib import redirect_stdout

import h5py
import numpy as np

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Q_SCALE = 1024
# Quantization range: use int16_t with Q_SCALE, prevent overflow
Q_MAX = 32767 / Q_SCALE / 130
Q_MIN = -32768 / Q_SCALE / 130

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
    class CustomConstraint(tf.keras.constraints.Constraint):
        def __init__(self, qmin=Q_MIN, qmax=Q_MAX):
            self.qmin = qmin
            self.qmax = qmax

        def __call__(self, w):
            return tf.clip_by_value(w, self.qmin, self.qmax)

    @tf.function
    def soft_clip(x, clip_value, alpha=0.1):
        return (2 * tf.math.sigmoid(.5 * x) - 1) * clip_value + x * alpha

    @tf.function
    def clipped_loss(y_true, y_pred, delta=args.clip):
        error = soft_clip(y_true - y_pred, delta)
        squared_loss = 0.5 * tf.square(error)
        linear_loss  = delta * tf.abs(error) - 0.5 * delta**2
        return tf.where(tf.abs(error) < delta, squared_loss, linear_loss)

    # @tf.function
    # def clipped_loss(y_true, y_pred, delta=args.clip):
    #     error = soft_clip(y_true - y_pred, delta)
    #     squared_loss = 0.5 * tf.square(error)
    #     linear_loss = delta * tf.abs(error) - 0.5 * delta**2
    #     loss_values = tf.where(tf.abs(error) < delta, squared_loss, linear_loss)
    #     tf.print("=== Position Eval Loss Debug ===")
    #     tf.print("y_true range:", tf.reduce_min(y_true), "to", tf.reduce_max(y_true))
    #     tf.print("y_pred range:", tf.reduce_min(y_pred), "to", tf.reduce_max(y_pred))
    #     tf.print("error range:", tf.reduce_min(error), "to", tf.reduce_max(error))
    #     tf.print("loss_values range:", tf.reduce_min(loss_values), "to", tf.reduce_max(loss_values))
    #     tf.print("loss_values mean:", tf.reduce_mean(loss_values))
    #     tf.print("================================")
    #     return loss_values

    class UnpackLayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs, **kwargs):
            super(UnpackLayer, self).__init__(**kwargs)
            self.num_outputs = num_outputs

        def call(self, packed):
            bitboards, turn = packed[:, :12], packed[:,-1:]

            f = tf.concat([tf_unpack_bits(bitboards), turn], axis=1)
            return tf.cast(f, tf.float32)

    if args.quantization:
        class FixedScaleQuantizer(quantizers.Quantizer):
            def build(self, tensor_shape, name, layer):
                return {}  # No new TensorFlow variables needed.

            @tf.function
            def __call__(self, inputs, training, weights, **kwargs):
                half_range = 32768  # 16-bit quantization
                alpha = tf.cast(args.soft_alpha, inputs.dtype)

                quantized_values = tfc.ops.soft_round(inputs * Q_SCALE, alpha)
                clipped_values = tf.keras.backend.clip(quantized_values, -half_range, half_range - 1)

                return clipped_values / Q_SCALE

            def get_config(self):
                return {}

        '''
        https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide.md
        '''
        class CustomQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
            # Return a list of tuple, each of which is:
            # (weight_variable, quantizer_function)
            def get_weights_and_quantizers(self, layer):
                return [(layer.kernel, FixedScaleQuantizer())]

            # Return a list of tuple, each of which is:
            # (activation_output, quantizer_function)
            def get_activations_and_quantizers(self, layer):
                return [(layer.activation, FixedScaleQuantizer())]

            # Given quantized weights, set the weights of the layer.
            def set_quantize_weights(self, layer, quantized_weights):
                layer.kernel = quantized_weights[0]

            # Given quantized activations, set the activations of the layer.
            def set_quantize_activations(self, layer, quantize_activations):
                layer.activation = quantize_activations[0]

            def get_output_quantizers(self, layer):
                return [FixedScaleQuantizer()]

            def get_config(self):
                return {}

            @classmethod
            def from_config(cls, config):
                return cls()

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
        constr = CustomConstraint()
        pool_size = 4
        hidden_1a_inputs = 640

        hidden_1a_layer = Dense(
            hidden_1a_inputs,
            activation=ACTIVATION,
            name='hidden_1a',
            kernel_initializer=K_INIT,
            kernel_constraint=constr,
            bias_constraint=constr
        )

        # Define hidden layer 1b (use kings and pawns to compute dynamic weights)
        # hidden_1b_layer: selects the pawns and kings features.
        input_1b = Lambda(lambda x: x[:, :256], name='kings_and_pawns')(unpack_layer)
        hidden_1b_layer = Dense(
            64,
            activation=ACTIVATION,
            name='hidden_1b',
            kernel_initializer=K_INIT,
            kernel_constraint=constr,
            bias_constraint=constr
        )

        # Add "attention layer" that computes dynamic weights based on hidden_1b (pawns & kings features).
        attn_fan_out = int(args.attn)
        attention_layer = Dense(attn_fan_out, activation=None, name='dynamic_weights')

        # Add 2nd hidden layer
        hidden_2_layer = Dense(16, activation=ACTIVATION, kernel_initializer=K_INIT, name='hidden_2')

        # ... and 3rd
        hidden_3_layer = Dense(16, activation=ACTIVATION, kernel_initializer=K_INIT, name='hidden_3')

        if args.quantization:
            quantization_config = CustomQuantizeConfig()
            hidden_1a_layer, hidden_1b_layer = (
                tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=quantization_config)
                for layer in [hidden_1a_layer, hidden_1b_layer]
            )

        hidden_1a = hidden_1a_layer(concat)
        hidden_1b = hidden_1b_layer(input_1b)

        dynamic_weights = attention_layer(hidden_1b)  # computes dynamic weights

        def custom_pooling(x):
            reshaped = tf.reshape(x, (-1, tf.shape(x)[1] // pool_size, pool_size))
            # Take the max over the last dimension
            return tf.reduce_mean(reshaped, axis=-1)

        pooled = Lambda(custom_pooling, name='pool')(hidden_1a)

        # The "reshaping" layer repeats or tiles the dynamic weights to match the output shape of pooled
        attn_reshape_layer = Lambda(lambda x: tf.tile(x, tf.constant([1, hidden_1a_inputs // pool_size // attn_fan_out])))

        # Apply weights to pooled (multiply pooled output with dynamic weights)
        weighted = Multiply(name='weighted')([pooled, attn_reshape_layer(dynamic_weights)])

        hidden_2 = hidden_2_layer(weighted)
        hidden_3 = hidden_3_layer(hidden_2)  # 3rd hidden layer

        # Define the position evaluation output (original output)
        eval_output = Dense(1, name='out', dtype='float32')(hidden_3)

        # Add move prediction heads if enabled
        outputs = [eval_output]

        if args.predict_moves:
            moves = Dense(
                128,
                activation=ACTIVATION,
                kernel_initializer=K_INIT,
                name='moves',
            )(hidden_1a)

            moves_ref = Dense(32, activation=ACTIVATION, kernel_initializer=K_INIT, name='moves_ref')(moves)

            # Output layers predicting values in 0-1 range
            from_square = Dense(1,
                       kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01),
                       bias_initializer=tf.keras.initializers.Constant(0.3),
                       name='F', dtype='float32')(moves_ref)

            to_square = Dense(1,
                     kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01),
                     bias_initializer=tf.keras.initializers.Constant(0.7),
                     name='T', dtype='float32')(moves_ref)

            outputs.extend([from_square, to_square])

        # Create the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs, name=args.name)

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
                momentum=args.momentum,
                nesterov=args.nesterov,
                use_ema=args.ema,
                weight_decay=args.decay if args.decay else None)
        else:
            assert False

        if args.quantization:
            with SafeModeScope(safe_mode=False), tfmot.quantization.keras.quantize_scope({
                'CustomConstraint': CustomConstraint,
                'CustomQuantizeConfig': CustomQuantizeConfig,
                'FixedScaleQuantizer': FixedScaleQuantizer,
                'UnpackLayer': UnpackLayer,
            }):
                model = tfmot.quantization.keras.quantize_apply(model)

        # Create loss dictionary
        losses = {'out': clipped_loss}
        loss_weights = {'out': 1.0}
        metrics = {}

        if args.predict_moves:
            @tf.function
            def manhattan_distance(y_true, y_pred):
                # Convert from 0-1 range to 0-63 for chess board coordinates
                true_idx = y_true * 63.0
                pred_idx = y_pred * 63.0

                # Convert to 2D board coordinates (file, rank)
                true_file = true_idx % 8
                true_rank = tf.floor(true_idx / 8)
                pred_file = pred_idx % 8
                pred_rank = tf.floor(pred_idx / 8)

                return tf.abs(true_file - pred_file) + tf.abs(true_rank - pred_rank)

            @tf.function
            def manhattan_loss(y_true, y_pred):
                return tf.square(manhattan_distance(y_true, y_pred))

            @tf.function
            def chess_move_loss(y_true, y_pred):
                out_of_bounds = tf.logical_or(y_pred < 0.0, y_pred > 1.0)
                loss = tf.where(out_of_bounds,  tf.ones_like(y_pred) * 500, manhattan_loss(y_true, y_pred))
                return loss / 10 # scale down so clipped_loss is not overwhelmed

            # @tf.function
            # def chess_move_loss(y_true, y_pred):
            #     out_of_bounds = tf.logical_or(y_pred < 0.0, y_pred > 1.0)
            #     manhattan_loss_vals = manhattan_loss(y_true, y_pred)
            #     loss_values = tf.where(out_of_bounds, tf.ones_like(y_pred) * 1000, manhattan_loss_vals)
            #     tf.print("=== Move Loss Debug ===")
            #     tf.print("y_true range:", tf.reduce_min(y_true), "to", tf.reduce_max(y_true))
            #     tf.print("y_pred range:", tf.reduce_min(y_pred), "to", tf.reduce_max(y_pred))
            #     tf.print("out_of_bounds count:", tf.reduce_sum(tf.cast(out_of_bounds, tf.int32)), "out of", tf.size(y_pred))
            #     tf.print("manhattan_loss range:", tf.reduce_min(manhattan_loss_vals), "to", tf.reduce_max(manhattan_loss_vals))
            #     tf.print("final loss range:", tf.reduce_min(loss_values), "to", tf.reduce_max(loss_values))
            #     tf.print("final loss mean:", tf.reduce_mean(loss_values))
            #     tf.print("=======================")
            #     return loss_values

            @tf.function
            def top_k(y_true, y_pred, k=1):
                # Consider prediction correct if within k squares (Manhattan distance)
                dist = manhattan_distance(y_true, y_pred)
                correct = tf.less_equal(dist, tf.cast(k, tf.float32))
                return tf.reduce_mean(tf.cast(correct, tf.float32))

            loss_weights['F'] = args.move_weight / 2
            loss_weights['T'] = args.move_weight / 2
            loss_weights['out'] = 1 - args.move_weight

            losses['F'] = chess_move_loss
            losses['T'] = chess_move_loss

            metrics['F'] = top_k
            metrics['T'] = top_k

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


def dataset_from_file(args, filepath, clip, strategy, callbacks):
    # Features are packed as np.uint64
    packed_feature_count = int(np.ceil(args.hot_encoding / 64))

    class BatchGenerator(tf.keras.utils.Sequence):
        def __init__(self, filepath, feature_count, batch_size):
            self.hf = h5py.File(filepath, 'r')
            self.data = self.hf['data']

            # Calculate the expected columns based on whether move prediction is enabled
            expected_cols = feature_count + 1
            if args.predict_moves:
                expected_cols += 2  # Add columns for from_square, to_square

            # Check data shape
            if self.data.shape[1] != expected_cols:
                if args.predict_moves and self.data.shape[1] == feature_count + 1:
                    raise ValueError("Move prediction enabled but training data doesn't include move coordinates. "
                                    f"Expected {expected_cols} columns, got {self.data.shape[1]}")
                elif not args.predict_moves and self.data.shape[1] > feature_count + 1:
                    logging.warning("Training data includes move coordinates but move prediction is disabled.")

            self.feature_count = feature_count
            self.batch_size = batch_size
            self.num_batches = int(np.floor(len(self.data) / self.batch_size))  # drop incomplete batch
            if args.sample:
                self.sample_batches()
            else:
                self.indices = np.arange(self.num_batches)
                np.random.shuffle(self.indices)

            logging.info(f'using {len(self.indices)} batches.')

        def __call__(self):
            return self

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            i = self.indices[index]
            start, end = i * self.batch_size, (i + 1) * self.batch_size

            # Get input features (bitboards)
            x = self.data[start:end, :self.feature_count]

            # Get position evaluation
            y_eval = self.data[start:end, self.feature_count:self.feature_count+1]
            y_eval = tf.cast(y_eval, tf.int64)  # Cast from unsigned to signed
            y_eval = tf.cast(y_eval, tf.float32) / SCALE  # Convert to float, and scale

            white_to_move = tf.equal(x[:,-1:], 1)  # Training data is from side-to-move POV
            y_eval = tf.where(white_to_move, y_eval, -y_eval)  # Convert to White's perspective

            # Prepare outputs based on whether move prediction is enabled
            if args.predict_moves and self.data.shape[1] > self.feature_count + 1:
                # Get move coordinates (from_square, to_square) as indices (NOT one-hot)
                from_square = self.data[start:end, self.feature_count+1]
                to_square = self.data[start:end, self.feature_count+2]

                # Scale values from 0-63 to 0-1 range
                from_square = tf.cast(from_square, tf.float32) / 63.0
                to_square = tf.cast(to_square, tf.float32) / 63.0

                # Reshape to match expected output shape
                from_square = tf.reshape(from_square, (-1, 1))
                to_square = tf.reshape(to_square, (-1, 1))

                # Return as tuple
                return x, (y_eval, from_square, to_square)
            else:
                return x, y_eval

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
                        'clip': args.clip,
                        'dataset size': f'{generator.rows():,}',
                        'filter': args.filter,
                        'learn rate': f'{self.model.optimizer.lr.read_value():.2e}',
                        'model': self.model.name,
                        'sampling ratio': args.sample,
                    }

                    # Add move prediction parameters if enabled
                    if args.predict_moves:
                        hyperparam['move_weight'] = args.move_weight

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
                (np.float32, np.float32, np.float32)
            )
            output_shapes = (
                (None, packed_feature_count),
                ((None, 1), (None, 1), (None, 1))
            )
        else:
            output_types = (np.uint64, np.float32)
            output_shapes = ((None, packed_feature_count), (None, 1))

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes,
        )

        if args.filter:
            @tf.function
            def filter(x, y):
                if args.predict_moves:
                    eval_y = y[0]
                    bound = args.filter / SCALE
                    lower_bound = tf.greater(eval_y, -bound)
                    upper_bound = tf.less(eval_y, bound)
                    condition = tf.logical_and(lower_bound, upper_bound)
                    condition = tf.reshape(condition, [-1])  # Flatten to 1D

                    # Apply mask to both input and all outputs
                    filtered_x = tf.boolean_mask(x, condition)
                    filtered_y = tuple(tf.boolean_mask(y_item, condition) for y_item in y)
                    return filtered_x, filtered_y
                else:
                    bound = args.filter / SCALE
                    lower_bound = tf.greater(y, -bound)
                    upper_bound = tf.less(y, bound)
                    condition = tf.logical_and(lower_bound, upper_bound)
                    condition = tf.reshape(condition, [-1])  # Flatten to 1D
                    return tf.boolean_mask(x, condition), tf.boolean_mask(y, condition)

            dataset = dataset.map(filter, num_parallel_calls=tf.data.AUTOTUNE)

        if args.gpu:
            dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))

        dataset = dataset.prefetch(tf.data.AUTOTUNE).repeat()

        if args.distribute:
            # distribute data accross several GPUs
            dataset = strategy.experimental_distribute_dataset(dataset)

        return dataset

    return make_dataset(), len(generator)


def load_model(path):
    ''' Load model ignoring missing loss function. '''
    custom_objects = { 'chess_move_loss': None, 'top_k': None }

    while True:
        try:
            return tf.keras.models.load_model(path, custom_objects=custom_objects)
        except ValueError as e:
            match = re.search(r'Unknown loss function: \'(\w+)\'.*', str(e))
            if match:
                missing_object = match.group(1)
                custom_objects[missing_object.strip()] = None
                continue
            raise


def set_weights(from_model, to_model):
    q_prefix = 'quant_'
    for layer in from_model.layers:
        params = layer.get_weights()
        if not params:
            continue
        name = layer.name
        if name.startswith(q_prefix):
            # quantized -> plain
            name = name[len(q_prefix):]
            params = params[:-1]
        try:
            to_layer = to_model.get_layer(name)
        except ValueError:
            try:
                # plain -> quantized
                to_layer = to_model.get_layer(q_prefix + name)
                params.append(np.empty(shape=()))
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

    if args.model and os.path.exists(args.model):
        saved_model = load_model(args.model)
        if not args.name:
            args.name = saved_model.name
        model = make_model(args, strategy)
        set_weights(saved_model, model)
        print(f'Loaded model {os.path.abspath(args.model)}.')
    else:
        model = make_model(args, strategy)

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
        dataset, steps_per_epoch = dataset_from_file(args, args.input[0], args.clip, strategy, callbacks)

        if args.schedule:
            from keras.callbacks import ReduceLROnPlateau
            lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-12)
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
        parser.add_argument('-b', '--batch-size', type=int, default=8192, help='batch size')
        parser.add_argument('-c', '--clip', type=float, default=3.0, help='Huber delta (used to be hard clip)')
        parser.add_argument('-d', '--decay', type=float, help='weight decay')
        parser.add_argument('-D', '--distribute', action='store_true', help='distribute dataset across GPUs')
        parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
        parser.add_argument('-E', '--ema', action='store_true', help='use Exponential Moving Average')
        parser.add_argument('-f', '--save-freq', type=int, help='frequency for saving model')
        parser.add_argument('-F', '--filter', type=int, default=3000, help='hard clip (filter out) scores with higher abs value')
        parser.add_argument('-L', '--logfile', default='train.log', help='log filename')
        parser.add_argument('-m', '--model', help='model checkpoint path')
        parser.add_argument('-r', '--learn-rate', type=float, default=1e-3, help='learning rate')
        parser.add_argument('-v', '--debug', action='store_true', help='verbose logging (DEBUG level)')
        parser.add_argument('-o', '--export', help='filename to export weights to, as C++ code')

        # Move prediction related arguments
        parser.add_argument('--predict-moves', action='store_true', help='enable move prediction')
        parser.add_argument('--move-weight', type=float, default=0.5, help='weight for move prediction loss')

        parser.add_argument('--attn', choices=('16', '32'), default='32', help='attention layer size')
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
        parser.add_argument('--quantization', '-q', action='store_true', help='simulate quantization effects during training')
        parser.add_argument('--sample', type=float, help='sampling ratio')
        parser.add_argument('--soft-alpha', type=float, default=0.01, help='alpha for soft_round operation')
        parser.add_argument('--tensorboard', '-t', action='store_true', help='enable TensorBoard logging callback')
        parser.add_argument('--schedule', action='store_true', help='use learning rate schedule')
        parser.add_argument('--validation', help='validation data filepath')
        parser.add_argument('--vfreq', type=int, default=1, help='validation frequency')
        parser.add_argument('--use-multiprocessing', action='store_true', help='(experimental)')
        parser.add_argument('--workers', '-w', type=int, default=4, help='(experimental)')

        args = parser.parse_args()

        # Validate move_weight
        if args.predict_moves and args.move_weight <= 0 or args.move_weight >= 1:
            parser.error("--move-weight must be between 0 and 1 (exclusive)")

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

        if args.quantization:
            # Experimental only
            import tensorflow_compression as tfc
            import tensorflow_model_optimization as tfmot
            from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantizers

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

        # Turn off mixed mode when using quantization-aware training,
        # due to issues with tfc.ops.soft_round during graph construction.

        if args.gpu and args.mixed_precision and not args.quantization:
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
