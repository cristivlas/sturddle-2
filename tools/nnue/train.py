#!/usr/bin/env python3
'''
**********************************************************************
Trainer for the Sturddle Chess 2.X engine's neural net.
Copyright (c) 2023 - 2026 Cristian Vlasceanu.
**********************************************************************
'''
import argparse
import json
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

ACCUMULATOR_SIZE = 2048
POOL_SIZE = 8
MAIN_BUCKETS = 16  # Number of buckets for hidden_1a / BucketShift (4 pawn x 4 king-file)

Q_SCALE = 1024

# Quantization range: use int16_t with Q_SCALE, prevent overflow
# 32 pieces + 1 side-to-move + 1 bias == 34
Q_MAX_A = 32767 / Q_SCALE / 34
Q_MIN_A = -Q_MAX_A

# (8 pawns + 1 king) x 2 + 1 bias == 19
Q_MAX_B = 32767  / Q_SCALE / 19
Q_MIN_B = -Q_MAX_B

SCALE = 100.0

# Square color masks for OCB detection
LIGHT_SQUARES = np.uint64(0x55AA55AA55AA55AA)
DARK_SQUARES = np.uint64(0xAA55AA55AA55AA55)

# Squares on files e-h, for king-file bucketing
FILES_EFGH = np.uint64(0xF0F0F0F0F0F0F0F0)


def scale_buckets(x):
    """Per-sample pawn x king-file bucket ids, same scheme as BucketShift / nnue.h get_bucket."""
    pawns = popcount(x[:, 2]) + popcount(x[:, 3])
    pawn_id = np.where(pawns <= 4, 0, np.minimum((pawns - 1) // 4, 3))
    wk_right = (x[:, 1] & FILES_EFGH) != 0
    bk_right = (x[:, 0] & FILES_EFGH) != 0
    return (pawn_id * 4 + wk_right * 2 + bk_right).astype(np.int64)


def load_profile(path):
    """Load per-bucket label scale ratios (dataset profile); eval labels are divided by these."""
    with open(path) as f:
        profile = json.load(f)
    ratios = profile['ratios'] if isinstance(profile, dict) else profile
    if not isinstance(ratios, list) or len(ratios) != MAIN_BUCKETS:
        raise ValueError(f'{path}: expected a list of {MAIN_BUCKETS} bucket ratios')
    ratios = np.array(ratios, dtype=np.float32)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError(f'{path}: ratios must be positive and finite')
    logging.info('profile %s: %s', path, ' '.join(f'{r:.3f}' for r in ratios))
    return ratios


def member_profile_table(data, filepath, args):
    """Per-member label-scale profiles: (member start rows, (members x buckets) ratio matrix).

    --profile overrides everything. Otherwise each member of a virtual dataset
    (or the file itself if not virtual) uses the <member>.profile.json sidecar
    when present, falling back to the container's sidecar, then to all-ones.
    Resolves one level, not recursively.
    """
    if args.profile_ratios is not None:
        return np.zeros(1, dtype=np.int64), args.profile_ratios[np.newaxis, :]

    if data.is_virtual:
        members = sorted((vs.vspace.get_select_bounds()[0][0], vs.file_name) for vs in data.virtual_sources())
        members = [(start, filepath if path == '.' else path) for start, path in members]
    else:
        members = [(0, filepath)]

    # members without their own sidecar fall back to the container's, then to all-ones
    default = np.ones(MAIN_BUCKETS, dtype=np.float32)
    own_sidecar = filepath + '.profile.json'
    if os.path.exists(own_sidecar):
        default = load_profile(own_sidecar)
        print(f'{filepath}: default profile {own_sidecar}')

    cache = {}
    profiles = []
    for start, path in members:
        sidecar = path + '.profile.json'
        if sidecar not in cache:
            if path != filepath and os.path.exists(sidecar):
                cache[sidecar] = load_profile(sidecar)
                print(f'{path}: using profile {sidecar}')
            else:
                cache[sidecar] = default
        profiles.append(cache[sidecar])

    print(f'{len(members)} member(s), {sum(os.path.exists(p) for p in cache)} profile(s)')
    return np.array([m[0] for m in members], dtype=np.int64), np.stack(profiles)


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
            if self.quantize_round:
                w = tf.round(w * Q_SCALE) / Q_SCALE
            w = tf.clip_by_value(w, self.qmin, self.qmax)
            return w

    @tf.function
    def soft_clip(x, clip_value):
        ALPHA = 0.1
        clipped = tf.clip_by_value(x, -clip_value, clip_value)
        overflow = x - clipped
        return clipped + ALPHA * overflow

    @tf.function
    def combined_loss(y_true, y_pred):
        """Combine eval with game outcome (WDL) losses"""

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        eval_target = y_true[:, 0:1] / y_true[:, 3:4]  # normalize label scale per bucket

        if args.clip_eval:
            clipped = soft_clip(eval_target, args.clip_eval / SCALE);
            # tf.print("\nEval:", eval_target, "\nClipped:", clipped)
            eval_target = clipped

        outcome_target = y_true[:, 1:2]
        piece_ratio = y_true[:, 2:3]  # 0 to 1, for dynamic outcome weighting

        sigmoid_scale = tf.constant(args.outcome_scale, dtype=tf.float32)

        # convert predicted and expected (target) eval scores to Win/Draw/Loss prob. scores
        wdl_eval_pred = tf.sigmoid(y_pred * SCALE / sigmoid_scale)
        wdl_eval_target = tf.sigmoid(eval_target * SCALE / sigmoid_scale)

        # Compute per-sample losses
        if args.loss_mae:
            loss_eval = tf.abs(wdl_eval_pred - wdl_eval_target)
            loss_outcome = tf.abs(wdl_eval_pred - outcome_target)
        elif args.loss_bce:
            loss_eval = tf.keras.losses.binary_crossentropy(wdl_eval_target, wdl_eval_pred)[:, tf.newaxis]
            loss_outcome = tf.keras.losses.binary_crossentropy(outcome_target, wdl_eval_pred)[:, tf.newaxis]
        elif args.loss_blend:
            # Huber on raw eval domain; BCE on outcome in probability domain
            huber_delta = tf.constant(args.huber_delta, dtype=tf.float32)
            err = y_pred - eval_target
            abs_err = tf.abs(err)
            loss_eval = tf.where(
                abs_err <= huber_delta,
                0.5 * tf.square(err),
                huber_delta * (abs_err - 0.5 * huber_delta)
            )
            loss_outcome = tf.keras.losses.binary_crossentropy(outcome_target, wdl_eval_pred)[:, tf.newaxis]
        else:
            # default: mean square error
            loss_eval = tf.square(wdl_eval_pred - wdl_eval_target)
            loss_outcome = tf.square(wdl_eval_pred - outcome_target)

        # Blend the losses with dynamic per-sample weighting based on piece count
        base_outcome_weight = tf.constant(args.outcome_weight, dtype=tf.float32)
        if args.dynamic_outcome_weight:
            # More pieces -> trust outcome more (eval less reliable in complex positions)
            outcome_weight = base_outcome_weight * piece_ratio
        else:
            outcome_weight = base_outcome_weight
        eval_weight = 1.0 - outcome_weight

        loss = loss_eval * eval_weight + loss_outcome * outcome_weight
        return tf.reduce_mean(loss)


    class Unpack(tf.keras.layers.Layer):
        def __init__(self, num_outputs, **kwargs):
            super(Unpack, self).__init__(**kwargs)
            self.num_outputs = num_outputs

        def call(self, packed):
            bitboards, turn = packed[:, :12], packed[:,-1:]

            f = tf.concat([tf_unpack_bits(bitboards), turn], axis=1)
            return tf.cast(f, tf.float32)

    class BucketShift(tf.keras.layers.Layer):
        def __init__(self, num_buckets, **kwargs):
            super(BucketShift, self).__init__(**kwargs)
            self.num_buckets = num_buckets

        def call(self, features):
            # Bitboards are encoded [black, white] per piece (black first).
            # Each bitboard unpacks to 64 bits, so:
            # - Piece 0 (black king):  features[:, 0:64]
            # - Piece 1 (white king):  features[:, 64:128]
            # - Piece 2 (black pawns): features[:, 128:192]
            # - Piece 3 (white pawns): features[:, 192:256]
            pawn_bits = features[:, 128:256]  # Shape: (batch, 128)

            # Count total pawns on the board
            pawn_count = tf.reduce_sum(tf.cast(pawn_bits, tf.float32), axis=1)

            # Pawn dimension: fat bucket 0 spans {0,1,2,3,4} pawns; every bucket above spans 4 pawns.
            pawn_id = tf.cast(
                tf.where(
                    pawn_count <= 4.0,
                    tf.zeros_like(pawn_count),
                    tf.minimum((pawn_count - 1.0) // 4.0, 3.0),
                ),
                tf.int32
            )

            # King-file dimension: board split into left (files a-d) / right (files e-h).
            # Feature index idx within a 64-block holds bitboard bit (63 - idx); file = bit % 8.
            right_mask = tf.constant([1.0 if ((63 - idx) % 8) >= 4 else 0.0 for idx in range(64)], dtype=tf.float32)
            black_king = tf.cast(features[:, 0:64], tf.float32)
            white_king = tf.cast(features[:, 64:128], tf.float32)
            wk_right = tf.cast(tf.reduce_sum(white_king * right_mask, axis=1), tf.int32)
            bk_right = tf.cast(tf.reduce_sum(black_king * right_mask, axis=1), tf.int32)
            king_id = wk_right * 2 + bk_right

            # Compose: pawn dimension (4) x king-file dimension (4) = 16 buckets.
            bucket_id = pawn_id * 4 + king_id

            # tf.print("\nPawn count:", pawn_count, "\nBucket id:", bucket_id)

            # Shift features into N buckets
            num_features = tf.shape(features)[1]
            bucket_mask = tf.one_hot(bucket_id, self.num_buckets, dtype=features.dtype)
            bucket_mask = tf.tile(tf.expand_dims(bucket_mask, 2), [1, 1, num_features])

            features_tiled = tf.tile(tf.expand_dims(features, 1), [1, self.num_buckets, 1])
            sparse = features_tiled * bucket_mask

            return tf.reshape(sparse, [-1, self.num_buckets * num_features])


    with strategy.scope():
        ACTIVATION = tf.keras.activations.relu
        K_INIT = tf.keras.initializers.HeNormal

        # Define the input layer
        input_layer = Input(shape=(13,), dtype=tf.uint64, name='input')
        unpack_layer = Unpack(args.hot_encoding, name='unpack')(input_layer)

        # Apply bucketing
        bucketed = BucketShift(MAIN_BUCKETS, name='bucket_shift')(unpack_layer)

        constr_a = QConstraint(Q_MIN_A, Q_MAX_A)
        hidden_1a = Dense(
            ACCUMULATOR_SIZE,
            activation=ACTIVATION,
            name='hidden_1a',
            kernel_initializer=K_INIT,
            kernel_constraint=constr_a,
            bias_constraint=constr_a,
            trainable=not args.freeze_eval,
        )(bucketed)

        constr_b = QConstraint(Q_MIN_B, Q_MAX_B)

        # Hidden layer 1b (kings and pawns) directly modulates the pooled main path.
        # Output width matches pooled (ACCUMULATOR_SIZE / POOL_SIZE) for 1:1 modulation.
        input_1b = Lambda(lambda x: x[:, :256], name='kings_and_pawns')(unpack_layer)
        hidden_1b = Dense(
            ACCUMULATOR_SIZE // POOL_SIZE,
            activation=None,
            name='hidden_1b',
            kernel_initializer=K_INIT,
            kernel_constraint=constr_b,
            bias_constraint=constr_b,
            trainable=not args.freeze_eval,
        )(input_1b)

        def custom_pooling(x):
            reshaped = tf.reshape(x, (-1, tf.shape(x)[1] // POOL_SIZE, POOL_SIZE))
            # Take the mean over the last dimension
            return tf.reduce_mean(reshaped, axis=-1)

        pooled = Lambda(custom_pooling, name='pool')(hidden_1a)

        # Modulate pooled by 1b: pooled * (1 + h1b)
        modulation = Multiply(name='modulation')([pooled, hidden_1b])
        residual = Add(name='residual')([pooled, modulation])

        hidden_2 = Dense(
            16,
            activation=ACTIVATION,
            kernel_initializer=K_INIT,
            name='hidden_2',
            trainable=not args.freeze_eval,
        )(residual)

        hidden_3 = Dense(
            16,
            activation=ACTIVATION,
            kernel_initializer=K_INIT,
            name='hidden_3',
            trainable=not args.freeze_eval,
        )(hidden_2)

        eval_output = Dense(1, name='out', dtype='float32', trainable=not args.freeze_eval)(hidden_3)

        # Add move prediction heads if enabled
        outputs = [eval_output]

        if args.predict_moves:
            stop_grad = tf.stop_gradient(unpack_layer)

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
            outcome_target = y_true[:, 1:2]

            # tf.debugging.assert_all_finite(outcome_target, "outcome_target has nan/inf")
            # tf.debugging.assert_all_finite(y_pred, "y_pred has nan/inf")

            centipawns = y_pred * SCALE
            scale = tf.constant(args.outcome_scale, dtype=tf.float32)
            logits = centipawns / scale
            probs = tf.sigmoid(logits)
            mae = tf.reduce_mean(tf.abs(probs - outcome_target))
            accuracy_score = 1.0 - mae
            return accuracy_score

        @tf.function
        def mae(y_true, y_pred):
            eval_target = y_true[:, 0:1] / y_true[:, 3:4]  # eval component, scale-normalized
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
                y_true = tf.cast(y_true, tf.int32)
                y_pred = tf.cast(y_pred, tf.float32)

                # Apply temperature scaling to logits to reduce magnitude
                temperature = tf.constant(args.move_temperature, dtype=tf.float32)
                scaled_logits = y_pred / temperature

                # Clip to logits to prevent extreme values
                max_logit = tf.constant(args.move_logit_clip, dtype=tf.float32)
                clipped_logits = soft_clip(scaled_logits, max_logit)

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


def get_layer_weights(args, layer):
    """Get layer weights, applying constraints if present."""
    params = layer.get_weights()
    if len(params) != 2:
        return None
    weights, biases = params
    if args.quantize_round:
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
        params = get_layer_weights(args, layer)
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
        params = get_layer_weights(args, layer)
        if params:
            kernel, bias = params
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
            if kernel_data.size == 0:
                print(f"{layer.name}: skipped empty data")
                continue
            kernel_data = kernel_data.reshape(kernel.shape)

            # Read bias weights
            bias_size = np.prod(bias.shape)
            bias_data = np.fromfile(file, dtype=np.float32, count=bias_size)
            bias_data = bias_data.reshape(bias.shape)

            # Set the weights back to the layer
            layer.set_weights([kernel_data, bias_data])


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


def popcount(bb):
    """Count bits in uint64 array using parallel bit counting."""
    bb = bb.astype(np.uint64)
    bb = bb - ((bb >> 1) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> 2) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((bb * np.uint64(0x0101010101010101)) >> 56).astype(np.float32)


def detect_ocb(x):
    """
    Detect opposite-colored bishop endgame positions (bishops + pawns only).
    x: shape (batch, 13) - bitboards + side-to-move
    Returns: boolean array of shape (batch,)
    """
    black_bishops = x[:, 6]
    white_bishops = x[:, 7]

    # Check each side's bishop square colors
    white_on_light = (white_bishops & LIGHT_SQUARES) != 0
    white_on_dark = (white_bishops & DARK_SQUARES) != 0
    black_on_light = (black_bishops & LIGHT_SQUARES) != 0
    black_on_dark = (black_bishops & DARK_SQUARES) != 0

    # Single color only (no bishop pair)
    white_light_only = white_on_light & ~white_on_dark
    white_dark_only = white_on_dark & ~white_on_light
    black_light_only = black_on_light & ~black_on_dark
    black_dark_only = black_on_dark & ~black_on_light

    # OCB: opposite colors, neither side has bishop pair
    is_ocb = (white_light_only & black_dark_only) | (white_dark_only & black_light_only)

    # Only apply to pure bishop endgames: no knights, rooks, or queens
    other_pieces = x[:, 4] | x[:, 5] | x[:, 8] | x[:, 9] | x[:, 10] | x[:, 11]

    return is_ocb & (other_pieces == 0)


# DEBUG: Remove this function once OCB feature is verified
def decode_position(array):
    """Decode bitboards to python-chess board for debugging."""
    import chess
    turn = array[12]
    bitboards = [int(x) for x in list(array[:12])]
    board = chess.Board(fen=None)
    for b in bitboards:
        board.occupied |= b
    for b in bitboards[::2]:
        board.occupied_co[chess.BLACK] |= b
    for b in bitboards[1::2]:
        board.occupied_co[chess.WHITE] |= b
    board.kings = bitboards[0] | bitboards[1]
    board.pawns = bitboards[2] | bitboards[3]
    board.knights = bitboards[4] | bitboards[5]
    board.bishops = bitboards[6] | bitboards[7]
    board.rooks = bitboards[8] | bitboards[9]
    board.queens = bitboards[10] | bitboards[11]
    board.turn = bool(turn)
    return board


def dataset_from_file(args, filepath, strategy, callbacks):
    # Features are packed as np.uint64
    packed_feature_count = int(np.ceil(args.hot_encoding / 64))

    def vertical_mirror(bitboards):
        """
        Mirror bitboard vertically (rank 1 <-> rank 8, etc.)
        Works on arrays of uint64.
        """
        b = bitboards.astype(np.uint64)
        b = ((b >> 56) & 0x00000000000000FF) | \
            ((b >> 40) & 0x000000000000FF00) | \
            ((b >> 24) & 0x0000000000FF0000) | \
            ((b >>  8) & 0x00000000FF000000) | \
            ((b <<  8) & 0x000000FF00000000) | \
            ((b << 24) & 0x0000FF0000000000) | \
            ((b << 40) & 0x00FF000000000000) | \
            ((b << 56) & 0xFF00000000000000)
        return b

    def flip_position(x):
        """
        Flip position colors: swap piece colors, mirror vertically, flip STM.
        Input x: shape (batch_size, 13) - 12 bitboards + side-to-move
        """
        flipped = np.empty_like(x)

        # Swap color pairs and vertical mirror
        for i in range(6):
            black_idx = i * 2
            white_idx = i * 2 + 1
            # Swap colors and mirror vertically
            flipped[:, black_idx] = vertical_mirror(x[:, white_idx])
            flipped[:, white_idx] = vertical_mirror(x[:, black_idx])

        # Flip side-to-move
        flipped[:, 12] = x[:, 12] ^ 1

        return flipped

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
            self.member_starts, self.member_profiles = member_profile_table(self.data, filepath, args)
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

            # Member (source file) of each row, for per-member label-scale profiles
            member_ids = np.searchsorted(self.member_starts, np.arange(start, end), side='right') - 1

            white_to_move = tf.equal(x[:,-1:], 1)  # Training data is from side-to-move POV

            # Get both evaluation and outcome data
            y_eval = self.data[start:end, self.feature_count:self.feature_count+1]
            y_eval = tf.cast(y_eval, tf.int64)  # Cast from unsigned to signed
            y_eval = tf.cast(y_eval, tf.float32) / SCALE  # Convert to float, and scale
            y_eval = tf.where(white_to_move, y_eval, -y_eval)  # Convert to White's perspective

            y_outcome = self.data[start:end, self.feature_count+1:self.feature_count+2]
            y_outcome = tf.cast(y_outcome, tf.float32) - 1.0  # Convert 0,1,2 -> -1,0,1
            # Convert from STM perspective to white's perspective
            y_outcome_white_pov = tf.where(white_to_move, y_outcome, -y_outcome)
            # Convert to win probability: -1->0.0, 0->0.5, 1->1.0
            y_outcome = (y_outcome_white_pov + 1.0) / 2.0

            # Capture draw status before smoothing (for OCB adjustment)
            is_draw = np.squeeze(y_outcome == 0.5)

            y_outcome = y_outcome * (1 - args.outcome_smoothing) + 0.5 * args.outcome_smoothing

            mask = None

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

                mask = ~is_capture

                x = x[mask]
                y_eval = y_eval[mask]
                y_outcome = y_outcome[mask]
                is_draw = is_draw[mask]
                member_ids = member_ids[mask]

            # OCB draw adjustment (must be before balance)
            if args.ocb_draw_margin > 0:
                piece_count_ocb = np.zeros(x.shape[0], dtype=np.float32)
                for i in range(12):
                    piece_count_ocb += popcount(x[:, i])

                margin = args.ocb_draw_margin / SCALE
                within_margin = np.abs(np.squeeze(y_eval)) <= margin
                is_ocb = detect_ocb(x)
                is_endgame = piece_count_ocb <= args.ocb_max_pieces

                not_already_zero = np.squeeze(y_eval) != 0
                ocb_mask = is_draw & within_margin & is_ocb & is_endgame & not_already_zero

                # DEBUG: Remove this block once OCB feature is verified
                if args.debug and np.any(ocb_mask):
                    indices = np.where(ocb_mask)[0][:3]  # First 3 matches
                    for idx in indices:
                        board = decode_position(x[idx])
                        print(f"OCB adjustment: {board.epd()} eval={y_eval[idx][0]*SCALE:.0f}cp -> 0")

                # Apply adjustment
                if np.any(ocb_mask):
                    y_eval = np.where(ocb_mask[:, np.newaxis], 0.0, y_eval)

            if args.balance:
                # Create balanced white/black batches by synthesizing symmetrical possitions
                assert not args.predict_moves, "balance and predict_moves cannot be used at the same time"
                # Flip positions
                x_flipped = flip_position(x)
                x = np.concatenate([x, x_flipped], axis=0)
                member_ids = np.concatenate([member_ids, member_ids])

                # Flip evals (negate)
                y_eval_flipped = -y_eval
                y_eval = tf.concat([y_eval, y_eval_flipped], axis=0)

                # Flip outcomes (1.0 - outcome swaps win/loss, keeps 0.5 draws)
                y_outcome_flipped = 1.0 - y_outcome
                y_outcome = tf.concat([y_outcome, y_outcome_flipped], axis=0)

            # Compute piece count ratio (0 to 1) for dynamic outcome weighting
            piece_count = np.zeros(x.shape[0], dtype=np.float32)
            for i in range(12):
                piece_count += popcount(x[:, i])
            piece_ratio = (piece_count / 32.0)[:, np.newaxis]

            # Per-member, per-bucket label scale (all ones without --profile or sidecars)
            label_scale = self.member_profiles[member_ids, scale_buckets(x)][:, np.newaxis]

            # Combine targets into a single tensor
            y_combined = tf.concat(
                [y_eval, y_outcome, tf.constant(piece_ratio), tf.constant(label_scale)], axis=1
            )  # Shape: (batch_size, 4)

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
                        'profile': args.profile,
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
                ((None, 4), (None, 1))
            )
        else:
            output_types = (np.uint64, np.float32)
            output_shapes = ((None, packed_feature_count), (None, 4))

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

                if args.discard_mismatch:
                    threshold = args.discard_mismatch / SCALE
                    # eval_y is from White's POV, outcome_y is 0.0 (loss), 0.5 (draw), 1.0 (win)

                    # Strong white advantage but black won
                    white_winning_black_won = tf.logical_and(
                        tf.greater(eval_y, threshold),
                        tf.less(outcome_y, 0.25)
                    )
                    # Strong black advantage but white won
                    black_winning_white_won = tf.logical_and(
                        tf.less(eval_y, -threshold),
                        tf.greater(outcome_y, 0.75)
                    )
                    mismatch = tf.logical_or(white_winning_black_won, black_winning_white_won)
                    condition = tf.logical_and(condition, tf.logical_not(mismatch))

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


def load_model(path, pool_size=None):
    custom_objects = {
        'combined_loss': None,
        'scaled_sparse_categorical_crossentropy': None,
        'top': None,
        'top_3': None,
        'top_5': None,
    }

    # The 'pool' Lambda binds the module-global POOL_SIZE at load time. When the
    # source model was trained with a different POOL_SIZE, temporarily restore it
    # so the saved graph reconstructs with the correct pooled width.
    global POOL_SIZE
    saved = POOL_SIZE
    if pool_size is not None:
        POOL_SIZE = pool_size
    try:
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    finally:
        POOL_SIZE = saved


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

        dst = to_layer.get_weights()
        if dst:  # Trainable?
            if [w.shape for w in dst] != [w.shape for w in params]:
                logging.warning(f"Layer {name}: shape mismatch {[w.shape for w in params]} -> {[w.shape for w in dst]}, skipping")
                continue
            try:
                to_layer.set_weights(params)
            except Exception:
                logging.exception(name)


def main(args):
    if args.gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    alt_model = None
    if args.alt_model and os.path.exists(args.alt_model):
        alt_model = load_model(args.alt_model, pool_size=args.alt_pool_size)

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

            lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=args.patience, min_lr=1e-10)
            callbacks.append(lr)

        if args.model is not None:
            assert os.path.exists(os.path.dirname(args.model))

            # When sampling, per-epoch loss is not comparable (different subset each
            # epoch), so best-only would save erratically: force every-epoch saves.
            save_best_only = not bool(args.save_freq) and not args.sample

            # https://keras.io/api/callbacks/model_checkpoint/
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                args.model,
                monitor='loss',
                mode='min',
                save_best_only=save_best_only,
                save_freq=args.save_freq if args.save_freq else 'epoch',
            )
            callbacks.append(model_checkpoint_callback)

        model.summary(line_length=148)
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

        parser.add_argument('--bin', action='store_true', help='export weights in binary format')
        parser.add_argument('--freeze-eval', action='store_true')
        parser.add_argument('--hex', action='store_true', help='export weights in hex format')
        parser.add_argument('--import-file', help='import weights from binary file')
        parser.add_argument('--save-model', action='store_true', help='save model immediately, use with --import and/or --alt-model')

        parser.add_argument('--loss-bce', action='store_true', help='use binary cross-entropy loss (default is MSE)')
        parser.add_argument('--loss-blend', action='store_true', help='use Huber on raw eval domain + BCE on outcome')
        parser.add_argument('--loss-mae', action='store_true', help='use mean absolute error loss (defaule is MSE)')
        parser.add_argument('--huber-delta', type=float, default=1.5, help='delta for Huber loss when using --loss-blend (eval domain units)')

        parser.add_argument('--no-capture', action='store_true', help='exclude captures from training')
        parser.add_argument('--no-draw', action='store_true', help='exclude draws from training')
        parser.add_argument('--discard-mismatch', type=float, default=0, help='discard examples where |eval| > threshold (in centipawns) AND game outcome disagrees')
        parser.add_argument('--ocb-draw-margin', type=float, default=0, help='force eval to 0 for OCB draws within margin in centipawns (0=disabled)')
        parser.add_argument('--ocb-max-pieces', type=int, default=12, help='max total pieces (incl. kings) for OCB draw adjustment')

        parser.add_argument('--outcome-weight', type=float, default=0.1, help='weight for outcome loss vs eval loss')
        parser.add_argument('--outcome-scale', type=float, default=400.0, help='scale factor for converting centipawns to win probability (sigmoid scaling)')
        parser.add_argument('--dynamic-outcome-weight', action='store_true', help='scale outcome weight by piece count (more pieces = trust outcome more)')

        # Move prediction related arguments
        parser.add_argument('--predict-moves', action='store_true', help='enable move prediction')
        parser.add_argument('--move-weight', type=float, default=0.3, help='blending weight for move prediction loss')

        # Arguments for move prediction stability
        parser.add_argument('--move-temperature', type=float, default=1.0, help='temperature scaling for move logits')
        parser.add_argument('--move-logit-clip', type=float, default=10.0, help='clip move logits to prevent extreme values')
        parser.add_argument('--move-loss-scale', type=float, default=0.1, help='scale factor for move prediction loss')
        parser.add_argument('--clip-norm', type=float, default=1.0, help='gradient clipping norm')

        parser.add_argument('--alt-model', help='Path to another model to load/merge weights from')
        parser.add_argument('--alt-pool-size', type=int, default=POOL_SIZE, help='POOL_SIZE the --alt-model was trained with (for loading its pool Lambda)')

        parser.add_argument('--profile', help='JSON dataset profile with per-bucket label scale ratios')

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
        parser.add_argument('--patience', type=int, default=3, help='how many iterations to wait before decaying LR when using --schedule')
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
        if args.predict_moves and (args.move_weight < 0 or args.move_weight > 1):
            parser.error("--move-weight must be between 0 and 1 (inclusive)")

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

        if args.profile:
            args.profile_ratios = load_profile(args.profile)
            print(f'Loaded profile {args.profile}')
        else:
            args.profile_ratios = None  # per-member sidecar profiles, see member_profile_table

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
