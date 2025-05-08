#! /usr/bin/env python3
import argparse
import ast
from collections import OrderedDict
import contextlib
import importlib.util
import importlib.machinery
import logging
import nevergrad as ng
import os
import sqlite3
import shutil
import signal
import sys
import tqdm


def load_engine(args, name='chess_engine'):
    engine_module_path = os.path.abspath(args.engine)
    if not os.path.isdir(engine_module_path):
        engine_module_path = os.path.dirname(engine_module_path)

    for ext in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(engine_module_path, name + ext)
        if os.path.isfile(path):
            loader = importlib.machinery.ExtensionFileLoader(name, path)
            spec = importlib.util.spec_from_file_location(name, path, loader=loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"Cannot find binary module {name} in {engine_module_path}")


def log_best_param(optimizer, scaled):
    recommendation = optimizer.provide_recommendation()
    best_param = recommendation.value[1]

    # Log best params
    logger.info(f'best param: {best_param}')

    # Scale the best parameters for logging
    config = {}
    for k, v in best_param.items():
        if k in scaled:
            lo, hi = scaled[k]
            config[k] = int((v + 1) * (hi - lo) / 2 + lo)
        else:
            config[k] = v

    logger.info(f'best config: {config}\n')


def checkpoint(args, optimizer):
    if args.checkpoint:
        # Flag to track if Ctrl+C was pressed during checkpoint
        interrupt_received = []

        # Custom handler that remembers the interrupt but doesn't act on it yet
        def delayed_interrupt_handler(signum, frame):
            interrupt_received.append(True)
            logger.debug("Interrupt received, will exit after checkpoint completes...")

        # Install our custom handler
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, delayed_interrupt_handler)

        try:
            if os.path.isfile(args.checkpoint):
                backup = args.checkpoint + ".bak"
                shutil.move(args.checkpoint, backup)

            optimizer.dump(args.checkpoint)

        finally:
            # Restore original handler
            signal.signal(signal.SIGINT, original_sigint_handler)

            # If we received an interrupt while checkpointing, process it now
            if interrupt_received:
                logger.debug("Checkpoint completed, now handling the deferred interrupt")
                original_sigint_handler(signal.SIGINT, None)


def create_scaled_ranges_map(engine_params):
    """Map normalized parameters to their value ranges"""
    scaled = {}

    for name, param_info in engine_params.items():
        if param_info and param_info[4]:  # If normal flag is True
            _, lo, hi, _, _ = param_info
            # Create closure with fixed lo/hi values
            scaled[name] = (lo, hi)

    return scaled


def convert_mate_score(score, from_mate_value, to_mate_value):
    """
    Convert mate scores between different engines that use different mate values
    Preserves the distance-to-mate information
    """
    mate_threshold = from_mate_value - 1000

    if score > mate_threshold:
        # Positive mate score
        distance_to_mate = from_mate_value - score
        adjusted_score = to_mate_value - distance_to_mate
        logger.info(f'adjust score from: {score} to: {adjusted_score}')
        return adjusted_score
    elif score < -mate_threshold:
        # Negative mate score
        distance_to_mate = from_mate_value + score  # score is negative
        adjusted_score = -(to_mate_value - distance_to_mate)
        logger.info(f'adjust score from: {score} to: {adjusted_score}')
        return adjusted_score
    else:
        return score


def huber_loss(y_true, y_pred, delta):
    """
    Calculate directional Huber loss between true and predicted values
    - Preserves sign of the error
    - Quadratic for errors smaller than delta in magnitude
    - Linear for errors larger than delta in magnitude
    """
    error = y_true - y_pred
    sign = 1 if error >= 0 else -1
    abs_error = abs(error)

    if abs_error <= delta:
        # Quadratic region
        return 0.5 * (abs_error ** 2) * sign
    else:
        # Linear region
        return delta * (abs_error - 0.5 * delta) * sign


@contextlib.contextmanager
def logging_context(logger, should_log):
    original_level = logger.level
    if not should_log:
        logger.setLevel(logging.CRITICAL)

    try:
        yield
    finally:
        logger.setLevel(original_level)


def tune(args, optimizer):
    engine = load_engine(args)
    logger.info(f'engine version: {engine.version()}\n')

    engine.set_param('Threads', args.threads)
    engine.set_param('Hash', args.hash)

    engine_params = engine.get_param_info()
    scaled = create_scaled_ranges_map(engine_params)

    # Track how many positions we need to skip when resuming
    positions_to_skip = args.offset if args.offset is not None else optimizer.num_ask
    if positions_to_skip:
        logger.info(f'Resuming from offset - skipping first {positions_to_skip} positions\n')

    SAVE_FREQ = max(1, args.save_frequency)
    LOG_FREQ = max(1, args.log_frequency)

    with sqlite3.connect(args.eval_db) as conn:
        cursor = conn.cursor()

        # Use OFFSET to skip already processed positions
        query_result = cursor.execute(
            "SELECT epd, depth, score FROM position LIMIT ? OFFSET ?",
            (args.budget - optimizer.num_ask if args.budget > 0 else -1, positions_to_skip)
        )

        # Wrap with tqdm if progress is requested
        if args.progress:
            total = None if args.budget <= 0 else args.budget
            query_result = tqdm.tqdm(
                query_result,
                desc='Tuning',
                dynamic_ncols=True,
                unit='pos',
                initial=optimizer.num_ask,
                total=total)

        for epd, _, eval in query_result:
            if args.budget > 0 and optimizer.num_ask > args.budget:
                break

            x = optimizer.ask()

            with logging_context(logger, optimizer.num_ask % LOG_FREQ == 0):
                params = x.kwargs
                for k, v in params.items():
                    # Apply scaling to parameters that need it
                    if k in scaled:
                        lo, hi = scaled[k]
                        v = int((v + 1) * (hi - lo) / 2 + lo)
                    engine.set_param(k, v)

                logger.info(f'budget: {optimizer.num_ask}')
                logger.info(f'recommended param: {params}')

                # Run engine evaluation on the position given by EPD
                score = engine.eval(epd, args.eval_as_white, max(1, args.depth), args.time_limit_ms)
                score = convert_mate_score(score, args.mate_value, args.db_mate_value)

                loss = eval - score

                logger.info(f'epd: {epd}, eval: {eval}, engine: {score}, loss: {loss}')

                if abs(loss) >= args.max_loss:
                    if (eval * score) > 0 and abs(score) >= args.max_loss:
                        loss = 0
                        logger.info('large loss with same sign - setting to zero')
                    else:  # Different signs (one positive, one negative)
                        logger.info('max loss exceeded with different signs - skipping')
                        continue

                loss = huber_loss(eval, score, delta=args.huber_delta)
                logger.info(f'huber_loss: {loss}')

                optimizer.tell(x, loss * args.loss_scale)

                if optimizer.num_ask % SAVE_FREQ == 0:
                    checkpoint(args, optimizer)

                log_best_param(optimizer, scaled)

        checkpoint(args, optimizer)
        log_best_param(optimizer, scaled)


def optimizer_instance(args, instrum):
    """Instantiate SPSA optimizer"""

    input_data_file = args.checkpoint
    budget = args.budget if args.budget > 0 else None

    if input_data_file and os.path.isfile(input_data_file):
        loaded_optimizer = ng.optimizers.SPSA(instrum, budget=budget)
        optimizer = loaded_optimizer.load(input_data_file)
    else:
        optimizer = ng.optimizers.SPSA(instrum, budget=budget)

    # k: iteration index, starts at 0 and increases each loop
    # a_k = a / (A + k + 1)^alpha is the decaying step size
    # a: base step size that controls how aggressively parameters are updated
    # A: stability constant that influences how quickly step size decays
    # alpha: step size decay rate, typically 0.602 for SPSA (constant)
    # c_k = c / (k + 1)^gamma is the perturbation size at iteration k
    # c: base perturbation size that determines how far to look when estimating gradients
    # gamma: perturbation decay rate, typically 0.101 for SPSA (constant)

    if args.initial_amplification is None:
        if budget is None:
            optimizer.A = 10000
        # else use nevergrad SPSA default
    else:
        optimizer.A = args.initial_amplification

    if args.step_size_decay_rate is not None:
        optimizer.a = args.step_size_decay_rate

    if args.perturbation_magnitude is not None:
        optimizer.c = args.perturbation_magnitude

    logger.info(f'SPSA: A={optimizer.A}, a={optimizer.a}, c={optimizer.c}, previous budget: {optimizer.num_ask}\n')
    return optimizer


def main(args):
    logger.info(f'nevergrad {ng.__version__}')

    # Convert the input param string to a dict of dict and sort by key.
    input_param = ast.literal_eval(args.input_param)
    input_param = OrderedDict(sorted(input_param.items()))
    logger.info(f'input param: {input_param}\n')

    # Prepare parameters to be optimized.
    arg = {}
    for k, v in input_param.items():
        if type(v) == list:
            arg.update({k: ng.p.Choice(v)})
        else:
            if isinstance(v["init"], int):
                arg.update({k: ng.p.Scalar(init=v['init'], lower=v['lower'], upper=v['upper']).set_integer_casting()})
            elif isinstance(v["init"], float):
                arg.update({k: ng.p.Scalar(init=v['init'], lower=v['lower'], upper=v['upper'])})

    instrum = ng.p.Instrumentation(**arg)
    optimizer = optimizer_instance(args, instrum)

    tune(args, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation-based SPSA tuner")
    parser.add_argument('eval_db', help='Path to sqlite3 DB containing evals')
    parser.add_argument('--budget', type=int, default=100000)
    parser.add_argument('--checkpoint', help='Path to checkpoint data file')
    parser.add_argument('--db-mate-value', type=int, default=15000, help='Mate value used in the database (default: 15000)')
    parser.add_argument('--depth', type=int, default=7, help='Evaluation depth (default: 7)')
    parser.add_argument('--engine', required=True, help='Path to engine')
    parser.add_argument('--eval-as-white', action='store_true', help="Engine evaluates from white's perspective")
    parser.add_argument('--hash', type=int, default=256, help='Engine hashtable size in MB')
    parser.add_argument('--huber-delta', type=int, default=50, help='Delta parameter for Huber loss (default: 50)')
    parser.add_argument('--initial-amplification', '-A', type=int, help='Initial SPSA amplification (stability constant)')
    parser.add_argument('--input-param', required=True, type=str, help='The parameters that will be optimized')
    parser.add_argument('--log-file', help='Path to log file', default='log_tune.txt')
    parser.add_argument('--log-frequency', '-f', type=int, default=10, help='INFO Logging frequency (default: 10)')
    parser.add_argument('--loss-scale', type=float, default=1.0, help='Scale factor for loss amplification (default: 1.0)')
    parser.add_argument('--mate-value', type=int, default=29999, help='Mate value used by the engine (default: 29999)')
    parser.add_argument('--max-loss', type=int, default=10000, help='Max absolute loss value (default: 10000)')
    parser.add_argument('--offset', type=int, help='Offset in the db to start from')
    parser.add_argument('--perturbation-magnitude', '-c', type=float)
    parser.add_argument('--progress', action='store_true', help='Show progress bar and disable console output')
    parser.add_argument('--save-frequency', type=int, default=1000, help='Checkpoint frequency (save state)')
    parser.add_argument('--step-size-decay-rate', '-a', type=float)
    parser.add_argument('--threads', type=int, default=1, help='Engine threads')
    parser.add_argument('--time-limit-ms', type=int, default=10000, help='Time limit per evaluation, unlimited if <= 0 (default: 10000)')

    args = parser.parse_args()

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')

    if not args.progress:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
