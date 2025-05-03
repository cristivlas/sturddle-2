#! /usr/bin/env python3
import argparse
import ast
from collections import OrderedDict
import importlib.util
import importlib.machinery
import logging
import nevergrad as ng
import os
import sqlite3
import shutil
import signal
import sys


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


def checkpoint(args, optimizer):

    if args.checkpoint:
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        # Set temporary handler to ignore interrupts
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            # Backup the checkpoint file in case a catastrophic error happens during pickling.
            if args.backup and os.path.isfile(args.checkpoint):
                shutil.copy(args.checkpoint, args.checkpoint + ".bak")

            optimizer.dump(args.checkpoint)

        finally:
            # Restore original handler
            signal.signal(signal.SIGINT, original_sigint_handler)


def scale(name, val, engine_params):
    """Re-scale normalized parameters"""
    p = engine_params.get(name)
    if p:
        (_, lo, hi, _, normal) = p
        if normal:
            val = int((val + 1) * (hi - lo) / 2 + lo)

    return val


def engine_eval(args, engine, engine_params, epd, **params):
    """Run engine evaluation on the position given by epd"""

    for k,v in params.items():
        v = scale(k, v, engine_params)
        engine.set_param(k, v)

    logging.info(f'recommended param: {params}')

    return engine.eval_as_white(epd)


def tune(args, optimizer):
    engine = load_engine(args)
    logging.info(f'engine version: {engine.version()}\n')

    engine_params = engine.get_param_info()

    # Track how many positions we need to skip when resuming
    if positions_to_skip := optimizer.num_ask:
        logging.info(f'Resuming from checkpoint - skipping first {positions_to_skip} positions\n')

    with sqlite3.connect(args.eval_db) as conn:
        cursor = conn.cursor()

        # Use OFFSET to skip already processed positions
        for epd, _, eval in cursor.execute(
            "SELECT epd, depth, score FROM position LIMIT ? OFFSET ?",
            (args.budget - optimizer.num_ask if args.budget > 0 else -1, positions_to_skip)
        ):
            if args.budget > 0 and optimizer.num_ask > args.budget:
                break

            logging.info(f'budget: {optimizer.num_ask}')

            x = optimizer.ask()

            result = engine_eval(args, engine, engine_params, epd, **x.kwargs)
            loss = eval - result

            logging.info(f'epd: {epd}, eval: {eval}, engine: {result}, loss: {loss}')

            optimizer.tell(x, loss * args.loss_scale)

            checkpoint(args, optimizer)

            recommendation = optimizer.provide_recommendation()
            best_param = recommendation.value[1]

            # Log best params same as lakas.py for compatibility with plotting tool.
            logging.info(f'best param: {best_param}')

            config = {k:scale(k, v, engine_params) for k,v in best_param.items()}
            logging.info(f'best config: {config}\n')


def optimizer_instance(args, instrum):
    input_data_file = args.checkpoint
    if input_data_file and os.path.isfile(input_data_file):
        loaded_optimizer = ng.optimizers.SPSA(instrum, budget=args.budget)
        optimizer = loaded_optimizer.load(input_data_file)
    else:
        optimizer = ng.optimizers.SPSA(instrum, budget=args.budget)
        optimizer.A = args.initial_amplification
        optimizer.a = args.step_size_decay_rate
        optimizer.c = args.perturbation_magnitude

    logging.info(f'SPSA: A={optimizer.A}, a={optimizer.a}, c={optimizer.c}, previous budget: {optimizer.num_ask}\n')
    return optimizer


def main(args):
    logging.info(f'nevegrad {ng.__version__}')

    # Convert the input param string to a dict of dict and sort by key.
    input_param = ast.literal_eval(args.input_param)
    input_param = OrderedDict(sorted(input_param.items()))
    logging.info(f'input param: {input_param}\n')

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
    parser.add_argument('--backup', action='store_true', help='Backup checkpoint file')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--checkpoint', help='Path to checkpoint data file')
    parser.add_argument('--engine', required=True, help='Path to engine')
    parser.add_argument('--perturbation-magnitude', '-c', type=float, default=1e-1)
    parser.add_argument('--initial-amplification', '-A', type=int, default=10000, help='Initial SPSA amplification')
    parser.add_argument('--input-param', required=True, type=str, help='The parameters that will be optimized')
    parser.add_argument('--log-file', help='Path to log file', default='log_tune.txt')
    parser.add_argument('--loss-scale', type=float, default=1.0, help='Scale factor for loss amplification')
    parser.add_argument('--step-size-decay-rate', '-a', type=float, default=1e-5)

    args = parser.parse_args()

    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=log_handlers
    )

    main(args)
