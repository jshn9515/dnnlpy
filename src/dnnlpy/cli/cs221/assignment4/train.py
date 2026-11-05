"""
Trains an agent on the mountain car task and plots its rewards per episode.

    > dnnlpy-cs221-a4-train.exe --agent value-iteration

See the usage string for more details.

    > dnnlpy-cs221-a4-train.exe --help
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

import dnnlpy
import dnnlpy.cs221.assignment4 as a4

from .utils import plot_rewards

__all__ = ['main']

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv('DNNL_LOG_LEVEL', 'INFO').upper(),
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def train_value_iteration(args: argparse.Namespace) -> None:
    logger.info('Starting training for value-iteration agent.')

    for i in range(1, args.num_trials + 1):
        mdp = a4.DiscreteMountainCarMDP(
            a4.CustomMountainCarEnv(),
            bins=20,
            discount=0.999,
            time_limit=1000,
        )
        rl = a4.ModelBasedMonteCarlo(
            mdp.actions,
            mdp.discount,
            mdp.num_states,
            calc_val_iter_every=int(1e5),
            epsilon=0.5,
        )

        try:
            logger.info(f'Starting training for trial {i} of value-iteration agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, num_episodes=1000)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of value-iteration agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(f'Running evaluation for trial {i} of value-iteration agent.')
            t1 = time.perf_counter()
            eval_rewards = a4.simulate(
                mdp, rl, num_episodes=500, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of value-iteration agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        if args.demo:
            mdp = a4.DiscreteMountainCarMDP(
                a4.CustomMountainCarEnv(render_mode='human'),
                bins=20,
                discount=0.999,
                time_limit=1000,
            )

            try:
                a4.simulate(
                    mdp, rl, num_episodes=1, train=False, verbose=False, demo=True
                )
            finally:
                mdp.close()

        results = {
            idx: rl.actions[action]
            for idx, action in enumerate(rl.pi_indices)
            if action != -1
        }

        with open(args.path / f'vi_trial_{i}_weights.json', 'w') as fp:
            json.dump(results, fp, indent=2, sort_keys=True)
        logger.info(f'Saved weights for trial {i} of value-iteration agent.')

        path = args.path / f'vi_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path)


def train_tabular_q_learning(args: argparse.Namespace) -> None:
    logger.info('Starting training for tabular Q-learning agent.')

    for i in range(1, args.num_trials + 1):
        mdp = a4.DiscreteMountainCarMDP(
            a4.CustomMountainCarEnv(),
            bins=20,
            discount=0.999,
            time_limit=1000,
        )
        rl = a4.TabularQLearning(
            mdp.actions,
            mdp.discount,
            mdp.num_states,
            epsilon=0.15,
        )

        try:
            logger.info(f'Starting training for trial {i} of tabular Q-learning agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, num_episodes=1000)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of tabular Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(
                f'Running evaluation for trial {i} of tabular Q-learning agent.'
            )
            t1 = time.perf_counter()
            eval_rewards = a4.simulate(
                mdp, rl, num_episodes=500, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of tabular Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        if args.demo:
            mdp = a4.DiscreteMountainCarMDP(
                a4.CustomMountainCarEnv(render_mode='human'),
                bins=20,
                discount=0.999,
                time_limit=1000,
            )

            try:
                a4.simulate(
                    mdp, rl, num_episodes=1, train=False, verbose=False, demo=True
                )
            finally:
                mdp.close()

        np.save(args.path / f'tabular_trial_{i}_weights.npy', rl.Q)
        logger.info(f'Saved weights for trial {i} of tabular Q-learning agent.')

        path = args.path / f'tabular_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path)


def train_function_approx_q_learning(args: argparse.Namespace) -> None:
    logger.info('Starting training for function approximation Q-learning agent.')

    for i in range(1, args.num_trials + 1):
        mdp = a4.ContinuousMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed),
            discount=0.999,
            time_limit=1000,
        )

        def feature_extractor(state: np.ndarray) -> np.ndarray:
            return a4.fourier_feature_extractor(state, max_coeff=5, scale=[1, 15])

        rl = a4.FunctionApproxQLearning(
            feature_dim=36,
            feature_extractor=feature_extractor,
            actions=mdp.actions,
            discount=mdp.discount,
            epsilon=0.2,
        )

        try:
            logger.info(
                f'Starting training for trial {i} of function approximation Q-learning agent.'
            )
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, num_episodes=1000)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of function approximation Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(
                f'Running evaluation for trial {i} of function approximation Q-learning agent.'
            )
            t1 = time.perf_counter()
            eval_rewards = a4.simulate(
                mdp, rl, num_episodes=500, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of function approximation Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        if args.demo:
            mdp = a4.ContinuousMountainCarMDP(
                a4.CustomMountainCarEnv(max_speed=args.max_speed, render_mode='human'),
                discount=0.999,
                time_limit=1000,
            )

            try:
                a4.simulate(
                    mdp, rl, num_episodes=1, train=False, verbose=False, demo=True
                )
            finally:
                mdp.close()

        np.save(args.path / f'function_approx_trial_{i}_weights.npy', rl.weight)
        logger.info(
            f'Saved weights for trial {i} of function approximation Q-learning agent.'
        )

        path = args.path / f'function_approx_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path)


def train_constrained_q_learning(args: argparse.Namespace) -> None:
    logger.info('Starting training for constrained Q-learning agent.')

    for i in range(1, args.num_trials + 1):
        mdp = a4.ContinuousMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed),
            discount=0.999,
            time_limit=1000,
        )

        def feature_extractor(state: np.ndarray) -> np.ndarray:
            return a4.fourier_feature_extractor(state, max_coeff=5, scale=[1, 15])

        rl = a4.ConstrainedQLearning(
            feature_dim=36,
            feature_extractor=feature_extractor,
            actions=mdp.actions,
            discount=mdp.discount,
            epsilon=0.2,
        )

        try:
            logger.info(
                f'Starting training for trial {i} of constrained Q-learning agent.'
            )
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, num_episodes=1000)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of constrained Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(
                f'Running evaluation for trial {i} of constrained Q-learning agent.'
            )
            t1 = time.perf_counter()
            eval_rewards = a4.simulate(
                mdp, rl, num_episodes=500, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of constrained Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        if args.demo:
            mdp = a4.ContinuousMountainCarMDP(
                a4.CustomMountainCarEnv(max_speed=args.max_speed, render_mode='human'),
                discount=0.999,
                time_limit=1000,
            )

            try:
                a4.simulate(
                    mdp, rl, num_episodes=1, train=False, verbose=False, demo=True
                )
            finally:
                mdp.close()

        np.save(args.path / f'constrained_trial_{i}_weights.npy', rl.weight)
        logger.info(f'Saved weights for trial {i} of constrained Q-learning agent.')

        path = args.path / f'constrained_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path)


def main() -> None:
    """Run the main training script for the ``MountainCar`` environment."""
    parser = argparse.ArgumentParser(
        prog='dnnlpy-cs221-a4-train.exe',
        description='Train and evaluate various agents on the MountainCar environment.',
        epilog='Example: `dnnlpy-cs221-a4-train.exe --agent tabular`',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        metavar='AGENT',
        choices=[
            'value-iteration',
            'tabular',
            'function-approx',
            'constrained',
        ],
        help='Choose the type of agent to train. Available options are: '
        "1. Model-based value iteration (`'value-iteration'`), "
        "2. Tabular Q-learning (`'tabular'`), "
        "3. Function approximation Q-learning (`'function-approx'`), "
        "4. Constrained Q-learning (`'constrained'`).",
    )
    parser.add_argument(
        '--max_speed',
        type=float,
        default=0.07,
        help='Velocity limit for function approximation and constrained Q-learning.',
    )
    parser.add_argument(
        '--num_trials',
        type=int,
        default=3,
        help='The number of trials to run.',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a demo of the trained agent.',
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=dnnlpy.get_experiment_path() / 'cs221' / 'assignment4',
        help='Specify the directory to save the results. '
        'This will override the default directory, which is `~/dnnlpy/experiments`.',
    )
    parser.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit.',
    )
    args = parser.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    match args.agent:
        case 'value-iteration':
            train_value_iteration(args)
        case 'tabular':
            train_tabular_q_learning(args)
        case 'function-approx':
            train_function_approx_q_learning(args)
        case 'constrained':
            train_constrained_q_learning(args)
        case _:
            raise ValueError(f'Unknown agent type: {args.agent}.')


if __name__ == '__main__':
    main()
