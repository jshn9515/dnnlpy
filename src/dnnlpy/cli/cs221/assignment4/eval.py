"""
Visualizes an agent (random, or trained with train.py) performing the MountainCar task.

    > dnnlpy-cs221-a4-eval.exe --agent value-iteration

See the usage string for more details.

    > dnnlpy-cs221-a4-eval.exe --help
"""

import argparse
import json
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np

import dnnlpy
import dnnlpy.cs221.assignment4 as a4

from .utils import (
    normalized_fourier_extractor,
    raw_fourier_extractor,
)

__all__ = ['main']

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv('DNNL_LOG_LEVEL', 'INFO').upper(),
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def eval_naive_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the naive agent.')

    mdp = a4.DiscreteMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        discount=args.discount,
        time_limit=args.time_limit,
    )
    rl = a4.RandomRLAlgorithm(mdp.actions, seed=args.seed)

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the naive agent completed.')


def eval_value_iteration_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the model-based value iteration agent.')

    mdp = a4.DiscreteMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        bins=20,
        discount=args.discount,
        time_limit=args.time_limit,
    )
    rl = a4.ModelBasedMonteCarlo(
        actions=mdp.actions,
        discount=mdp.discount,
        num_states=mdp.num_states,
        epsilon=0.15,
        seed=args.seed,
    )

    # Load the pre-trained policy from a JSON file.
    path = args.path / f'value_iteration_trial_{args.trial}_weights.json'
    if not path.exists():
        logger.error(
            f'The file `value_iteration_trial_{args.trial}_weights.json` does not exist. '
            f'Please run the training script first.'
        )
        mdp.close()
        return

    with open(path) as fp:
        policy = json.load(fp)

    policy = {int(k): v for k, v in policy.items()}
    for key, action in policy.items():
        rl.pi_indices[key] = rl.actions.index(action)
    logger.info('Loaded pre-trained policy from JSON file.')

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the model-based value iteration agent completed.')


def eval_tabular_q_learning_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the Tabular Q-Learning agent.')

    mdp = a4.DiscreteMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        bins=20,
        discount=args.discount,
        time_limit=args.time_limit,
    )
    rl = a4.TabularQLearning(
        actions=mdp.actions,
        discount=mdp.discount,
        num_states=mdp.num_states,
        epsilon=0.15,
        seed=args.seed,
    )

    # Load the pre-trained Q-table from a NumPy file.
    path = args.path / f'tabular_trial_{args.trial}_weights.npy'
    if not path.exists():
        logger.error(
            f'The file `tabular_trial_{args.trial}_weights.npy` does not exist. '
            f'Please run the training script first.'
        )
        mdp.close()
        return

    rl.Q = np.load(path)
    logger.info('Loaded pre-trained Q-table from NumPy file.')

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the Tabular Q-Learning agent completed.')


def eval_function_approx_q_learning_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the Function Approximation Q-Learning agent.')

    mdp = a4.ContinuousMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        discount=args.discount,
        time_limit=args.time_limit,
    )

    if args.extractor_mode == 'raw':
        feature_extractor = raw_fourier_extractor
    else:
        feature_extractor = partial(normalized_fourier_extractor, mdp)

    rl = a4.FunctionApproxQLearning(
        feature_dim=36,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        epsilon=0.2,
        seed=args.seed,
    )

    # Load the pre-trained weights from a NumPy file.
    path = args.path / f'function_approx_trial_{args.trial}_weights.npy'
    if not path.exists():
        logger.error(
            f'The file `function_approx_trial_{args.trial}_weights.npy` does not exist. '
            f'Please run the training script first.'
        )
        mdp.close()
        return

    rl.weight = np.load(path)
    logger.info('Loaded pre-trained weights from NumPy file.')

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the Function Approximation Q-Learning agent completed.')


def eval_constrained_q_learning_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the Constrained Q-Learning agent.')

    mdp = a4.ContinuousMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        discount=args.discount,
        time_limit=args.time_limit,
    )

    if args.extractor_mode == 'raw':
        feature_extractor = raw_fourier_extractor
    else:
        feature_extractor = partial(normalized_fourier_extractor, mdp)

    rl = a4.ConstrainedQLearning(
        feature_dim=36,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        epsilon=0.2,
        max_speed=args.max_speed,
        seed=args.seed,
    )

    # Load the pre-trained weights from a NumPy file.
    path = args.path / f'constrained_trial_{args.trial}_weights.npy'
    if not path.exists():
        logger.error(
            f'The file `constrained_trial_{args.trial}_weights.npy` does not exist. '
            f'Please run the training script first.'
        )
        mdp.close()
        return

    rl.weight = np.load(path)
    logger.info('Loaded pre-trained weights from NumPy file.')

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the Constrained Q-Learning agent completed.')


def eval_policy_gradient_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the Policy Gradient (REINFORCE) agent.')

    mdp = a4.ContinuousMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        discount=args.discount,
        time_limit=args.time_limit,
    )

    if args.extractor_mode == 'raw':
        feature_extractor = partial(raw_fourier_extractor, max_coeff=3)
    else:
        feature_extractor = partial(normalized_fourier_extractor, mdp, max_coeff=3)

    rl = a4.PolicyGradient(
        feature_dim=16,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        time_limit=mdp.time_limit,
        seed=args.seed,
    )

    # Load the pre-trained weights from a NumPy file.
    path = args.path / f'policy_gradient_trial_{args.trial}_weights.npy'
    if not path.exists():
        logger.error(
            f'The file `policy_gradient_trial_{args.trial}_weights.npy` does not exist. '
            f'Please run the training script first.'
        )
        mdp.close()
        return

    rl.theta = np.load(path)
    logger.info('Loaded pre-trained weights from NumPy file.')

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the Policy Gradient (REINFORCE) agent completed.')


def eval_actor_critic_agent(args: argparse.Namespace) -> None:
    logger.info('Starting evaluation of the Actor-Critic agent.')

    mdp = a4.ContinuousMountainCarMDP(
        a4.CustomMountainCarEnv(
            max_speed=args.max_speed,
            render_mode='human',
            seed=args.seed,
        ),
        discount=args.discount,
        time_limit=args.time_limit,
    )

    if args.extractor_mode == 'raw':
        feature_extractor = partial(raw_fourier_extractor, max_coeff=3)
    else:
        feature_extractor = partial(normalized_fourier_extractor, mdp, max_coeff=3)

    rl = a4.ActorCritic(
        feature_dim=16,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        time_limit=mdp.time_limit,
        seed=args.seed,
    )

    # Load the pre-trained weights from a NumPy file.
    path = args.path / f'actor_critic_trial_{args.trial}_weights.npy'
    if not path.exists():
        logger.error(
            f'The file `actor_critic_trial_{args.trial}_weights.npy` does not exist. '
            f'Please run the training script first.'
        )
        mdp.close()
        return

    rl.theta = np.load(path)
    logger.info('Loaded pre-trained weights from NumPy file.')

    try:
        a4.simulate(mdp, rl, num_episodes=1, train=False, verbose=False, demo=True)
    finally:
        mdp.close()

    logger.info('Evaluation of the Actor-Critic agent completed.')


def main() -> None:
    """Run the main evaluate script for the ``MountainCar`` environment."""
    parser = argparse.ArgumentParser(
        prog='dnnlpy-cs221-a4-eval.exe',
        description='Evaluate various agents on the MountainCar environment.',
        epilog='Example: `dnnlpy-cs221-a4-eval.exe --agent value-iteration`',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '-a',
        '--agent',
        type=str,
        required=True,
        metavar='AGENT',
        choices=[
            'naive',
            'value-iteration',
            'tabular',
            'function-approx',
            'constrained',
            'policy-gradient',
            'actor-critic',
        ],
        help='Choose the type of agent to train. Available options are: '
        "1. Naive agent (`'naive'`), "
        "2. Model-based value iteration (`'value-iteration'`), "
        "3. Tabular Q-learning (`'tabular'`), "
        "4. Function approximation Q-learning (`'function-approx'`), "
        "5. Constrained Q-learning (`'constrained'`), "
        "6. Policy Gradient (REINFORCE) (`'policy-gradient'`), "
        "7. Actor-Critic (`'actor-critic'`).",
    )
    parser.add_argument(
        '--max-speed',
        type=float,
        default=0.07,
        help='Velocity limit for function approximation and constrained Q-learning.',
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0,
        metavar='N',
        help='Specify the trial number of the pre-trained model to evaluate.',
    )
    parser.add_argument(
        '--discount',
        type=float,
        default=0.999,
        help='The discount factor for the MDP.',
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=1000,
        help='The maximum steps per episode for the MountainCar environment.',
    )
    parser.add_argument(
        '--extractor-mode',
        type=str,
        choices=['raw', 'normalized'],
        default='normalized',
        help='Choose feature extractor mode for function-approx and constrained Q-learning. '
        'Available options are: '
        '1. `raw` for raw Fourier features, '
        '2. `normalized` for normalized Fourier features.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=dnnlpy.get_experiment_path() / 'cs221' / 'assignment4',
        help='Specify the directory to save the results. '
        'This will override the default directory, which is `~/dnnlpy/experiments`.',
    )
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version=f'%(prog)s {dnnlpy.__version__}',
        help='Show the program version number and exit.',
    )
    parser.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit.',
    )
    args = parser.parse_args()

    match args.agent:
        case 'naive':
            eval_naive_agent(args)
        case 'value-iteration':
            eval_value_iteration_agent(args)
        case 'tabular':
            eval_tabular_q_learning_agent(args)
        case 'function-approx':
            eval_function_approx_q_learning_agent(args)
        case 'constrained':
            eval_constrained_q_learning_agent(args)
        case 'policy-gradient':
            eval_policy_gradient_agent(args)
        case 'actor-critic':
            eval_actor_critic_agent(args)
        case _:
            logger.error(f'Unknown agent type: {args.agent}.')


if __name__ == '__main__':
    main()
