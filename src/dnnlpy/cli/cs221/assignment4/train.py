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
from functools import partial
from pathlib import Path

import numpy as np

import dnnlpy
import dnnlpy.cs221.assignment4 as a4

from .utils import (
    normalized_fourier_extractor,
    plot_rewards,
    plot_training_curves,
    raw_fourier_extractor,
)

__all__ = ['main']

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv('DNNL_LOG_LEVEL', 'INFO').upper(),
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def train_value_iteration(args: argparse.Namespace) -> None:
    logger.info('Starting training for value-iteration agent.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        mdp = a4.DiscreteMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed, seed=trial_seed),
            bins=20,
            discount=args.discount,
            time_limit=args.time_limit,
        )
        rl = a4.ModelBasedMonteCarlo(
            actions=mdp.actions,
            discount=mdp.discount,
            num_states=mdp.num_states,
            epsilon=0.5,
            seed=trial_seed,
        )

        try:
            logger.info(f'Starting training for trial {i} of value-iteration agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of value-iteration agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(f'Running evaluation for trial {i} of value-iteration agent.')
            t1 = time.perf_counter()
            eval_rewards = a4.simulate(
                mdp, rl, args.num_eval_episodes, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of value-iteration agent. '
                f'Time: {t2 - t1:.4f} seconds.'
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
        plot_rewards(train_rewards, eval_rewards, path=path, show=args.show_plot)


def train_tabular_q_learning(args: argparse.Namespace) -> None:
    logger.info('Starting training for tabular Q-learning agent.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        mdp = a4.DiscreteMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed, seed=trial_seed),
            bins=20,
            discount=args.discount,
            time_limit=args.time_limit,
        )
        rl = a4.TabularQLearning(
            actions=mdp.actions,
            discount=mdp.discount,
            num_states=mdp.num_states,
            epsilon=0.15,
            seed=trial_seed,
        )

        try:
            logger.info(f'Starting training for trial {i} of tabular Q-learning agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)
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
                mdp, rl, args.num_eval_episodes, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of tabular Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        np.save(args.path / f'tabular_trial_{i}_weights.npy', rl.Q)
        logger.info(f'Saved weights for trial {i} of tabular Q-learning agent.')

        path = args.path / f'tabular_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path, show=args.show_plot)


def train_function_approx_q_learning(args: argparse.Namespace) -> None:
    logger.info('Starting training for function approximation Q-learning agent.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        mdp = a4.ContinuousMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed, seed=trial_seed),
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
            seed=trial_seed,
        )

        try:
            logger.info(
                f'Starting training for trial {i} of function approximation Q-learning agent.'
            )
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)
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
                mdp, rl, args.num_eval_episodes, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of function approximation Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        np.save(args.path / f'function_approx_trial_{i}_weights.npy', rl.weight)
        logger.info(
            f'Saved weights for trial {i} of function approximation Q-learning agent.'
        )

        path = args.path / f'function_approx_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path, show=args.show_plot)


def train_constrained_q_learning(args: argparse.Namespace) -> None:
    logger.info('Starting training for constrained Q-learning agent.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        mdp = a4.ContinuousMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed, seed=trial_seed),
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
            seed=trial_seed,
        )

        try:
            logger.info(
                f'Starting training for trial {i} of constrained Q-learning agent.'
            )
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)
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
                mdp, rl, args.num_eval_episodes, train=False, verbose=False
            )
            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of constrained Q-learning agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        np.save(args.path / f'constrained_trial_{i}_weights.npy', rl.weight)
        logger.info(f'Saved weights for trial {i} of constrained Q-learning agent.')

        path = args.path / f'constrained_trial_{i}.svg'
        plot_rewards(train_rewards, eval_rewards, path=path, show=args.show_plot)


def train_policy_gradient(args: argparse.Namespace) -> None:
    logger.info('Starting training for policy gradient agent.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        mdp = a4.ContinuousMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed, seed=trial_seed),
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
            seed=trial_seed,
        )

        try:
            logger.info(f'Starting training for trial {i} of policy gradient agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of policy gradient agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(f'Running evaluation for trial {i} of policy gradient agent.')
            t1 = time.perf_counter()
            eval_sample_rewards = a4.simulate(
                mdp,
                rl,
                args.num_eval_episodes,
                train=False,
                eval_mode='sample',
                verbose=False,
            )
            logger.info(
                f'Average reward (sampling): {np.mean(eval_sample_rewards):.4f}.'
            )

            eval_greedy_rewards = None
            if args.eval_mode == 'greedy':
                eval_greedy_rewards = a4.simulate(
                    mdp,
                    rl,
                    args.num_eval_episodes,
                    train=False,
                    eval_mode='greedy',
                    verbose=False,
                )
                logger.info(
                    f'Average reward (greedy): {np.mean(eval_greedy_rewards):.4f}.'
                )

            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of policy gradient agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        np.save(args.path / f'policy_gradient_trial_{i}_weights.npy', rl.theta)
        logger.info(f'Saved weights for trial {i} of policy gradient agent.')

        path = args.path / f'policy_gradient_trial_{i}.svg'
        plot_rewards(
            train_rewards,
            eval_sample_rewards,
            eval_greedy_rewards,
            path=path,
            show=args.show_plot,
        )


def train_actor_critic(args: argparse.Namespace) -> None:
    logger.info('Starting training for actor-critic agent.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        mdp = a4.ContinuousMountainCarMDP(
            a4.CustomMountainCarEnv(max_speed=args.max_speed, seed=trial_seed),
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
            seed=trial_seed,
        )

        try:
            logger.info(f'Starting training for trial {i} of actor-critic agent.')
            t1 = time.perf_counter()
            train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)
            t2 = time.perf_counter()
            logger.info(
                f'Training complete for trial {i} of actor-critic agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )

            logger.info(f'Running evaluation for trial {i} of actor-critic agent.')
            t1 = time.perf_counter()
            eval_sample_rewards = a4.simulate(
                mdp,
                rl,
                args.num_eval_episodes,
                train=False,
                eval_mode='sample',
                verbose=False,
            )
            logger.info(
                f'Average reward (sampling): {np.mean(eval_sample_rewards):.4f}.'
            )

            eval_greedy_rewards = None
            if args.eval_mode == 'greedy':
                eval_greedy_rewards = a4.simulate(
                    mdp,
                    rl,
                    args.num_eval_episodes,
                    train=False,
                    eval_mode='greedy',
                    verbose=False,
                )
                logger.info(
                    f'Average reward (greedy): {np.mean(eval_greedy_rewards):.4f}.'
                )

            t2 = time.perf_counter()
            logger.info(
                f'Evaluation complete for trial {i} of actor-critic agent. '
                f'Time: {t2 - t1:.4f} seconds.'
            )
        finally:
            mdp.close()

        np.save(args.path / f'actor_critic_trial_{i}_weights.npy', rl.theta)
        logger.info(f'Saved weights for trial {i} of actor-critic agent.')

        path = args.path / f'actor_critic_trial_{i}.svg'
        plot_rewards(
            train_rewards,
            eval_sample_rewards,
            eval_greedy_rewards,
            path=path,
            show=args.show_plot,
        )


def train_all_agents(args: argparse.Namespace) -> None:
    logger.info('Training six agents from the same seed to perform MountainCar task.')

    for i in range(args.num_trials):
        trial_seed = None if args.seed is None else args.seed + i
        logger.info(f'Starting training for trial {i} of all agents.')

        curves: list[tuple[str, np.ndarray]] = []
        for name in [
            'value-iteration',
            'tabular',
            'function-approx',
            'constrained',
            'policy-gradient',
            'actor-critic',
        ]:
            logger.info(f'Starting training for agent: {name}.')

            match name:
                case 'value-iteration':
                    mdp = a4.DiscreteMountainCarMDP(
                        a4.CustomMountainCarEnv(
                            max_speed=args.max_speed,
                            seed=trial_seed,
                        ),
                        bins=20,
                        discount=args.discount,
                        time_limit=args.time_limit,
                    )
                    rl = a4.ModelBasedMonteCarlo(
                        actions=mdp.actions,
                        discount=mdp.discount,
                        num_states=mdp.num_states,
                        epsilon=0.5,
                        seed=trial_seed,
                    )

                case 'tabular':
                    mdp = a4.DiscreteMountainCarMDP(
                        a4.CustomMountainCarEnv(
                            max_speed=args.max_speed,
                            seed=trial_seed,
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
                        seed=trial_seed,
                    )

                case 'function-approx':
                    mdp = a4.ContinuousMountainCarMDP(
                        a4.CustomMountainCarEnv(
                            max_speed=args.max_speed,
                            seed=trial_seed,
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
                        seed=trial_seed,
                    )

                case 'constrained':
                    mdp = a4.ContinuousMountainCarMDP(
                        a4.CustomMountainCarEnv(
                            max_speed=args.max_speed,
                            seed=trial_seed,
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
                        seed=trial_seed,
                    )

                case 'policy-gradient':
                    mdp = a4.ContinuousMountainCarMDP(
                        a4.CustomMountainCarEnv(
                            max_speed=args.max_speed,
                            seed=trial_seed,
                        ),
                        discount=args.discount,
                        time_limit=args.time_limit,
                    )

                    if args.extractor_mode == 'raw':
                        feature_extractor = partial(raw_fourier_extractor, max_coeff=3)
                    else:
                        feature_extractor = partial(
                            normalized_fourier_extractor, mdp, max_coeff=3
                        )

                    rl = a4.PolicyGradient(
                        feature_dim=16,
                        feature_extractor=feature_extractor,
                        actions=mdp.actions,
                        discount=mdp.discount,
                        seed=trial_seed,
                    )

                case 'actor-critic':
                    mdp = a4.ContinuousMountainCarMDP(
                        a4.CustomMountainCarEnv(
                            max_speed=args.max_speed,
                            seed=trial_seed,
                        ),
                        discount=args.discount,
                        time_limit=args.time_limit,
                    )

                    if args.extractor_mode == 'raw':
                        feature_extractor = partial(raw_fourier_extractor, max_coeff=3)
                    else:
                        feature_extractor = partial(
                            normalized_fourier_extractor, mdp, max_coeff=3
                        )

                    rl = a4.ActorCritic(
                        feature_dim=16,
                        feature_extractor=feature_extractor,
                        actions=mdp.actions,
                        discount=mdp.discount,
                        seed=trial_seed,
                    )

                case _:
                    logger.error(f'Unknown agent type: {name}.')
                    return

            try:
                logger.info(f'Starting training for trial {i} of agent: {name}.')
                t1 = time.perf_counter()
                train_rewards = a4.simulate(mdp, rl, args.num_train_episodes)  # type: ignore
                t2 = time.perf_counter()
                logger.info(
                    f'Training complete for trial {i} of agent: {name}. '
                    f'Time: {t2 - t1:.4f} seconds.'
                )
            finally:
                mdp.close()

            curves.append((name, train_rewards))

        plot_training_curves(
            curves,
            path=args.path / f'training_curves_trial_{i}.svg',
            show=args.show_plot,
        )
        logger.info(f'Saved training curve for trial {i}.')


def main() -> None:
    """Run the main training script for the ``MountainCar`` environment."""
    parser = argparse.ArgumentParser(
        prog='dnnlpy-cs221-a4-train.exe',
        description='Train various agents on the MountainCar environment.',
        epilog='Example: `dnnlpy-cs221-a4-train.exe --agent value-iteration`',
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
            'value-iteration',
            'tabular',
            'function-approx',
            'constrained',
            'policy-gradient',
            'actor-critic',
            'all',
        ],
        help='Choose the type of agent to train. Available options are: '
        "1. Model-based value iteration (`'value-iteration'`), "
        "2. Tabular Q-learning (`'tabular'`), "
        "3. Function approximation Q-learning (`'function-approx'`), "
        "4. Constrained Q-learning (`'constrained'`), "
        "5. Policy Gradient (REINFORCE) (`'policy-gradient'`), "
        "6. Actor-Critic (`'actor-critic'`), "
        "7. Train all agents (`'all'`).",
    )
    parser.add_argument(
        '--max-speed',
        type=float,
        default=0.07,
        help='Velocity limit for function approximation and constrained Q-learning.',
    )
    parser.add_argument(
        '--eval-mode',
        choices=['sample', 'greedy'],
        default='sample',
        help='Evaluation mode for Policy Gradient agents only: '
        '`sample` samples actions from the policy distribution, '
        'while `greedy` selects the action with the highest probability.',
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=3,
        help='The number of trials to run.',
    )
    parser.add_argument(
        '--num-train-episodes',
        type=int,
        default=1000,
        help='The number of training episodes to run for each trial.',
    )
    parser.add_argument(
        '--num-eval-episodes',
        type=int,
        default=500,
        help='The number of evaluation episodes to run for each trial.',
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
        '--show-plot',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to display the plot of rewards per episode after training.',
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
        case 'policy-gradient':
            train_policy_gradient(args)
        case 'actor-critic':
            train_actor_critic(args)
        case 'all':
            train_all_agents(args)
        case _:
            logger.error(f'Unknown agent type: {args.agent}.')


if __name__ == '__main__':
    main()
