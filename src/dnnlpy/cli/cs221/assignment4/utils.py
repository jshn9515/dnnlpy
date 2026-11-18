import os
from argparse import Namespace
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dnnlpy.cs221.assignment4 as a4

__all__ = [
    'plot_rewards',
    'plot_training_curves',
]


def get_file_path(args: Namespace, trial: int, *, plot: bool = False) -> Path:
    """Return the weights or plot path for an agent and trial."""
    match args.agent:
        case 'value-iteration':
            stem = f'value_iteration_{trial}'
        case 'tabular':
            stem = f'tabular_{trial}'
        case 'function-approx' | 'constrained' | 'policy-gradient' | 'actor-critic':
            norm = 'norm' if args.normalized else 'raw'
            stem = (
                f'{args.agent.replace("-", "_")}_{trial}_'
                f'{args.feature_extractor}_{norm}'
            )
            if args.agent in {'policy-gradient', 'actor-critic'} and not args.baseline:
                stem += '_nobase'
        case 'all' if plot:
            stem = f'training_curves_trial_{trial}'
        case _:
            raise ValueError(f'Unknown agent name: {args.agent}.')

    if plot:
        ext = args.format
    elif args.agent == 'value-iteration':
        ext = 'json'
    else:
        ext = 'npy'

    return args.path / f'{stem}.{ext}'


def get_feature_extractor(
    args: Namespace,
    mdp: a4.ContinuousMountainCarMDP,
    degree: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the selected feature extractor for a two-dimensional MountainCar state."""
    if args.normalized:
        bounds = {'low': mdp.base_env.low, 'high': mdp.base_env.high}
    else:
        bounds = {}

    if args.feature_extractor == 'poly':
        return partial(a4.polynomial_feature_extractor, degree=degree, **bounds)
    if args.feature_extractor == 'fourier':
        return partial(a4.fourier_feature_extractor, max_coeff=degree, **bounds)

    raise ValueError(f'Unknown feature extractor: {args.feature_extractor}.')


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Returns the moving average of the input array with the given window size."""
    cum_sum = np.cumsum(x)
    mov_avg = (cum_sum[window:] - cum_sum[:-window]) / window
    return mov_avg


def plot_rewards(
    train_rewards: np.ndarray,
    eval_rewards: np.ndarray,
    eval_greedy_rewards: np.ndarray | None = None,
    path: str | os.PathLike[str] | None = None,
    show: bool = True,
) -> None:
    """Plots the rewards of training and evaluation episodes, with moving averages.

    If ``eval_greedy_rewards`` is not None, also plots those evaluation episodes (orange)
    on top of the green ones.
    """
    window = 30
    train_mov_avg = moving_average(train_rewards, window)
    eval_mov_avg = moving_average(eval_rewards, window)
    t_len = len(train_rewards)
    e_len = len(eval_rewards)

    plt.rc('font', size=12)
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        range(t_len),
        train_rewards,
        s=8,
        c='tab:blue',
        linewidth=0,
        alpha=0.5,
        label='train rewards',
    )
    ax.plot(
        range(window // 2, t_len - window // 2),
        train_mov_avg,
        linewidth=2,
        color='blue',
        label='train moving average',
    )
    ax.scatter(
        range(t_len, t_len + e_len),
        eval_rewards,
        s=8,
        c='tab:green',
        linewidth=0,
        alpha=0.5,
        label='eval rewards (sampling)',
    )
    ax.plot(
        range(t_len + window // 2, t_len + e_len - window // 2),
        eval_mov_avg,
        color='darkgreen',
        linewidth=2,
        label='eval moving average (sampling)',
    )

    if eval_greedy_rewards is not None:
        ax.scatter(
            range(t_len, t_len + len(eval_greedy_rewards)),
            eval_greedy_rewards,
            s=8,
            c='tab:orange',
            alpha=0.5,
            linewidth=0,
            label='eval rewards (greedy)',
        )
        ax.plot(
            range(
                t_len + window // 2,
                t_len + len(eval_greedy_rewards) - window // 2,
            ),
            moving_average(eval_greedy_rewards, window),
            color='darkorange',
            linewidth=2,
            label='eval moving average (greedy)',
        )

    ax.legend(loc='lower right')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Discounted Reward in Episode')

    if path:
        fig.savefig(path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()


def plot_training_curves(
    curves: list[tuple[str, np.ndarray]],
    scatter: bool = True,
    path: str | os.PathLike[str] | None = None,
    show: bool = True,
) -> None:
    """
    Plots several training runs together: for each (label, rewards) in curves, the reward of every
    episode (if scatter is True) and its moving average over 30 episodes.
    """
    window = 30

    plt.rc('font', size=12)
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(1, 1, 1)

    for label, rewards in curves:
        if scatter:
            ax.scatter(
                range(len(rewards)),
                rewards,
                s=5,
                linewidth=0,
                alpha=0.3,
            )
        ax.plot(
            range(window // 2, len(rewards) - window // 2),
            moving_average(rewards, window),
            linewidth=2,
            label=f'{label} (moving average)',
        )

    ax.legend(loc='lower right')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Discounted Reward in Episode')

    if path is not None:
        fig.savefig(path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
