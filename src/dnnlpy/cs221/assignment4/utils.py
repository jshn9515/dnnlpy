from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .algorithms import RLAlgorithm
    from .mdp import MDP

type Array1D = np.ndarray[tuple[int]]

__all__ = [
    'fourier_feature_extractor',
    'polynomial_feature_extractor',
    'simulate',
]

logger = logging.getLogger(__name__)


def polynomial_feature_extractor(
    state: Array1D,
    degree: int = 3,
    scale: npt.ArrayLike | None = None,
) -> Array1D:
    r"""Extracts polynomial features from the given state up to the specified degree.

    For a state with :math:`d` dimensions, this function computes all polynomial combinations
    of the state dimensions up to the specified degree, including interaction terms. The
    resulting feature vector will have a length of :math:`m`, which includes all combinations
    of the state dimensions up to the specified degree.

    For example, we have a 3D state :math:`(x, y, z)` with degree 2, and scale :math:`[2, 1, 1]`,
    the output will include features like:

    .. math::

        [1, 2x, y, z, 4x^2, y^2, z^2, 2xy, 2xz, yz, 4x^2y, 4x^2z, \dots, 4x^2y^2z^2]

    Args:
        state (np.ndarray): The input state, a NumPy array of shape ``(features,)``.
        degree (int, optional): The maximum degree of the polynomial features. Defaults to 3.
        scale (npt.ArrayLike, optional): An optional scaling factor for each dimension of
            the state. If provided, each dimension of the state will be multiplied by the
            corresponding scale factor before computing the polynomial features. If not
            provided, no scaling is applied. Defaults to ``None``.

    Returns:
        np.ndarray: A 1D array containing the polynomial features of the input state, including
        all combinations of the state dimensions up to the specified degree.

    Examples:
        >>> state = np.array([1, 2])
        >>> features = polynomial_feature_extractor(state, degree=2)
    """
    if scale is None:
        scale = np.ones_like(state)
    else:
        scale = np.asarray(scale)

    features = np.array([1.0])
    for s, sc in zip(state, scale, strict=True):
        features = np.outer(features, (s * sc) ** np.arange(degree + 1))
        features = features.flatten()

    return features


# Problem 4b: Fourier feature extractor
def fourier_feature_extractor(
    state: Array1D,
    max_coeff: int = 5,
    scale: npt.ArrayLike | None = None,
) -> Array1D:
    r"""Extracts Fourier features from the given state up to the specified maximum coefficient.

    For a state with :math:`d` dimensions, this function computes the Fourier features of
    the state dimensions up to the specified maximum coefficient. The resulting feature vector
    will have a length of :math:`m`, which includes all combinations of the state dimensions
    up to the specified maximum coefficient.

    .. math::

        [1, \cos(\pi c_1 s_1), \cos(\pi c_2 s_2), \ldots, \cos(\pi c_d s_d),
        \cos(\pi c_1 s_1 + \pi c_2 s_2 + \ldots + \pi c_d s_d), \ldots]

    Args:
        state (np.ndarray): The input state, a NumPy array of shape ``(features,)``.
        max_coeff (int, optional): The maximum coefficient for the Fourier features.
            Defaults to 5.
        scale (npt.ArrayLike, optional): An optional scaling factor for each dimension of
            the state. If provided, each dimension of the state will be multiplied by the
            corresponding scale factor before computing the Fourier features. If not
            provided, no scaling is applied. Defaults to ``None``.

    Returns:
        np.ndarray: A 1D array containing the Fourier features of the input state, including
        all combinations of the state dimensions up to the specified maximum coefficient.

    Examples:
        >>> state = np.array([0.5, 1.0])
        >>> features = fourier_feature_extractor(state, max_coeff=2)
    """
    if scale is None:
        scale = np.ones_like(state)
    else:
        scale = np.asarray(scale)

    features = np.array([0.0])
    for s, sc in zip(state, scale, strict=True):
        new_features = np.arange(max_coeff + 1) * (s * sc)
        features = features[:, None] + new_features[None, :]
        features = features.flatten()

    return np.cos(np.pi * features)


def _render_mdp[StateT, ActionT](mdp: MDP[StateT, ActionT]) -> None:
    """Renders a Mountain Car MDP environment."""
    from .mdp import MountainCarMDP

    if not isinstance(mdp, MountainCarMDP):
        raise TypeError('Demo mode is only supported for MountainCarMDP.')

    mdp.env.render()


def simulate[StateT, ActionT](
    mdp: MDP[StateT, ActionT],
    rl: RLAlgorithm[StateT, ActionT],
    num_episodes: int = 100,
    train: bool = True,
    verbose: bool = True,
    demo: bool = False,
) -> Array1D:
    """Runs a simulation of the given MDP and RL algorithm for a specified number of trials.

    On each trial, the RL algorithm interacts with the MDP according to its dynamics, and
    the discounted reward is calculated for each trial. A trial ends when a terminal state
    is reached, a transition is truncated, the MDP's time limit is reached, or when the
    RL algorithm returns ``None`` for an action (problem 5c).

    Args:
        mdp (MDP): The Markov Decision Process to simulate.
        rl (RLAlgorithm): The reinforcement learning algorithm to be simulated.
        num_episodes (int, optional): The number of episodes to run. Defaults to 100.
        train (bool, optional): Whether to allow the RL algorithm to learn from feedback.
            Defaults to ``True``.
        verbose (bool, optional): Whether to print progress information during the simulation.
            Defaults to ``True``.
        demo (bool, optional): Whether to render the MDP environment during the simulation.
            Defaults to ``False``.

    Returns:
        np.ndarray: A 1D array containing the discounted rewards obtained from each trial.
    """
    total_rewards: list[float] = []  # The discounted rewards we get on each trial

    for episode in range(num_episodes):
        state = mdp.reset()  # Reset the MDP to its initial state for each episode

        if demo:
            _render_mdp(mdp)

        total_discount = 1
        total_reward = 0
        episode_length = 0

        for _ in range(mdp.time_limit):
            if demo:
                time.sleep(0.05)

            action = rl.get_action(state, explore=train)
            if action is None:
                break  # No feasible action is available; end this episode

            transition = mdp.step(action)
            episode_length += 1

            if train:
                rl.step(transition)

            total_reward += total_discount * transition.reward
            total_discount *= mdp.discount
            state = transition.next_state

            if transition.terminated or transition.truncated:
                break  # The episode has ended

        if verbose and episode % 100 == 0:
            n = len(str(num_episodes))
            logger.info(
                f'Episode {episode:{n}} | reward: {total_reward:.4f}, '
                f'length: {episode_length:d}'
            )

        total_rewards.append(total_reward)

    return np.array(total_rewards)
