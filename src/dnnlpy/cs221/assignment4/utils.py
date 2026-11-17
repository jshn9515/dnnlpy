from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .algorithms import RLAlgorithm
    from .mdp import MDP

type Array1D = np.ndarray[tuple[int]]
type EvalMode = Literal['sample', 'greedy']

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
    low: Array1D | None = None,
    high: Array1D | None = None,
) -> Array1D:
    r"""Extracts polynomial features from the given state up to the specified degree.

    Each state dimension is raised to powers from zero through ``degree``, including
    interaction terms. For a two-dimensional state, the result has
    ``(degree + 1) ** 2`` features.

    For example, we have a 3D state :math:`(x, y, z)` with degree 2, and scale :math:`[2, 1, 1]`,
    the output will include features like:

    .. math::

        [1, 2x, y, z, 4x^2, y^2, z^2, 2xy, 2xz, yz, 4x^2y, 4x^2z, \dots, 4x^2y^2z^2]

    Args:
        state (np.ndarray): A state vector. The default scale assumes two dimensions.
        degree (int, optional): The maximum power per state dimension. Defaults to 3.
        scale (npt.ArrayLike, optional): An optional scaling factor for each dimension of
            the state. If provided, each dimension of the state will be multiplied by the
            corresponding scale factor before computing the polynomial features. If not
            provided, the default is ``[1, 15]`` for raw states and ``[1, 1]`` for
            normalized states.
        low (np.ndarray, optional): The lower bound of the state space for normalization.
            Provide with ``high`` to normalize the state to ``[0, 1]`` before scaling.
            Defaults to ``None``.
        high (np.ndarray, optional): The upper bound of the state space for normalization.
            Provide with ``low`` to normalize the state to ``[0, 1]`` before scaling.
            Defaults to ``None``.

    Returns:
        np.ndarray: A 1D array containing the polynomial features of the input state, including
        all combinations of the state dimensions up to the specified degree.
    """
    if (low is None) ^ (high is None):
        raise AssertionError('`low` and `high` must both be provided or both be None.')

    if low is not None and high is not None:
        state = (state - low) / (high - low)
        default_scale = [1.0, 1.0]
    else:
        default_scale = [1.0, 15.0]

    scale = np.asarray(default_scale if scale is None else scale)
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
    low: Array1D | None = None,
    high: Array1D | None = None,
) -> Array1D:
    r"""Extracts Fourier features from the given state up to the specified maximum coefficient.

    The extractor combines coefficients from zero through ``max_coeff`` for each
    state dimension. For a two-dimensional state, the result has
    ``(max_coeff + 1) ** 2`` features.

    .. math::

        [1, \cos(\pi c_1 s_1), \cos(\pi c_2 s_2), \ldots, \cos(\pi c_d s_d),
        \cos(\pi c_1 s_1 + \pi c_2 s_2 + \ldots + \pi c_d s_d), \ldots]

    Args:
        state (np.ndarray): A state vector. The default scale assumes two dimensions.
        max_coeff (int, optional): The maximum coefficient for the Fourier features.
            Defaults to 5.
        scale (npt.ArrayLike, optional): An optional scaling factor for each dimension of
            the state. If provided, each dimension of the state will be multiplied by the
            corresponding scale factor before computing the Fourier features. If not
            provided, the default is ``[1, 15]`` for raw states and ``[1, 1]`` for
            normalized states.
        low (np.ndarray, optional): The lower bound of the state space for normalization.
            Provide with ``high`` to normalize the state to ``[0, 1]`` before scaling.
            Defaults to ``None``.
        high (np.ndarray, optional): The upper bound of the state space for normalization.
            Provide with ``low`` to normalize the state to ``[0, 1]`` before scaling.
            Defaults to ``None``.

    Returns:
        np.ndarray: A 1D array containing the Fourier features of the input state, including
        all combinations of the state dimensions up to the specified maximum coefficient.
    """
    if (low is None) ^ (high is None):
        raise AssertionError('`low` and `high` must both be provided or both be None.')

    if low is not None and high is not None:
        state = (state - low) / (high - low)
        default_scale = [1.0, 1.0]
    else:
        default_scale = [1.0, 15.0]

    scale = np.asarray(default_scale if scale is None else scale)
    features = np.array([0.0])

    for s, sc in zip(state, scale, strict=True):
        new_features = np.arange(max_coeff + 1) * (s * sc)
        features = features[:, None] + new_features[None, :]
        features = features.flatten()

    return np.cos(np.pi * features)


def _get_action[StateT, ActionT](
    rl: RLAlgorithm[StateT, ActionT],
    state: StateT,
    explore: bool,
    eval_mode: EvalMode = 'sample',
) -> ActionT | None:
    """Gets an action from the RL algorithm, handling the case where no feasible action is
    available.
    """
    from .algorithms import PolicyGradient

    if isinstance(rl, PolicyGradient) and eval_mode == 'greedy':
        action = rl.get_greedy_action(state)
    else:
        action = rl.get_action(state, explore=explore)

    return action


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
    eval_mode: EvalMode = 'sample',
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
        eval_mode (EvalMode, optional): The evaluation mode for the RL algorithm. For Policy
            Gradient agents, this can be either ``sample`` (sample actions from the policy
            distribution) or ``greedy`` (select the action with the highest probability).
            Defaults to ``sample``.
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

            action = _get_action(rl, state, explore=train, eval_mode=eval_mode)
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

        advantage = getattr(rl, 'advantage', None)
        advantage = f'{advantage: .4f}' if isinstance(advantage, float) else 'N/A'

        if verbose and (episode % 100 == 0 or episode == num_episodes - 1):
            n1 = len(str(num_episodes))
            n2 = len(str(mdp.time_limit))
            logger.info(
                f'Episode {episode:>{n1}d} | reward: {total_reward:9.4f}, '
                f'length: {episode_length:<{n2}d} | advantage: {advantage}'
            )

        total_rewards.append(total_reward)

    return np.array(total_rewards)
