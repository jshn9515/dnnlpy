import math

import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4
import dnnlpy.cs221.assignment4.utils as utils


def test_polynomial_feature_extractor_with_scaling():
    actual = utils.polynomial_feature_extractor(
        np.array([2.0, 3.0]), degree=1, scale=[0.5, 2.0]
    )
    expected = [1.0, 6.0, 1.0, 6.0]

    npt.assert_allclose(actual, expected)


def test_fourier_feature_extractor_for_two_features():
    actual = utils.fourier_feature_extractor(np.array([0.5, 0.25]), max_coeff=1)
    expected = [1.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5)]

    npt.assert_allclose(actual, expected, atol=1e-15)


class TwoStepMDP(a4.MDP[int, int]):
    state = 0
    discount = 0.5
    time_limit = 3

    @property
    def actions(self) -> list[int]:
        return [1]

    @property
    def num_actions(self) -> int:
        return 1

    @property
    def states(self) -> list[int]:
        return [0, 1, 2]

    @property
    def num_states(self) -> int:
        return 3

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> a4.MDPTransition[int, int]:
        state = self.state
        self.state += 1
        terminated = self.state == 2

        return a4.MDPTransition(
            state, action, self.state, self.state, terminated, False
        )

    def close(self) -> None:
        pass


def test_simulate_stops_on_terminal_and_discounts_rewards():
    mdp = TwoStepMDP()
    rl = a4.FixedRLAlgorithm({0: 1, 1: 1}, mdp.actions, epsilon=0)

    actual = utils.simulate(mdp, rl, num_episodes=2, train=False, verbose=False)
    expected = [2.0, 2.0]
    npt.assert_allclose(actual, expected)


class EarlyTruncationMDP(TwoStepMDP):
    def step(self, action: int) -> a4.MDPTransition[int, int]:
        if self.state != 0:
            raise RuntimeError('Step called after truncation.')

        transition = super().step(action)
        return transition._replace(truncated=True)


def test_simulate_stops_on_early_truncation():
    mdp = EarlyTruncationMDP()
    rl = a4.FixedRLAlgorithm({0: 1, 1: 1}, mdp.actions, epsilon=0)

    rewards = utils.simulate(mdp, rl, num_episodes=2, train=True, verbose=False)

    npt.assert_allclose(rewards, [1.0, 1.0])


def test_simulate_resets_number_line_between_episodes():
    env = a4.NumberLineEnv(n=1, seed=0)
    mdp = a4.NumberLineMDP(env)
    rl = a4.FixedRLAlgorithm({mdp.n: 0}, mdp.actions, epsilon=0)

    rewards = utils.simulate(mdp, rl, num_episodes=2, train=False, verbose=False)
    assert len(rewards) == 2

    rewards_set = set(rewards)
    assert rewards_set.issubset({mdp.base_env.left_reward, mdp.base_env.right_reward})


class NoActionAfterFirstStep(a4.RLAlgorithm[int, int]):
    def __init__(self) -> None:
        self.transitions: list[a4.MDPTransition[int, int]] = []

    def get_action(self, state: int, explore: bool = True) -> int | None:
        return 1 if state == 0 else None

    def step(self, transition: a4.MDPTransition[int, int]) -> None:
        self.transitions.append(transition)


@pytest.mark.parametrize('train', [False, True])
def test_simulate_ends_episode_when_agent_has_no_action(train: bool):
    mdp = TwoStepMDP()
    rl = NoActionAfterFirstStep()

    rewards = utils.simulate(mdp, rl, num_episodes=2, train=train, verbose=False)

    npt.assert_allclose(rewards, [1.0, 1.0])
    assert mdp.state == 1
    assert len(rl.transitions) == (2 if train else 0)
