import math
from pathlib import Path

import numpy as np
import numpy.testing as npt

import dnnlpy.cs221.assignment4 as a4

from ..utils import timeout

FIXTURES_PATH = Path(__file__).parent / 'fixtures'


@timeout
def test_problem_4a_0():
    mdp = a4.NumberLineMDP(a4.NumberLineEnv())
    rl = a4.TabularQLearning(
        mdp.actions,
        mdp.discount,
        mdp.num_states,
        epsilon=0.15,
    )

    rl.step(a4.MDPTransition(2, 0, -5, 3, False))

    assert rl.Q[1 + mdp.n, mdp.actions.index(1)] == 0
    assert rl.Q[1 + mdp.n, mdp.actions.index(0)] == 0
    assert rl.Q[mdp.n, mdp.actions.index(0)] == -0.5

    rl.step(a4.MDPTransition(3, 0, 50, 4, True))

    assert rl.Q[1 + mdp.n, mdp.actions.index(0)] == 5.0
    assert rl.Q[1 + mdp.n, mdp.actions.index(1)] == 0
    assert rl.Q[mdp.n, mdp.actions.index(0)] == -0.5

    rl.step(a4.MDPTransition(1, 1, -5, 2, False))

    assert rl.Q[1 + mdp.n, mdp.actions.index(0)] == 5.0
    assert rl.Q[1 + mdp.n, mdp.actions.index(1)] == 0
    assert rl.Q[mdp.n, mdp.actions.index(0)] == -0.5
    assert rl.Q[mdp.n, mdp.actions.index(1)] == 0
    assert rl.Q[mdp.n - 1, mdp.actions.index(1)] == -0.5

    # The number line MDP has discount 1 and the Q-values above start at 0,
    # so we also check the same updates with a discount of 0.9 and initial_q=1.
    # The Q-values of the end state 2 are never updated, so they stay at 1: a transition
    # to a terminal state has to use 0 as the value of the next state instead.
    rl = a4.TabularQLearning(
        actions=mdp.actions,
        discount=0.9,
        num_states=mdp.num_states,
        epsilon=0.15,
        initial_q=1.0,
    )

    # 1 + 0.1 * (50 + 0 - 1) = 5.9
    rl.step(a4.MDPTransition(3, 0, 50, 4, True))
    if not rl.Q[1 + mdp.n, mdp.actions.index(0)] == 5.9:
        raise AssertionError(
            'Incorrect Q-value after a transition to a terminal state.'
        )

    # 1 + 0.1 * (-5 + 0.9 * 5.9 - 1) = 0.931
    rl.step(a4.MDPTransition(2, 1, -5, 3, False))
    if not rl.Q[mdp.n, mdp.actions.index(1)] == 0.931:
        raise AssertionError('Incorrect Q-value with discount 0.9')

    assert rl.Q[mdp.n, mdp.actions.index(0)] == 1


@timeout
def test_problem_4a_1():
    mdp = a4.NumberLineMDP(a4.NumberLineEnv())
    rl = a4.TabularQLearning(
        mdp.actions,
        mdp.discount,
        mdp.num_states,
        epsilon=0.15,
    )

    rl.step(a4.MDPTransition(2, 0, -5, 3, False))
    rl.step(a4.MDPTransition(2, 0, -5, 3, False))
    rl.step(a4.MDPTransition(2, 0, -5, 1, False))
    rl.step(a4.MDPTransition(2, 1, -5, 3, False))
    rl.step(a4.MDPTransition(2, 1, -5, 1, False))
    rl.step(a4.MDPTransition(3, 0, 50, 4, True))
    rl.step(a4.MDPTransition(3, 0, 50, 4, True))
    rl.step(a4.MDPTransition(3, 0, -5, 2, False))
    rl.step(a4.MDPTransition(3, 1, 50, 4, True))
    rl.step(a4.MDPTransition(3, 1, -5, 2, False))
    rl.step(a4.MDPTransition(1, 0, -5, 2, False))
    rl.step(a4.MDPTransition(1, 0, 10, 0, True))
    rl.step(a4.MDPTransition(1, 1, -5, 2, False))

    pi = {-1: 0, 0: 1, 1: 0}
    for state in range(-mdp.n + 1, mdp.n):
        if pi[state] != rl.get_action(state + mdp.n, explore=False):
            raise AssertionError(f'Incorrect greedy action with the state {state}.')


@timeout
def test_problem_4a_2_hidden():
    env = a4.NumberLineEnv()
    mdp = a4.NumberLineMDP(env)
    rl = a4.TabularQLearning(
        mdp.actions,
        mdp.discount,
        mdp.num_states,
        epsilon=0.15,
    )

    rl.step(a4.MDPTransition(2, 0, -5, 3, False))
    rl.step(a4.MDPTransition(2, 0, -5, 3, False))
    rl.step(a4.MDPTransition(2, 0, -5, 1, False))
    rl.step(a4.MDPTransition(2, 1, -5, 3, False))
    rl.step(a4.MDPTransition(2, 1, -5, 1, False))
    rl.step(a4.MDPTransition(3, 0, 50, 4, True))
    rl.step(a4.MDPTransition(3, 0, 50, 4, True))
    rl.step(a4.MDPTransition(3, 0, -5, 2, False))
    rl.step(a4.MDPTransition(3, 1, 50, 4, True))
    rl.step(a4.MDPTransition(3, 1, -5, 2, False))
    rl.step(a4.MDPTransition(1, 0, -5, 2, False))
    rl.step(a4.MDPTransition(1, 0, 10, 0, True))
    rl.step(a4.MDPTransition(1, 1, -5, 2, False))


@timeout
def test_problem_4b_0():
    state = np.array([0.5, 0.3])
    actual = a4.fourier_feature_extractor(state)
    expected = np.load(FIXTURES_PATH / '4b-0-gold.npy', allow_pickle=True)

    npt.assert_allclose(actual, expected)


def feature_extractor(state: tuple[float, float]) -> np.ndarray:
    return a4.fourier_feature_extractor(np.asarray(state), max_coeff=5, scale=[1, 15])


@timeout
def test_problem_4c_0():
    env = a4.CustomMountainCarEnv()
    mdp = a4.ContinuousMountainCarMDP(env, discount=0.999, time_limit=1000)
    rl = a4.FunctionApproxQLearning(
        feature_dim=36,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        epsilon=0.2,
    )
    rl.weight = np.zeros((36, 3))

    rl.step(a4.MDPTransition((0, 0), 1, -1, (-0.2, -0.01), False))
    rl.step(a4.MDPTransition((0.7, -0.03), 2, -1, (0.8, -0.01), False))
    rl.step(a4.MDPTransition((-0.3, -0.05), 0, -1, (-0.4, -0.03), False))

    actual = rl.get_q((0.2, -0.02), 1)
    expected = -0.0074065262637628875

    if not math.isclose(actual, expected, abs_tol=1e-6):
        raise AssertionError('Wrong Q value computed for given state and action.')


def test_problem_4c_1():
    env = a4.CustomMountainCarEnv()
    mdp = a4.ContinuousMountainCarMDP(env, discount=0.999, time_limit=1000)
    rl = a4.FunctionApproxQLearning(
        feature_dim=36,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        epsilon=0.2,
    )
    rl.weight = np.zeros((36, 3))

    rl.step(a4.MDPTransition((0, 0), 1, -1, (-0.2, -0.01), False))
    rl.step(a4.MDPTransition((0.7, -0.03), 2, -1, (0.8, -0.01), False))
    rl.step(a4.MDPTransition((-0.3, -0.05), 0, -1, (-0.4, -0.03), False))

    action = rl.get_action((0.2, -0.02), explore=False)
    assert action == 0

    action = rl.get_action((1, 0.03), explore=False)
    assert action == 2

    action = rl.get_action((-0.6, -0.06), explore=False)
    assert action == 0


def test_problem_4c_2():
    env = a4.CustomMountainCarEnv()
    mdp = a4.ContinuousMountainCarMDP(env, discount=0.999, time_limit=1000)
    rl = a4.FunctionApproxQLearning(
        feature_dim=36,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=mdp.discount,
        epsilon=0.2,
    )
    rl.weight = np.zeros((36, 3))

    rl.step(a4.MDPTransition((0, 0), 1, -1, (-0.2, -0.01), False))
    rl.step(a4.MDPTransition((0.7, -0.03), 2, -1, (0.8, -0.01), False))
    rl.step(a4.MDPTransition((-0.3, -0.05), 0, -1, (-0.4, -0.03), False))

    gold = np.load(FIXTURES_PATH / '4c-2-gold.npy', allow_pickle=True)
    actual = np.sort(rl.weight, axis=0)
    expected = np.sort(gold, axis=0)

    npt.assert_allclose(actual, expected, atol=1e-6)

    # One update with discount 0.9 and one ending at a terminal state, starting from
    # all-ones weights.
    # Since the features can be in any order, we compare Q-values instead of weights.
    rl = a4.FunctionApproxQLearning(
        feature_dim=36,
        feature_extractor=feature_extractor,
        actions=mdp.actions,
        discount=0.9,
        epsilon=0.2,
    )
    rl.weight = np.ones((36, 3))

    rl.step(a4.MDPTransition((0.2, 0.01), 0, -1, (0.25, 0.02), False))
    actual = rl.get_q((0.2, 0.01), 0)
    expected = -11.089247347330799

    if not math.isclose(actual, expected, abs_tol=1e-6):
        raise AssertionError('Weight update incorrect with discount 0.9.')

    rl.step(a4.MDPTransition((0.4, 0.03), 2, -1, (0.5, 0.04), True))
    actual = rl.get_q((0.4, 0.03), 2)
    expected = 1.0601519022049946

    if not math.isclose(actual, expected, abs_tol=1e-6):
        raise AssertionError(
            'Weight update incorrect for a transition to a terminal state.'
        )
