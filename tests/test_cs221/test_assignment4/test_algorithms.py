import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4
import dnnlpy.cs221.assignment4.algorithms as algo


def test_softmax_is_stable_along_selected_axis():
    values = np.array([[1000.0, 1000.0], [-1000.0, -999.0]])
    result = algo.softmax(values, axis=1)

    npt.assert_allclose(result.sum(axis=1), [1.0, 1.0])
    npt.assert_allclose(result[0], [0.5, 0.5])

    assert np.all(np.isfinite(result))


def test_value_iteration_masks_invalid_actions_and_empty_states():
    transitions = np.zeros((3, 2, 3))
    rewards = np.zeros_like(transitions)
    transitions[0, 0, 1] = 1
    transitions[0, 1, 2] = 1
    rewards[0, 0, 1] = 2
    rewards[0, 1, 2] = 20
    mask = np.array([[True, False], [False, False], [False, False]], dtype=np.bool)

    vi = a4.ValueIteration(transitions, rewards, discount=0.9, action_mask=mask)
    npt.assert_allclose(vi.get_q()[0, 0], 2)
    npt.assert_allclose(vi.get_q()[0, 1], -np.inf)
    npt.assert_equal(vi.compute_policy(), [0, -1, -1])
    npt.assert_allclose(vi.values, [2, 0, 0])


def test_model_based_agent_estimates_transition_and_reward():
    agent = a4.ModelBasedMonteCarlo(
        [0, 1], discount=0.9, num_states=2, calc_val_iter_every=1, seed=0
    )
    agent.num_iters = 1
    agent.step(a4.MDPTransition(0, 1, 4, 1, True, False))

    assert agent.transition_counts[0, 1, 1] == 1
    assert agent.reward_sums[0, 1, 1] == 4

    npt.assert_equal(agent.pi_indices, [1, -1])
    assert agent.get_action(0, explore=False) == 1


def test_number_line_indices_work_with_tabular_learning():
    mdp = a4.NumberLineMDP(a4.NumberLineEnv(n=1))
    agent = a4.TabularQLearning(mdp.actions, mdp.discount, mdp.num_states, epsilon=0)

    a4.simulate(mdp, agent, num_episodes=1, verbose=False)
    assert np.count_nonzero(agent.Q[mdp.n]) == 1

    mdp.close()


@pytest.mark.parametrize(
    ('terminated', 'truncated', 'expected'),
    [
        (True, False, 1.0),
        (False, True, 1.5),
    ],
)
def test_tabular_q_learning_bootstraps_only_on_nonterminal_transitions(
    terminated: bool, truncated: bool, expected: float
):
    agent = a4.TabularQLearning([0, 1], discount=0.5, num_states=2, initial_q=1.0)
    agent.Q[1] = [10.0, 0.0]

    agent.step(a4.MDPTransition(0, 1, 1.0, 1, terminated, truncated))
    assert agent.Q[0, 1] == pytest.approx(expected)


def test_function_approx_q_values_and_terminal_update():
    features = lambda state: np.array([1.0, state], dtype=np.float32)
    agent = a4.FunctionApproxQLearning(2, features, [0, 1], discount=0.9, seed=0)
    agent.weight[:] = np.array([[1.0, 2.0], [3.0, 4.0]])

    npt.assert_allclose(agent.get_vector_q(2.0), [7.0, 10.0])
    assert agent.get_single_q(2.0, 1) == 10.0

    weight = agent.weight.copy()
    agent.step(a4.MDPTransition(2.0, 1, 1.0, 3.0, True, False))

    npt.assert_allclose(agent.weight[:, 0], weight[:, 0])
    npt.assert_allclose(
        agent.weight[:, 1], weight[:, 1] - 9 * agent.step_size * features(2.0)
    )


def test_linear_value_function_distinguishes_truncation_from_termination():
    features = lambda state: np.array([1.0, state], dtype=np.float32)
    value = algo.LinearValueFunction(2, features, step_size=0.1, discount=0.5)
    value.weight[:] = np.array([2.0, 1.0])

    value.update(a4.MDPTransition(0.0, 0, 1.0, 2.0, False, True))
    npt.assert_allclose(value.weight, [2.1, 1.0])

    value.update(a4.MDPTransition(0.0, 0, 1.0, 2.0, True, False))
    npt.assert_allclose(value.weight, [1.99, 1.0])


def test_policy_gradient_flushes_truncated_rollout():
    features = lambda state: np.array([1.0])
    agent = a4.PolicyGradient(1, features, [0, 1], time_limit=10, seed=0)

    agent.step(a4.MDPTransition(0, 0, 1.0, 1, False, True))
    assert agent.rollout == []
    assert agent.baseline == 1.0
