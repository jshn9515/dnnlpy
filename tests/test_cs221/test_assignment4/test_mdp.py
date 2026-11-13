import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4


def test_number_line_mdp_transforms_observations_to_indices():
    mdp = a4.NumberLineMDP(a4.NumberLineEnv(n=1), time_limit=2)

    assert mdp.states == [0, 1, 2]
    assert mdp.num_states == 3
    assert mdp.reset(seed=0) == 1

    transition = mdp.step(0)
    assert transition.state == 1
    assert transition.next_state == mdp.base_env.state + mdp.n
    assert transition.next_state in (0, 2)
    assert transition.terminated and not transition.truncated

    mdp.close()


def test_number_line_time_limit_marks_truncation():
    mdp = a4.NumberLineMDP(a4.NumberLineEnv(n=2, seed=0), time_limit=1)
    mdp.reset(seed=0)
    transition = mdp.step(0)

    assert transition.truncated is True
    assert transition.terminated is False

    mdp.close()


@pytest.mark.parametrize(
    'mdp_type', [a4.ContinuousMountainCarMDP, a4.DiscreteMountainCarMDP]
)
def test_mountain_car_step_tracks_current_state(mdp_type: type[a4.MountainCarMDP]):
    mdp = mdp_type(a4.CustomMountainCarEnv(seed=0))
    mdp.reset(seed=0)

    first = mdp.step(2)
    second = mdp.step(2)

    npt.assert_equal(first.next_state, second.state)
    assert second.terminated is False

    mdp.close()


def test_discrete_mountain_car_tracks_states_across_multiple_bins():
    mdp = a4.DiscreteMountainCarMDP(a4.CustomMountainCarEnv(seed=0), bins=20)
    start_state = mdp.reset(seed=0)
    visited = {start_state}

    for _ in range(100):
        transition = mdp.step(2)

        assert transition.state in visited
        assert 0 <= transition.next_state < mdp.num_states

        visited.add(transition.next_state)
        if transition.terminated or transition.truncated:
            break

    assert len(visited) > 1
    assert mdp.reset(seed=0) == start_state

    mdp.close()


def test_continuous_mountain_car_uses_shaped_reward():
    mdp = a4.ContinuousMountainCarMDP(a4.CustomMountainCarEnv(seed=0))
    mdp.reset(seed=0)
    transition = mdp.step(2)

    assert isinstance(transition.next_state, np.ndarray)
    assert transition.reward == pytest.approx(
        mdp.base_env.reward(transition.next_state)
    )

    mdp.close()


def test_mountain_car_time_limit_marks_truncation():
    mdp = a4.ContinuousMountainCarMDP(a4.CustomMountainCarEnv(seed=0), time_limit=1)
    mdp.reset(seed=0)
    transition = mdp.step(2)

    assert transition.truncated is True
    assert transition.terminated is False

    mdp.close()
