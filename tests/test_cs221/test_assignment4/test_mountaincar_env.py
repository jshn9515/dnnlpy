import gymnasium as gym
import numpy as np
import numpy.testing as npt
import pytest
from gymnasium.utils.env_checker import check_env

import dnnlpy.cs221.assignment4 as a4


def test_custom_mountain_car_env():
    env = gym.make('CustomMountainCar-v0')
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_number_line_env_is_registered():
    env = gym.make('NumberLine-v0')
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize(
    'mdp_type', [a4.ContinuousMountainCarMDP, a4.DiscreteMountainCarMDP]
)
def test_mountain_car_step_tracks_current_state(mdp_type: type[a4.MountainCarMDP]):
    mdp = mdp_type(a4.CustomMountainCarEnv(seed=0))
    mdp.reset(seed=0)

    first = mdp.step(2)
    second = mdp.step(2)

    npt.assert_equal(second.state, first.next_state)
    assert second.terminal is False
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
        if transition.terminal:
            break

    assert len(visited) > 1
    assert mdp.reset(seed=0) == start_state
    mdp.close()


def test_continuous_mountain_car_uses_shaped_reward():
    mdp = a4.ContinuousMountainCarMDP(a4.CustomMountainCarEnv(seed=0))
    mdp.reset(seed=0)

    transition = mdp.step(2)
    reward = mdp.base_env.reward(transition.next_state)

    assert isinstance(transition.next_state, np.ndarray)
    assert transition.reward == pytest.approx(reward)
    mdp.close()
