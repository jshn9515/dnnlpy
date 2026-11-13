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


def test_number_line_constructor_seed_and_reset_override():
    env1 = a4.NumberLineEnv(n=100, seed=11)
    env2 = a4.NumberLineEnv(n=100, seed=11)
    env1.reset()
    env2.reset()

    assert env1.np_random_seed == env2.np_random_seed == 11
    assert [env1.step(0)[0] for _ in range(20)] == [env2.step(0)[0] for _ in range(20)]

    rng_state = env1.np_random.bit_generator.state
    env1.reset()
    assert env1.np_random.bit_generator.state == rng_state

    env1.reset(seed=0)
    assert env1.np_random_seed == 0

    env1.close()
    env2.close()


def test_mountain_car_constructor_seed_and_reset_override():
    env1 = a4.CustomMountainCarEnv(seed=11)
    env2 = a4.CustomMountainCarEnv(seed=11)

    x1, _ = env1.reset()
    y1, _ = env2.reset()
    x2, _ = env1.reset()
    y2, _ = env2.reset()

    npt.assert_equal(x1, y1)
    npt.assert_equal(x2, y2)
    assert not np.array_equal(x1, x2)
    assert not np.array_equal(y1, y2)

    z1, _ = env1.reset(seed=0)
    env3 = a4.CustomMountainCarEnv()
    z2, _ = env3.reset(seed=0)

    npt.assert_equal(z1, z2)
    assert env3.np_random_seed == 0

    env1.close()
    env2.close()
    env3.close()


def test_number_line_reward_and_invalid_actions():
    env = a4.NumberLineEnv(n=1, left_reward=7, right_reward=11, seed=0)

    assert env.states == [-1, 0, 1]
    assert env.actions == [0, 1]
    assert env.reward(-1) == 7
    assert env.reward(1) == 11

    with pytest.raises(RuntimeError, match='Invalid action'):
        env.step(2)

    env.reset(seed=0)
    state, reward, terminated, truncated, _ = env.step(0)

    assert state in (-1, 1)
    assert reward == env.reward(state)
    assert terminated and not truncated

    with pytest.raises(RuntimeError, match='terminated environment'):
        env.step(0)

    env.close()
