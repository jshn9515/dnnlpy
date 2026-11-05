from typing import Any, Literal, NoReturn, override

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv

type RenderMode = Literal['human', 'rgb_array']
type Transition[StateT] = tuple[StateT, float, bool, bool, dict[str, Any]]

__all__ = [
    'CustomMountainCarEnv',
    'NumberLineEnv',
]


class NumberLineEnv(gym.Env):
    """A simple environment representing a number line MDP.

    The agent starts at the center of a number line and moves left or right based on the
    selected action's transition probabilities. Each transition produces a reward, and
    the episode ends when either boundary is reached.

    Args:
        n (int, optional): The maximum absolute value of the states. The states will be
            in the range ``[-n, n]``. Defaults to 2.
        left_reward (float, optional): The reward for reaching the left terminal state
            ``-n``. Defaults to 10.
        right_reward (float, optional): The reward for reaching the right terminal state
            ``n``. Defaults to 50.
        penalty (float, optional): The penalty for reaching any non-terminal state.
            Defaults to -5.
        seed (int, optional): The seed for the random number generator. If ``None``, the
            random number generator will be initialized randomly. Defaults to ``None``.
    """

    def __init__(
        self,
        n: int = 2,
        left_reward: float = 10,
        right_reward: float = 50,
        penalty: float = -5,
        *,
        seed: int | None = None,
    ) -> None:
        """Initialize the ``NumberLineEnv`` with the given parameters."""
        super().__init__()
        self.n = n
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.penalty = penalty

        # -n, ..., 0, ..., n
        self.observation_space = spaces.Discrete(2 * n + 1, start=-n, seed=seed)
        self._num_states = int(self.observation_space.n)

        # 0: move right, 1: move left
        self.action_space = spaces.Discrete(2, seed=seed)
        self._num_actions = int(self.action_space.n)

        self.state = 0
        self.terminal_states = {-n, n}

    @property
    def actions(self) -> list[int]:
        """Returns the list of discrete actions in the MDP."""
        return list(range(self._num_actions))

    @property
    def num_actions(self) -> int:
        """Returns the number of actions in the environment."""
        return self._num_actions

    @property
    def states(self) -> list[int]:
        """Returns the list of discrete states in the MDP."""
        return list(range(-self.n, self.n + 1))

    @property
    def num_states(self) -> int:
        """Returns the number of states in the environment."""
        return self._num_states

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def reward(self, state: int) -> float:
        """Compute the reward for transitioning to the given state.

        Args:
            state (int): The state after the transition.

        Returns:
            float: The reward for the transition.
        """
        if state == self.n:
            return self.right_reward
        elif state == -self.n:
            return self.left_reward
        else:
            return self.penalty

    @override
    def step(self, action: int) -> Transition[int]:
        """Execute an action and return the resulting transition."""
        if not self.action_space.contains(action):
            raise RuntimeError(f'Invalid action: {action}.')

        if self.state in self.terminal_states:
            raise RuntimeError('Attempting to step in a terminated environment.')

        # Preserve the original transition probabilities.
        forward_prob = 0.2 if action == 0 else 0.3

        if self.np_random.random() < forward_prob:
            self.state += 1
        else:
            self.state -= 1

        reward = self.reward(self.state)
        terminated = self.state in self.terminal_states
        return self.state, reward, terminated, False, {}


class CustomMountainCarEnv(MountainCarEnv):
    """A custom implementation of the MountainCar environment with adjustable parameters.

    This class extends the original MountainCarEnv from Gymnasium, allowing users to
    customize various parameters such as position limits, speed limits, goal position,
    and more. It also modifies the reward function to provide a shaped reward based on
    the car's position and velocity, rather than the original sparse reward structure.

    For more details on the original MountainCar environment, refer to the
    :class:`MountainCarEnv` documentation in the Gymnasium library.

    Args:
        min_position (float, optional): Minimum position of the car. Defaults to -1.2.
        max_position (float, optional): Maximum position of the car. Defaults to 0.6.
        max_speed (float, optional): Maximum speed of the car. Defaults to 0.07.
        goal_position (float, optional): Position at which the episode is considered
            successful. Defaults to 0.5.
        goal_velocity (float, optional): Velocity at which the episode is considered
            successful. Defaults to 0.0.
        force (float, optional): The force applied to the car when an action is taken.
            Defaults to 0.001.
        gravity (float, optional): The gravitational constant affecting the car's movement.
            Defaults to 0.0025.
        render_mode (RenderMode, optional): The mode in which to render the environment.
            Can be 'human' for rendering to the screen, 'rgb_array' for returning an RGB
            array, or ``None`` for no rendering. Defaults to ``None``.
        screen_width (int, optional): Width of the rendering window in pixels. Defaults
            to 800.
        screen_height (int, optional): Height of the rendering window in pixels. Defaults
            to 600.
        seed (int, optional): Seed for the random number generator. If ``None``, the random
            number generator will be initialized randomly. Defaults to ``None``.
    """

    def __init__(
        self,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        goal_position: float = 0.5,
        goal_velocity: float = 0.0,
        force: float = 0.001,
        gravity: float = 0.0025,
        *,
        render_mode: RenderMode | None = None,
        screen_width: int = 800,
        screen_height: int = 600,
        seed: int | None = None,
    ):
        """Initializes the ``CustomMountainCarEnv`` with the specified parameters."""
        super().__init__(render_mode)
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.goal_velocity = goal_velocity
        self.force = force
        self.gravity = gravity

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        # Box space: (min_position, -max_speed) to (max_position, max_speed)
        self.observation_space = spaces.Box(self.low, self.high, seed=seed)

        # 0: push left, 1: no push, 2: push right
        self.action_space = spaces.Discrete(3, seed=seed)
        self._num_actions = int(self.action_space.n)

        self.screen_width = screen_width
        self.screen_height = screen_height

    @property
    def actions(self) -> list[int]:
        """Returns the list of discrete actions in the MDP."""
        return list(range(self._num_actions))

    @property
    def num_actions(self) -> int:
        """Returns the number of actions in the environment."""
        return self._num_actions

    @property
    def states(self) -> NoReturn:
        """Returns the list of discrete states in the MDP."""
        raise NotImplementedError('Observation space is not discrete.')

    @property
    def num_states(self) -> NoReturn:
        """Returns the number of states in the environment."""
        raise NotImplementedError('Observation space is not discrete.')

    def reward(self, state: np.ndarray) -> float:
        """Compute the reward for transitioning to the given continuous state.

        Replaces the original reward with a shaped reward based on the car's position relative
        to the maximum position and the magnitude of its velocity.

        Args:
            state (np.ndarray): Continuous state after the transition, containing the car's
                position and velocity.

        Returns:
            float: Shaped reward for MountainCar environments, or the original reward for
            other environments.
        """
        position_reward = -(self.high[0] - state[0])
        velocity_reward = -(self.high[1] - np.abs(state[1]))
        return position_reward + velocity_reward

    @override
    def step(self, action: int) -> Transition[np.ndarray]:
        """Takes action in the current state and returns :class:`MDPTransition`."""
        state, _, terminated, truncated, _ = super().step(action)
        reward = self.reward(state)
        return state, reward, terminated, truncated, {}


def register_number_line_env():
    """Registers the ``NumberLine-v0`` environment in the Gym registry."""
    if 'NumberLine-v0' not in gym.registry:

        def factory(**kwargs: Any) -> NumberLineEnv:
            return NumberLineEnv(**kwargs)

        gym.register(
            id='NumberLine-v0',
            entry_point=factory,
            max_episode_steps=100,
        )


def register_custom_mountain_car_env():
    """Registers the ``CustomMountainCar-v0`` environment in the Gym registry."""
    if 'CustomMountainCar-v0' not in gym.registry:

        def factory(**kwargs: Any) -> CustomMountainCarEnv:
            return CustomMountainCarEnv(**kwargs)

        gym.register(
            id='CustomMountainCar-v0',
            entry_point=factory,
            max_episode_steps=1000,
        )
