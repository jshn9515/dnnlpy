from typing import Any, Literal, NamedTuple, override

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

type RiverCurrent = Literal['weak', 'medium', 'strong']

__all__ = [
    'RiverSwimAgent',
    'RiverSwimEnv',
    'RiverSwimModel',
    'RiverSwimResult',
]


class RiverSwimModel(NamedTuple):
    """A named tuple representing the RiverSwim MDP model.

    Attributes:
        P (np.ndarray): The transition function of shape ``(num_states, num_actions,
            num_states)``.
        R (np.ndarray): The reward function of shape ``(num_states, num_actions)``.
    """

    P: np.ndarray
    R: np.ndarray


class RiverSwimResult(NamedTuple):
    """A named tuple representing the result of running an algorithm on the RiverSwim MDP.

    Attributes:
        policy (np.ndarray): The optimal policy found.
        val_func (np.ndarray): The value function corresponding to the optimal policy.
    """

    policy: np.ndarray
    val_func: np.ndarray


class RiverSwimEnv(gym.Env):
    """Create a RiverSwim MDP environment for CS234 Assignment 1.

    RiverSwim is a simple MDP with 6 states and 2 actions. The agent starts in state 0 and can
    either swim left (action 0) or swim right (action 1). The goal is to reach the rightmost
    state (state 5) to receive a high reward, while swimming against the current makes it more
    challenging.

    The environment is parameterized by the strength of the river current, which can be 'weak',
    'medium', or 'strong'. The current affects the transition probabilities when swimming
    right. The reward function is designed to encourage the agent to reach the rightmost state,
    while also providing a small reward for staying in the leftmost state.

    For more details, refer to the Gymnasium documentation:
    `Mountain Car <https://gymnasium.farama.org/environments/classic_control/mountain_car/>`_.

    Args:
        current (RiverCurrent): The strength of the river current ('weak', 'medium', or 'strong').
        seed (int, optional): The seed for the random number generator. Defaults to 1234.
    """

    def __init__(self, current: RiverCurrent, seed: int = 1234):
        super().__init__()
        self.num_states = 6
        self.num_actions = 2

        current_list = ['weak', 'medium', 'strong']
        self.current = current_list.index(current) + 1

        self.observation_space = spaces.Discrete(self.num_states, dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_actions, dtype=np.int32)

        # Configure reward function
        R = np.zeros((self.num_states, self.num_actions))
        R[0, 0] = 0.005
        R[5, 1] = 1.0

        # Configure transition function
        P = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Boundary states
        P[0, 0, 0] = 1.0
        P[0, 1, 0] = 0.6
        P[0, 1, 1] = 0.4

        P[5, 1, 5] = 0.6
        P[5, 1, 4] = 0.4
        P[5, 0, 4] = 1.0

        # Intermediate states
        for s in range(1, self.num_states - 1):
            l, r = 0, 1

            P[s, l, s - 1] = 1.0

            P[s, r, s] = 0.6
            P[s, r, s - 1] = 0.09 * self.current
            P[s, r, s + 1] = 0.4 - P[s, r, s - 1]

            assert np.isclose(np.sum(P[s, l]), 1.0)
            assert np.isclose(np.sum(P[s, r]), 1.0)

        self.R = R
        self.P = P
        self.R.setflags(write=False)
        self.P.setflags(write=False)

        self.init_state = 0
        self.curr_state = self.init_state

        # Initialize Gymnasium's random generator
        self.reset(seed=seed)

    @property
    def model(self) -> RiverSwimModel:
        """Return copies of transition and reward matrices."""
        return RiverSwimModel(self.P.copy(), self.R.copy())

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            tuple[int, dict]: A tuple containing the initial state and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.curr_state = self.init_state
        return self.init_state, {}

    @override
    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Execute a given action and transition to the next state.

        Args:
            action (int): The action to take in the current state. 0 for swimming left, 1 for
                swimming right.

        Returns:
            tuple[int, float, bool, bool, dict]: A tuple containing the next state, the reward
            received, a boolean indicating if the episode is done, a boolean indicating if the
            episode was truncated, and an empty info dictionary.

        .. note::

            The RiverSwim environment is an infinite-horizon MDP, so the ``done`` and ``truncated``
            flags are always ``False``.
        """
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}.')

        # Reward depends on current state and action
        reward = float(self.R[self.curr_state, action])

        # Sample next state
        prob = self.P[self.curr_state, action]
        next_state = self.np_random.choice(self.num_states, p=prob)
        self.curr_state = next_state

        return next_state, reward, False, False, {}


class RiverSwimAgent:
    """An agent that runs PI and VI algorithms on the RiverSwim MDP.

    Args:
        env (RiverSwimEnv): The RiverSwim environment to interact with.
        gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.
        tol (float, optional): The tolerance for convergence in policy evaluation and value
            iteration. Defaults to 1e-3.
    """

    def __init__(self, env: RiverSwimEnv, gamma: float = 0.9, tol: float = 1e-3):
        """Initialize the ``RiverSwim`` agent with the given MDP environment."""
        self.env = env
        self.num_states = env.num_states
        self.num_actions = env.num_actions

        if not (0 <= gamma < 1):
            raise AssertionError('Discount factor must be in [0, 1).')
        self.gamma = gamma
        self.tol = tol

        self.R = env.R
        self.P = env.P

    def bellman_backup(self, state: int, action: int, V: np.ndarray) -> float:
        r"""Performs a Bellman backup for a given state and action.

        The Bellman backup is defined as:

        .. math::

            Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s, a, s') V(s')

        Args:
            state (int): The current state.
            action (int): The action taken in the current state.
            V (np.ndarray): The value function of shape ``(num_states,)``.

        Returns:
            float: The updated value for the given state and action.
        """
        return self.R[state, action] + self.gamma * np.dot(self.P[state, action], V)

    def policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        r"""Perform policy evaluation for a given **deterministic** policy.

        The value function for a policy is defined as:

        .. math::

            V_{k+1}(s) = Q_k(s, \pi(s))

        Args:
            policy (np.ndarray): The policy to evaluate.

        Returns:
            np.ndarray: The value function for the given policy :math:`\pi`.

        Example:
            >>> agent = RiverSwimAgent(RiverSwimEnv('weak'))
            >>> policy = np.array([1] * agent.num_states)
            >>> V = agent.policy_evaluation(policy)
            >>> V.shape
            (6,)
        """
        old_V = np.zeros(self.num_states)

        while True:
            new_V = np.array(
                [
                    self.bellman_backup(state, policy[state], old_V)
                    for state in range(self.num_states)
                ]
            )

            # The same as `np.linalg.norm(new_V - old_V, ord=np.inf)`
            if np.max(np.abs(new_V - old_V)) < self.tol:
                return new_V

            old_V = new_V

    def policy_improvement(self, V_pi: np.ndarray) -> np.ndarray:
        r"""Compute the improved policy given the value function of the current policy.

        The improved policy is defined as:

        .. math::

            \pi'(s) = \arg\max_a Q^\pi(s, a)

        Args:
            V_pi (np.ndarray): The value function of the current policy :math:`\pi`.

        Returns:
            np.ndarray: The improved policy :math:`\pi'`.

        Example:
            >>> agent = RiverSwimAgent(RiverSwimEnv('weak'))
            >>> old_policy = np.array([1] * agent.num_states)
            >>> V_pi = agent.policy_evaluation(old_policy)
            >>> new_policy = agent.policy_improvement(V_pi)
            >>> new_policy.shape
            (6,)
        """
        new_policy = np.zeros(self.num_states, dtype=np.int8)

        for state in range(self.num_states):
            Q_pi = np.array(
                [
                    self.bellman_backup(state, action, V_pi)
                    for action in range(self.num_actions)
                ]
            )
            new_policy[state] = np.argmax(Q_pi)

        return new_policy

    def policy_iteration(self) -> RiverSwimResult:
        r"""Perform Policy Iteration to find the optimal policy for the RiverSwim MDP.

        The algorithm alternates between policy evaluation and policy improvement until
        convergence.

        Returns:
            RiverSwimResult: The optimal policy found by policy iteration.

        Example:
            >>> agent = RiverSwimAgent(RiverSwimEnv('weak'))
            >>> result = agent.policy_iteration()
            >>> result.policy.shape
            (6,)
        """
        old_policy = np.zeros(self.num_states, dtype=np.int8)

        while True:
            V_pi = self.policy_evaluation(old_policy)
            new_policy = self.policy_improvement(V_pi)

            if np.array_equal(new_policy, old_policy):
                break

            old_policy = new_policy

        return RiverSwimResult(new_policy, V_pi)

    def value_iteration(self) -> RiverSwimResult:
        r"""Perform Value Iteration to find the optimal policy for the RiverSwim MDP.

        The algorithm iteratively updates the value function until convergence, and then
        derives the optimal policy from the final value function.

        .. math::

            \begin{aligned}
                Q_k(s, a) &= R(s, a) + \gamma \sum_{s'} P(s, a, s') V_k(s') \\
                V_{k+1}(s) &= \max_a Q_k(s, a)
            \end{aligned}

        Returns:
            RiverSwimResult: The optimal policy found by value iteration.

        Example:
            >>> agent = RiverSwimAgent(RiverSwimEnv('weak'))
            >>> result = agent.value_iteration()
            >>> result.policy.shape
            (6,)
        """
        old_V = np.zeros(self.num_states)

        while True:
            new_V = np.array(
                [
                    max(
                        self.bellman_backup(state, action, old_V)
                        for action in range(self.num_actions)
                    )
                    for state in range(self.num_states)
                ]
            )

            # The same as `np.linalg.norm(new_V - old_V, ord=np.inf)`
            if np.max(np.abs(new_V - old_V)) < self.tol:
                break

            old_V = new_V

        # One more policy improvement step to get the optimal policy.
        new_policy = self.policy_improvement(new_V)
        return RiverSwimResult(new_policy, new_V)


def register_custom_river_swim_env():
    """Registers the ``RiverSwimEnv-v0`` environment in the Gym registry."""
    if 'RiverSwimEnv-v0' not in gym.registry:

        def factory(**kwargs: Any) -> RiverSwimEnv:
            return RiverSwimEnv(**kwargs)

        gym.register(
            id='RiverSwimEnv-v0',
            entry_point=factory,
            max_episode_steps=1000,
        )
