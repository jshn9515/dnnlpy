import importlib.util

if importlib.util.find_spec('gymnasium') is None:
    raise ModuleNotFoundError(
        'Missing optional dependency `gymnasium` required by dnnlpy.cs221.assignment4. '
        'Install it with `pip install gymnasium`.'
    )

from .algorithms import (
    BootstrappedPolicyGradient as BootstrappedPolicyGradient,
    ConstrainedQLearning as ConstrainedQLearning,
    FixedRLAlgorithm as FixedRLAlgorithm,
    FunctionApproxQLearning as FunctionApproxQLearning,
    ModelBasedMonteCarlo as ModelBasedMonteCarlo,
    PolicyGradient as PolicyGradient,
    RandomRLAlgorithm as RandomRLAlgorithm,
    RLAlgorithm as RLAlgorithm,
    TabularQLearning as TabularQLearning,
    ValueIteration as ValueIteration,
)
from .env import (
    CustomMountainCarEnv as CustomMountainCarEnv,
    NumberLineEnv as NumberLineEnv,
    register_custom_mountain_car_env as register_custom_mountain_car_env,
    register_number_line_env as register_number_line_env,
)
from .mdp import (
    MDP as MDP,
    ContinuousMountainCarMDP as ContinuousMountainCarMDP,
    DiscreteMountainCarMDP as DiscreteMountainCarMDP,
    MDPTransition as MDPTransition,
    MountainCarMDP as MountainCarMDP,
    NumberLineMDP as NumberLineMDP,
)
from .utils import (
    fourier_feature_extractor as fourier_feature_extractor,
    polynomial_feature_extractor as polynomial_feature_extractor,
    simulate as simulate,
)

register_number_line_env()
register_custom_mountain_car_env()
del register_number_line_env
del register_custom_mountain_car_env
