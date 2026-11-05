from .riverswim import (
    RiverSwimAgent as RiverSwimAgent,
    RiverSwimEnv as RiverSwimEnv,
    RiverSwimModel as RiverSwimModel,
    RiverSwimResult as RiverSwimResult,
    register_custom_river_swim_env as register_custom_river_swim_env,
)

register_custom_river_swim_env()
del register_custom_river_swim_env
