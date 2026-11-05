import importlib.util

if importlib.util.find_spec('gymnasium') is None:
    raise ModuleNotFoundError(
        'Missing optional dependency `gymnasium` required by dnnlpy.cs234.*. '
        'Install it with `pip install gymnasium`.'
    )

from . import assignment1 as assignment1
