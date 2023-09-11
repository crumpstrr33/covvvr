from importlib.metadata import version

from .cvintegrator import CVIntegrator, classic_integrate
from .functions import make_func

__all__ = ["make_func", "CVIntegrator", "classic_integrate"]
__version__ = version("covvvr")
