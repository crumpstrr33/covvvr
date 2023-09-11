from importlib.metadata import version

from ._cvitime import CVITime
from .cvintegrator import CVIntegrator, classic_integrate
from .functions import make_func

__all__ = ["make_func", "CVIntegrator", "classic_integrate", "CVITime"]
__version__ = version("covvvr")
