from importlib.metadata import version

from .cvintegrator import CVIntegrator, quick_integrate
from .functions import make_func

__all__ = ["make_func", "CVIntegrator", "quick_integrate"]
__version__ = version("congas")
