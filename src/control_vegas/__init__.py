from importlib.metadata import version

from .cvintegrator import CVIntegrator, quick_integrate
from .functions import make_func
from .utilities import save

__all__ = ["make_func", "CVIntegrator", "quick_integrate", "save"]
__version__ = version("control_vegas")
