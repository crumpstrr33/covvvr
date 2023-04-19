from importlib.metadata import version

from .functions import make_func
from .mccv import CVIntegrator, quick_integrate
from .utilities import save

__all__ = ["make_func", "CVIntegrator", "quick_integrate", "save"]
__version__ = version("control_vegas")
