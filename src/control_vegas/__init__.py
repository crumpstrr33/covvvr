from importlib.metadata import version

from .functions import make_func
from .mccv import CVIntegrator
from .utilities import save

__version__ = version("control_vegas")
