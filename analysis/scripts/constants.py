from os import environ
from pathlib import Path

from covvvr.functions import (
    AnnulusWCuts,
    EntangledCircles,
    NCamel,
    NGauss,
    NPolynomial,
    ScalarTopLoop,
)

DATA_DIR = Path(environ["DATADIR"]) / "mc_var"
FUNCTIONS = (
    [NGauss(d) for d in [2, 4, 8, 16]]
    + [NCamel(d) for d in [2, 4, 8, 16]]
    + [EntangledCircles(), AnnulusWCuts(), ScalarTopLoop()]
    + [NPolynomial(d) for d in [18, 54, 96]]
)
