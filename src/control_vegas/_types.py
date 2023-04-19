"""
Custom types for type hinting 
"""
from typing import Callable

from nptyping import Float, NDArray, Shape

# Type for input to the batchintegrand vegas functions
_x = NDArray[Shape["'* batchSize, Dim dimension'"], Float]
_f = NDArray[Shape["'* batchSize'"], Float]
# The batch integrand vegas function type
_ftype = Callable[[_x], _f]
