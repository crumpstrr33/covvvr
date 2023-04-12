"""
Custom types for type hinting 
"""
from typing import Callable

from nptyping import Float, NDArray, Shape

# Type for input to the batchintegrand vegas functions
_x = NDArray[Shape["N, Dim"], Float]
# The batch integrand vegas function type
_ftype = Callable[[_x], NDArray[Shape["N"], Float]]
