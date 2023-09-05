"""
This file contains a dataclass for various functions, Each class has the following
properties and attributes:
    - name: Capitalized and spaced name
    - dimension/dim: The dimension of the integral
    - true_value: The actual value of the integral
    - _function/_f: The function that is actually called by Vegas
    - function/f: The function in question to integrate. Wraps _f/_function
        for direct use.
    - And any other parameters that the integrand may have.

_function/_f are vectorized to support batch integration by Vegas, so they take in
a Numpy array of lists of length `dim`. So f/function exist to be used manually by the
user. So running:

    f(1, 2, 3) == _f(np.array([[1], [2], [3]])) or
    f([1, 2], [3, 4]) = _f(np.array([[1, 2], [3, 4]]))

to cut down on clutter.

All functions are to be integrated from 0 to 1.
"""
from dataclasses import dataclass, field, fields, make_dataclass
from math import erf
from numbers import Number
from typing import Optional, Sequence, Union

import numpy as np
from vegas import batchintegrand

from ._exceptions import ParameterBoundError
from ._types import _f, _ftype, _x


# TODO: add getters and setters?
@dataclass(repr=False)
class Function:
    """
    Base dataclass for functions. When inheriting this class, the vectorized
    function _function is defined to override the dummy function defined here
    for Vegas's batchintegrand.

    Parameters:
    dimension - The dimension of the integral
    name - Capitalized and spaced name
    true_value (default 0) - The actual value of the integral

    dim, _f and f are equivalent to dimension, _function and function but are just
    shorter.
    """

    dimension: int
    name: str
    true_value: Optional[float] = None

    def __post_init__(self):
        """Shorthand property names"""
        self.dim = self.dimension
        self._f = self._function
        self.f = self.function

    def __repr__(self):
        """Doesn't output parameters that equal None"""
        parameters = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"{type(self).__name__}({', '.join(parameters)})"

    def _function(self, x: _x) -> _f:
        """
        Function to be called by Vegas's batch integration. Defined by inheriting
        classes.
        """
        pass

    def function(self, *x: Union[float, Sequence[float]]) -> _f:
        """
        A wrapper around self._f that allows for easier manual use. Just supply however
        many data points to evaluate as individual arguments and this method will
        deal with the proper array shapes needed.
        """
        if len(x) == 1:
            if isinstance(x[0], Number):
                # If just a single number is passed
                return self._function(np.array([[x]]))[0][0]
            else:
                # If just a single tuple of `dim` numbers are passed
                return self._function(np.array([x[0]]))[0]
        # If multiple args are passed
        return self._function(np.array(np.atleast_1d(*x)))


# TODO: Allow passing function being nonvectorized, use np.vectorize or too slow?
def make_func(
    cname: str,
    dimension: int,
    function: _ftype,
    name: Optional[str] = None,
    **params: float,
) -> Function:
    """
    Create your own dataclass function. The function provided must be vectorized.
    If the integral is n dimensional, then the input to `f` is (..., n) dimensional
    where ... is a variable amount representing how many points are being evaluated
    simultaneously. The callable `f` to be passed is the function to be integrated.
    Since batchintegration is being used from Vegas, this function must be vectorized.
    So the one dimensional function, `def f(x): return x**2 + 1`, would become:

                def f(self, x):
                    return x[:, 0]**2 + 1

    using Numpy array slicing (the `self` is also necessary since this is a class
    method). For an n-dimensional integrand, one can reference the ith variable with
    x[:, i]. Or if all variables are being summed together (like for the exponent of a
    n-dimensional Gaussian), one can use `np.sum(x, axis=1)`. More information on this
    and an example can be found in the file docstring of this file, functions.py.
    Also keyword arguments for the parameters that may be used in `f`. For example,
    if we pass to this method a=3, then we can write

                def f(self, x):
                    return self.a * (x[:, 0]**2 * x[:, 1] + 1)

    to parameterize it.

    Parameters:
    cname - The class name
    dimension - The dimension of the integral
    f - The function, must be vectorized. See above.
    name (default None) - The spaced, capitalized name for the function. If not given,
        will just use whatever `cname` is.
    params - Function parameters

    Returns (
        Initialized Function class
    )
    """
    name = cname if name is None else name
    # Dataclass fields
    fields = [(param, float, field(default=val)) for param, val in params.items()]

    # Create Function class then added new stuff
    func = Function(
        name=name,
        dimension=dimension,
    )
    func.__class__ = make_dataclass(
        cname,
        fields=fields,
        bases=(Function,),
        namespace={"_function": batchintegrand(function)},
        repr=False,
    )
    func._f = func._function
    return func


@dataclass(repr=False)
class NGauss(Function):
    """
    N-dimensional Gaussian

    Parameters:
    mu (default 0.5) - Mean of Gaussian, must be between 0 and 1
    sigma (default 0.2) - Standard deviation of Gaussian
    dimension - Dimension of Gaussian
    """

    mu: float = 0.5
    sigma: float = 0.2
    name: str = field(default="{}D Gaussian", init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.mu > 1 or self.mu < 0:
            raise ParameterBoundError(
                f"Mean must be between the bounds [0, 1]. Currently set at {self.mu}."
            ) from None

        self.true_value = (
            (erf((1 - self.mu) / self.sigma) + erf(self.mu / self.sigma)) / 2
        ) ** self.dim
        self.name = self.name.format(self.dim)

    @batchintegrand
    def _function(self, x: _x) -> _f:
        norm_factor = 1 / (self.sigma * np.sqrt(np.pi)) ** self.dim
        exp = -np.sum((x - 0.5) ** 2, axis=1) / self.sigma**2
        return norm_factor * np.exp(exp)


@dataclass(repr=False)
class NCamel(Function):
    """
    N-dimensional Camel (double Gaussian) Function

    Parameters:
    mu1 (default 1/3) - Mean of first Gaussian, must be between 0 and 1
    mu2 (default 2/3)- Mean of second Gaussian, must be between 0 to 1
    sigma (default 0.2) - Standard deviation of both Gaussians
    dimension - Dimension of Camel function
    """

    mu1: float = 1 / 3
    mu2: float = 2 / 3
    sigma: float = 0.2
    name: str = field(default="{}D Camel", init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.mu1 > 1 or self.mu1 < 0:
            raise ParameterBoundError(
                f"First mean must be between [0, 1]. Currently set at {self.mu1}."
            ) from None
        if self.mu2 > 1 or self.mu2 < 0:
            raise ParameterBoundError(
                f"Second mean must be between [0, 1]. Currently set at {self.mu2}."
            ) from None

        self.true_value = (
            (
                erf((1 - self.mu1) / self.sigma)
                + erf(self.mu1 / self.sigma)
                + erf((1 - self.mu2) / self.sigma)
                + erf(self.mu2 / self.sigma)
            )
            / 4
        ) ** self.dim
        self.name = self.name.format(self.dim)

    @batchintegrand
    def _function(self, x: _x) -> _f:
        norm_factor = 1 / (2 * (self.sigma * np.sqrt(np.pi)) ** (self.dim))
        exp1 = -np.sum((x - self.mu1) ** 2, axis=1) / self.sigma**2
        exp2 = -np.sum((x - self.mu2) ** 2, axis=1) / self.sigma**2
        return norm_factor * (np.exp(exp1) + np.exp(exp2))


@dataclass(repr=False)
class EntangledCircles(Function):
    """
    A function of two shifted circles. True value depends on default values.

    Parameters:
    p1, p2 (defaults 0.4, 0.6) - Shifts of the two variables
    r (default 0.25) - Radius
    w, a (default 1/0.004, 3)
    """

    p1: float = 0.4
    p2: float = 0.6
    r: float = 0.25
    w: float = 1 / 0.004
    a: float = 3
    dimension: int = field(default=2, init=False)
    name: str = field(default="Entangled Circles", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.true_value = 0.01368

    @batchintegrand
    def _function(self, x: _x) -> _f:
        x1 = x[:, 0]
        x2 = x[:, 1]
        exp1 = np.exp(
            -self.w * abs((x2 - self.p2) ** 2 + (x1 - self.p1) ** 2 - self.r**2)
        )
        exp2 = np.exp(
            -self.w
            * abs((x2 - 1 + self.p2) ** 2 + (x1 - 1 + self.p1) ** 2 - self.r**2)
        )
        return x2**self.a * exp1 + (1 - x2) ** self.a * exp2


@dataclass(repr=False)
class AnnulusWCuts(Function):
    """
    An annulus with hard cuts

    Parameters:
    rmin - Radius of inner circle
    rmax - Radius of outer circle
    """

    rmin: float = 0.2
    rmax: float = 0.45
    dimension: int = field(default=2, init=False)
    name: str = field(default="Annulus with Cuts", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.rminsq = self.rmin**2
        self.rmaxsq = self.rmax**2
        if self.rmin < 0:
            raise ParameterBoundError(
                f"Minimum radius must be positive. Currently set at {self.rmin}."
            ) from None
        if self.rmax < 0:
            raise ParameterBoundError(
                f"Maximum radius must be positive. Currently set at {self.rmax}."
            ) from None
        if self.rmin > self.rmax:
            raise ParameterBoundError(
                "Minimum radius must be less than maximum radius but they are "
                + f"{self.rmin} and {self.rmax} currently."
            ) from None
        self.true_value = np.pi / 4 * (self.rmaxsq - self.rminsq)

    @batchintegrand
    def _function(self, x: _x) -> _f:
        dist = x[:, 0] ** 2 + x[:, 1] ** 2
        return np.logical_and(dist > self.rminsq, dist < self.rmaxsq).astype(float)


@dataclass(repr=False)
class ScalarTopLoop(Function):
    """
    A one-loop scalar box integral. True value depends on default values.

    Parameters:
    s12, s23, s1, s2, s3, s4 (defaults 130**2, -130**2, 0, 0, 0, 125**2) =
        Parameters of function
    mtsq (default 173.9**2) - Square of top quark mass
    """

    s12: float = 130**2
    s23: float = -(130**2) / 2
    s1: float = 0
    s2: float = 0
    s3: float = 0
    s4: float = 125**2
    mtsq: float = 175**2
    dimension: int = field(default=3, init=False)
    name: str = field(default="Scalar Top Loop", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.true_value = 1.93741e-10

    def _Fbox(self, x, s12, s23, s1, s2, s3, s4):
        m1sq, m2sq, m3sq, m4sq = self.mtsq, self.mtsq, self.mtsq, self.mtsq
        return (
            -s12 * x[:, 1]
            - s23 * x[:, 0] * x[:, 2]
            - s1 * x[:, 0]
            - s2 * x[:, 0] * x[:, 1]
            - s3 * x[:, 1] * x[:, 2]
            - s4 * x[:, 2]
            + (1 + x[:, 0] + x[:, 1] + x[:, 2])
            * (x[:, 0] * m1sq + x[:, 1] * m2sq + x[:, 2] * m3sq + m4sq)
        )

    def _Sbox(self, x, s12, s23, s1, s2, s3, s4):
        return 1 / self._Fbox(x, s12, s23, s1, s2, s3, s4) ** 2

    @batchintegrand
    def _function(self, x: _x) -> _f:
        return (
            self._Sbox(x, self.s12, self.s23, self.s1, self.s2, self.s3, self.s4)
            + self._Sbox(x, self.s23, self.s12, self.s2, self.s3, self.s4, self.s1)
            + self._Sbox(x, self.s12, self.s23, self.s3, self.s4, self.s1, self.s2)
            + self._Sbox(x, self.s23, self.s12, self.s4, self.s1, self.s2, self.s3)
        )


@dataclass(repr=False)
class NPolynomial(Function):
    """
    N-dimensional Polynomial of the form: sum of -x_i^2 + x_i.

    Parameters:
    dimension - Dimension of Polynomial
    """

    name: str = field(default="{}D Polynomial", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.true_value = self.dim / 6
        self.name = self.name.format(self.dim)

    @staticmethod
    @batchintegrand
    def _function(x: _x) -> _f:
        return np.sum(-(x**2) + x, axis=1)
