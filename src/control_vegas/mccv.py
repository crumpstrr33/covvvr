"""
File that holds the CVIntegrator class
"""

from copy import deepcopy
from itertools import product
from numbers import Number
from re import findall
from typing import Optional, Sequence, Union

import numpy as np
from nptyping import Float, NDArray, Shape
from numpy.random import RandomState
from vegas import Integrator

from ._types import _ftype
from .functions import Function, make_func
from .utilities import check_value, timing

# Should you print out the time it takes for the main 3 functions to run?
TIMING = False


def quick_integrate(
    function: _ftype,
    evals: int,
    tot_iters: int,
    bounds: Union[Sequence[tuple[float, float]], tuple[float, float]],
    cv_iters: Optional[Union[list[int], int, str]] = None,
    cv_means: Union[float, Sequence[float]] = 1,
    rng: Optional[RandomState] = None,
    cname: str = None,
    name: str = None,
    **params: float,
):
    """
    A convenience method if you don't want to create a custom Function class via
    make_func. This method will do it for you, pass the class to the CVIntegrator and
    run the integrate method, passing back the CVIntegrator.

    Parameters:
    function - The function to be integrated. Must be vectorized, see the docstring for
        make_func for more info (from control_vegas import make_func).
    evals - Number of Vegas evaluations per iteration (called `neval` by Vegas). This
        is the default value used by create_maps, get_is_cv_values but those can be
        specified separately.
    tot_iters - Total number of iterations for Vegas to do (called `nitn` by Vegas).
    bounds - The bounds of the integration for each dimension given as a list of tuples.
        The CVIntegrator class uses the dimension from the Function class so that the
        bounds argument is optional. But here it is opposite, the dimension of the
        function is implied from the number of bounds.
    cv_iters - List of iterations to use as CVs. See the docstring for CVIntegrator
        for more information about the options. Most simply can be an integer
        representing a single CV or a list of integers representing multiple.
    cv_means (default 1) - The value of E[g_i] but by the scheme laid out in
        `get_is_cv_values` to obtain the control variate, E[g_i] should be approximately
        one.
    rng (default None) - The Numpy Randomstate to use. If None, will create a new
        one.
    cname (default None) - The name of the class of the Function passed to CVIntegrator.
        If not specified, the __name__ attribute of `function` is passed capitalized.
    name (default None) - The name attribute of the Function. If not specified,
        the __name__ attribute is passed as is.
    params - Parameters for the function.
    """
    # make Function object
    function = make_func(
        cname=function.__name__.capitalize() if cname is None else cname,
        dimension=len(bounds),
        function=function,
        name=function.__name__ if name is None else name,
        **params,
    )
    # Create integrator object
    cvi = CVIntegrator(
        function=function,
        evals=evals,
        tot_iters=tot_iters,
        bounds=bounds,
        cv_iters=cv_iters,
        cv_means=cv_means,
        rng=rng,
    )
    # And integrate
    cvi.integrate()
    return cvi


# TODO: Automatically choose single CV: find the variance dip
class CVIntegrator:
    """
    Integrating a function f, we can equivalently integrate f'=f + c(g +E[g]) where the
    expectation value of g is known. Choosing the optimal value for c will necessarily
    reduce the variance. We can have an arbitrary number of g's (called control variates
    (CVs)). Here we use a previous Vegas adapted interation of f for g. So if we have a
    total of N iterations of f, then we can have a maximum of N-1 CVs.

    Below, f' is called self.weight_prime (or the 'CV function'), f is self.weight_value
    (or the 'adapted/IS function') and the CVs (the g_i's) are elements of the list
    self.cv_values (called CVs). These are the terminology used in the docstrings below.

    Explanation of parameters:
        neval - Passed on instantiation of class, check __init__ docstring for details.
        nitn - The number of iterations the Integrator is ran to get the adapted
            function.
        cv_nums - The number of control variates.
        map_neval - The number of evaluations per iteration when initially creating the
            map. Defaults to neval.
        jac_neval - The size of the Jacobian arrays, i.e. how finely split up the
            integration region is. Defaults to neval*nitn
        tot_neval - The total number of evaluations done when creating the fully adapted
            map.
    """

    def __init__(
        self,
        function: Function,
        evals: int,
        tot_iters: int,
        bounds: Optional[Sequence[tuple[float, float]]] = None,
        cv_iters: Optional[Union[list[int], int, str]] = None,
        cv_means: Union[float, Sequence[float]] = 1,
        rng: Optional[RandomState] = None,
    ):
        """
        Takes in a Function class object from functions.py. One can make their own using
        the make_func function found in that file.

        Parameters:
        f - Function class with f to integrate.
        evals - Number of Vegas evaluations per iteration (called `neval` by Vegas).
            This is the default value used by create_maps, get_is_cv_values but those
            can be specified separately.
        tot_iters - Total number of iterations for Vegas to do (called `nitn` by Vegas).
        bounds (default None) - The bounds of the integration for each dimension. If not
            given, defaults to [0, 1] for every dimension.
        cv_iters (default None) - List of iterations to use as CVs. Defaults to no CVs.
            Can be passed as a single integer which is considered as a single control
            variate. Can also be passed as a string:
                - 'all': Use every iteration as a control variate.
                - 'all%n': Use every iteration mod n. For example, if tot_iters=10
                    and cv_iters='all%2', then it uses [2, 4, 6, 8]
                - 'all%n+b': Use every iteration (shifted by b) mod n. For example, if
                    tot_iters=10 and cv_iters='all%2+1', then use [1, 3, 5, 7, 9]
        cv_means (default 1) - The value of E[g_i] but by the scheme laid out in
            `get_is_cv_values` to obtain the control variate, E[g_i] should be
            approximately one.
        rng (default None) - The Numpy Randomstate to use. If None, will create a new
            one.
        """
        self.function = function
        self.bounds = self.function.dim * [[0, 1]] if bounds is None else bounds
        self.neval = evals
        self.nitn = tot_iters

        self.cv_nitn = cv_iters
        # Create empty list if not specified, i.e. no control variates
        if self.cv_nitn is None:
            self.cv_nitn = []
        # If cv_iters is a number, put it into a list
        if isinstance(self.cv_nitn, int):
            self.cv_nitn = [self.cv_nitn]
        if isinstance(self.cv_nitn, str):
            # Find the mod and shift using regex
            all_str = findall(r"^all%(\d+)(?:\+(\d+))?$", self.cv_nitn)
            if all_str:
                # Extract those parameters (0 shift if not specified)
                mod = int(all_str[0][0])
                shift = 0 if not all_str[0][1] else int(all_str[0][1])
                # Create list according to those numbers
                shifted_cv_nitns = np.where(np.arange(self.nitn) % mod == 0)[0] + shift
                self.cv_nitn = list(shifted_cv_nitns[shifted_cv_nitns < self.nitn])
            elif self.cv_nitn == "all":
                self.cv_nitn = list(range(1, self.nitn))

        # Iteration 0 is no iteration at all, so remove it
        if 0 in self.cv_nitn:
            self.cv_nitn.remove(0)

        self.num_cvs = len(self.cv_nitn)
        self.cv_means = cv_means
        # A number implies a constant mean value
        if isinstance(self.cv_means, Number):
            self.cv_means = self.num_cvs * [cv_means]

        self.rng = RandomState() if rng is None else rng

    @timing(active=TIMING)
    def create_maps(self, map_neval: Optional[int] = None) -> None:
        """
        Creates the maps corresponding to the adapted function, f, and the
        control variates, g_i.

        Parameters:
        map_neval (default None) - The number of evaluations per iteration as
            the maps are being created. Defaults to `self.neval`.
        """
        self.map_neval = self.neval if map_neval is None else map_neval
        integrator = Integrator(self.bounds)
        self._cv_maps = []
        self.tot_neval = 0

        # Do this if there actually are CVs, otherwise you don't need to
        if self.cv_nitn:
            result = integrator(
                self.function._f, nitn=self.cv_nitn[0], neval=self.map_neval
            )
            self._cv_maps.append(deepcopy(integrator.map))
            self.tot_neval += int(result.sum_neval)
            # For loop if there is more than 1 CV to save the others
            for cv_nitn_diff in np.diff(self.cv_nitn):
                result = integrator(
                    self.function._f, nitn=cv_nitn_diff, neval=self.map_neval
                )
                self._cv_maps.append(deepcopy(integrator.map))
                self.tot_neval += int(result.sum_neval)

            # And save the final map as the IS map
            result = integrator(
                self.function._f,
                nitn=self.nitn - self.cv_nitn[-1],
                neval=self.map_neval,
            )
            self._is_map = deepcopy(integrator.map)
            self.tot_neval += int(result.sum_neval)
        else:
            # Only have an IS map if there are no CVs
            result = integrator(self.function._f, nitn=self.nitn, neval=self.map_neval)
            self._is_map = deepcopy(integrator.map)
            self.tot_neval += int(result.sum_neval)

    @timing(active=TIMING)
    def get_is_cv_values(self, jac_neval: Optional[int] = None) -> None:
        """
        Calculates the  adapted function and the control variates from their maps.

        Parameters:
        jac_neval (default None) - The number of steps to split up `ys`, the unit
            hypercube, into. Defaults to `self.tot_neval`, the total number of
            iterations used when adapting the map.
        """
        self.jac_neval = self.neval * self.nitn if jac_neval is None else jac_neval

        # Uniformly distributed unit hypercube
        ys = self.rng.uniform(0, 1, (self.jac_neval, self.function.dim))
        # Find the Jacobian. If by importance sampling we transform f -> f/p, then
        # the Jacobian is 1/p
        self.xs = np.empty(ys.shape, float)
        is_jac = np.empty(ys.shape[0], float)
        self._is_map.map(ys, self.xs, is_jac)

        # The IS values
        self.weight_value = is_jac * self.function._f(self.xs)

        # Find the Jacobian(s) for the CV(s)
        self.cv_values = []
        for cv_map in self._cv_maps:
            # Use inverse map for control variate to find CV Jacobian
            t_inv = np.empty(self.xs.shape, float)
            cv_jac = np.empty(self.xs.shape[0], float)
            cv_map.invmap(self.xs, t_inv, cv_jac)

            self.cv_values.append(is_jac / cv_jac)

    @timing(active=TIMING)
    def get_weight_prime(self, constant: bool = True) -> None:
        """
        Calculates the final CV function by finding the optimal coefficients
        for the control variates.

        Parameters:
        constant - If `True`, then the N optimal coefficients (for N CVs) will be
            constant for every `self.jac_neval` values in each function. This is much
            faster. If `False`, then calculate a different coefficient for each element.
            Explained in more detail in the `self._find_coefficients` docstring.
        """
        self._find_coefficients(constant=constant)
        self.weight_prime = self.weight_value + sum(
            [
                self.cs[ind] * (self.cv_values[ind] - self.cv_means[ind])
                for ind in range(self.num_cvs)
            ]
        )

    def _find_coefficients(self, constant: bool) -> None:
        """
        Finds the optimized values for the CV coefficients to minimize the variance via
        a matrix. Our equation to solve is of the form A=Bc where A and c are arrays and
        B is a matrix. We solve for c.

        Parameters:
        constant (default True) - Since the ith value of a control variate can be
            slightly correlated to its coefficient, we can remove said value to
            calculate the variance/covariance (and therefore the coefficient). Thus we
            can have different values (albeit similar ones) for each value. This is so
            if `constant=True`. If `constant=False`, only do single coefficient per CV.
        """
        # Create (num_cv, num_cv) matrix
        if constant:
            Bs = np.zeros((self.num_cvs, self.num_cvs))
        else:
            Bs = np.zeros((self.num_cvs, self.num_cvs, self.jac_neval))

        # Populate the B matrix
        for i, j in product(range(self.num_cvs), repeat=2):
            Bs[i, j] = self._cov(
                self.cv_values[i], self.cv_values[j], constant=constant
            )
        As = np.array(
            [
                -self._cov(self.weight_value, cv_value, constant=constant)
                for cv_value in self.cv_values
            ]
        )

        # Solve the system of equations for each index
        if constant:
            cs = np.linalg.solve(Bs, As)
        else:
            cs = np.array(
                [
                    np.linalg.solve(Bs[:, :, ind], As[:, ind])
                    for ind in range(self.jac_neval)
                ]
            )
        self.cs = cs.T

    def _cov(
        self,
        f1: NDArray[Shape["'*'"], Float],
        f2: NDArray[Shape["'*'"], Float],
        constant: bool,
    ) -> Union[float, NDArray[Shape["'*'"], Float]]:
        """
        Calculates the covariance between `f1` and `f2`.

        If `constant` is `False`: for each element of the array returned, the covariance
        is calculated with said element of `f1` and `f2` removed. This removes the
        potential correlation between the CV and its coefficient so that when
        calculating the expectation value of f', we can act linearly with
        E[c(g-E(g))]=0.

        If `constant` is `True`: return a single covariance.
        """
        if constant:
            return np.cov(f1, f2)[0, 1]

        prod = f1 * f2
        # For each index, calculate the covariance without the value
        # for said index to remove that bias otherwise E[x_prime] =/= E[x]
        cov = (np.sum(prod) - prod) / (self.jac_neval - 1) - (np.sum(f1) - f1) * (
            np.sum(f2) - f2
        ) / (self.jac_neval - 1) ** 2
        return cov

    def integrate(
        self,
        map_neval: Optional[int] = None,
        jac_neval: Optional[int] = None,
        constant: bool = False,
    ) -> None:
        """
        Runs the necessary functions to integrate the function in the order:
            1) self.create_maps
            2) self.get_is_cv_values
            3) self.get_weight_primes
        Check out the docstrings of these functions for more info on them.

        Parameters:
        map_neval (default None) - From self.create_maps docstring: The number of
            evaluations per iteration as the maps are being created. Defaults to
            `self.neval`.
        jac_neval (default None) - From self.get_is_cv_values docstring: The number of
            steps to split up `ys`, the unit hypercube, into. Defaults to
            `self.tot_neval`, the total number of iterations used when adapting the map.
        constant (default False) - From self.get_weight_primes: Since the ith value of a
            control variate can be slightly correlated to its coefficient, we can remove
            said value to calculate the variance/covariance (and therefore the
            coefficient). Thus we can have different values (albeit similar ones) for
            each value. This is so if `constant=True`. If `constant=False`, just do a
            single coefficient per CV.
        """
        self.create_maps(map_neval=map_neval)
        self.get_is_cv_values(jac_neval=jac_neval)
        if self.cv_values:
            # only run if we are using control variates
            self.get_weight_prime(constant=constant)

    @property
    @check_value
    def stdev(self) -> float:
        """Standard deviation of CV function"""
        return np.std(self.weight_prime) / np.sqrt(self.jac_neval)

    @property
    @check_value
    def w_stdev(self) -> float:
        """Standard deviation of IS function"""
        return np.std(self.weight_value) / np.sqrt(self.jac_neval)

    @property
    @check_value
    def var(self) -> float:
        """Variance of CV function"""
        return self.stdev**2

    @property
    @check_value
    def w_var(self) -> float:
        """Variance of IS function"""
        return self.w_stdev**2

    @property
    @check_value
    def mean(self) -> float:
        """Mean of CV function"""
        return np.mean(self.weight_prime)

    @property
    @check_value
    def w_mean(self) -> float:
        """Mean of IS function"""
        return np.mean(self.weight_value)

    @property
    @check_value
    def vpr(self) -> float:
        """
        Variance percentage reduction, i.e. by what percent was the variance
        reduced due to the CVs.
        """
        return 1 - self.var / self.w_var

    def compare(
        self, rounding: int = 3, cutoff: Union[int, tuple[int, int]] = 3
    ) -> None:
        """
        Print out mean, variance and standard deviation for without and with the CVs.

        Parameters:
        rounding - How many digits to round the numbers to.
        cutoff - The power (base 10) at which to switch from floating point to
            scientific notation. That is if `cutoff=3`, then any value that is less than
            10^-3 or greater than 10^-3 will be printed in scientific notation. Can also
            be given as a tuple representing the lower and upper bound separately. So if
            `cutoff=(-3, 5)`, then the value would be printed in scientific notation if
            it is less than 10^-3 or greater than 10^5.
        """
        if isinstance(cutoff, int):
            neg_co, pos_co = 10 ** (-cutoff), 10**cutoff
        else:
            neg_co, pos_co = 10 ** (-cutoff[0]), 10 ** cutoff[1]

        def vtype(val):
            """Returns 'e' for scientific method or 'f' for float"""
            if np.isnan(val) or (val > neg_co and val < pos_co):
                return "f"
            return "e"

        w_mean = f"{self.w_mean:.{rounding}{vtype(self.w_mean)}}"
        w_var = f"{self.w_var:.{rounding}{vtype(self.w_var)}}"
        w_stdev = f"{self.w_stdev:.{rounding}{vtype(self.w_stdev)}}"
        mean = f"{self.mean:.{rounding}{vtype(self.mean)}}"
        var = f"{self.var:.{rounding}{vtype(self.var)}}"
        stdev = f"{self.stdev:.{rounding}{vtype(self.stdev)}}"
        vpr = f"{100 * self.vpr:.{rounding}{vtype(100 * self.vpr)}}%"

        plural = "s" if len(self.cv_nitn) > 1 else ""
        titles = f"No CV{plural}", f"With CV{plural}"
        print(f"{9*' '}|{titles[0]:^{8 + rounding}}|{titles[1]:^{8 + rounding}}")
        print(f"---------+{'-'*(8 + rounding)}+{'-'*(8 + rounding)}")
        print(f"Mean     |{w_mean:>{7 + rounding}} |{mean:>{7 + rounding}}")
        print(f"Variance |{w_var:>{7 + rounding}} |{var:>{7 + rounding}}")
        print(f"St Dev   |{w_stdev:>{7 + rounding}} |{stdev:>{7 + rounding}}")
        print(f"VPR      |{' ' * (7 + rounding)} |{vpr:>{7 + rounding}}")
