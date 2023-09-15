"""
File that holds the CVIntegrator class
"""

from copy import deepcopy
from itertools import product
from numbers import Number
from re import findall
from sys import getsizeof
from typing import Optional, Sequence, Union

import numpy as np
from nptyping import Float, NDArray, Shape
from numpy.random import RandomState
from vegas import Integrator

from ._types import _ftype
from ._wrappers import check_attrs, timing
from .functions import Function, make_func


def classic_integrate(
    function: _ftype,
    evals: int,
    tot_iters: int,
    bounds: Union[Sequence[tuple[float, float]], tuple[float, float]],
    cv_iters: Optional[Union[list[int], int, str]] = None,
    cv_means: Union[float, Sequence[float]] = 1,
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
    )
    # And integrate
    cvi.integrate()
    return cvi


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

    # Should you print out the time it takes for the main functions to run?
    TIMING = False
    # Memory threshold for when `memory="tiny"`
    TINY_THRESHOLD = 100
    # Integer code for the 'auto1' option of cv_iters
    AUTO1 = [1e15]

    def __init__(
        self,
        function: Function,
        evals: int,
        tot_iters: int,
        bounds: Optional[Sequence[tuple[float, float]]] = None,
        cv_iters: Optional[Union[list[int], int, str]] = None,
        cv_means: Union[float, Sequence[float]] = 1,
        rng_seed: Optional[int] = None,
        memory: str = "medium",
    ):
        """
        Takes in a Function class object from functions.py. One can make their own using
        the make_func function found in that file.

        Parameters:
        function - Function class with f to integrate.
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
                - 'auto1': Will automatically assign a single CV by testing each
                    possibility with a small number of events
        cv_means (default 1) - The value of E[g_i] but by the scheme laid out in
            `get_is_cv_values` to obtain the control variate, E[g_i] should be
            approximately one.
        rng_seed (default None) - The seed to use for the numpy RandomState class.
            Useful for testing with the same numbers. Note: vegas's Integrator does
            not have a seed argument and so self.create_maps must be run separately
            than with self.get_is_cv_values and self.get_weight_prime. If no seed is
            passed, a random one will be created.
        memory (default 'medium') - Either 'low', 'medium', 'large' or `max. Determines
            what is saved. If `max`, save everything. If `large, don't save self.xs
            and self.is_jac. If 'medium', additionally don't save self.weight_value and
            self.weight_prime. If 'tiny', remove everything below the threshold
            TINY_TRESHOLD.
        """
        # Initialize private attributes for the properties
        self._init_results()
        # Ordering of timing if activated
        if self.TIMING:
            self.timing_count = 1

        self.function = function
        self.bounds = self.function.dim * [[0, 1]] if bounds is None else bounds
        self.neval = evals
        self.nitn = tot_iters

        self.cv_nitn = cv_iters
        # Create empty list if not specified, i.e. no control variates
        if self.cv_nitn is None:
            self.cv_nitn = []
        # If cv_iters is a number, put it into a list
        if isinstance(self.cv_nitn, (int, np.integer)):
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
            elif self.cv_nitn == "auto1":
                self.cv_nitn = self.AUTO1

        # Iteration 0 is no iteration at all, so remove it
        if 0 in self.cv_nitn:
            self.cv_nitn.remove(0)

        self.num_cvs = len(self.cv_nitn)
        self.cv_means = cv_means
        # A number implies a constant mean value
        if isinstance(self.cv_means, Number):
            self.cv_means = self.num_cvs * [cv_means]

        self.rng_seed = np.random.randint(0, 1e9) if rng_seed is None else rng_seed
        self.memory = memory

    def _init_results(self):
        """Initializes the private attributes for the listed properties."""
        for name, obj in self.__class__.__dict__.items():
            if isinstance(obj, property):
                self.__setattr__(f"_{name}", np.nan)

    @timing
    def create_maps(
        self, map_neval: Optional[int] = None, auto1_neval: Optional[int] = None
    ) -> None:
        """
        Creates the maps corresponding to the adapted function, f, and the
        control variates, g_i.

        Parameters:
        map_neval (default None) - The number of evaluations per iteration as
            the maps are being created. Defaults to `self.neval`.
        auto1_neval (defaut None) - Only used if `cv_iters` was set to 'auto1'.
            The number of iterations to use for the testing of each CV. Defaults
            to the value of `self.map_neval`.
        """
        self.map_neval = self.neval if map_neval is None else map_neval
        integrator = Integrator(self.bounds)
        self._cv_maps = []
        self.tot_neval = 0

        # Will adapt maps as usual, but then test each map as a single CV with
        # `auto1_neval` events to get an estimate of which CV is best to use
        if self.cv_nitn == self.AUTO1:
            auto1_neval = self.map_neval if auto1_neval is None else auto1_neval
            # Run for a smaller number of points for every possibility
            self._tmp_cv_maps = []
            # Copy every map iteration
            for nitn in range(self.nitn - 1):
                result = integrator(self.function._f, nitn=1, neval=self.map_neval)
                self._tmp_cv_maps.append(deepcopy(integrator.map))
                self.tot_neval += int(result.sum_neval)

            result = integrator(self.function._f, nitn=1, neval=self.map_neval)
            self._is_map = deepcopy(integrator.map)
            self.tot_neval += int(result.sum_neval)

            # Run through each map and see what the VRP is
            vrps = []
            for ind, cv_map in enumerate(self._tmp_cv_maps):
                self._cv_maps = [cv_map]

                self.get_is_cv_values(jac_neval=auto1_neval)
                self.get_weight_prime()
                vrps.append(self.vrp)
            # Find which index/map gives the maximum VRP and use that
            max_vrp_ind = np.argmax(vrps)
            self.cv_nitn = [max_vrp_ind + 1]
            self._cv_maps = [self._tmp_cv_maps[max_vrp_ind]]
        # Will adapt maps until a CV is reached and save that CV and keep going
        elif self.cv_nitn:
            # Run integrator for number of iterations until we reach first CV
            result = integrator(
                self.function._f, nitn=self.cv_nitn[0], neval=self.map_neval
            )
            # Save map for CV
            self._cv_maps.append(deepcopy(integrator.map))
            self.tot_neval += int(result.sum_neval)
            # For loop if there is more than 1 CV to save the others
            for cv_nitn_diff in np.diff(self.cv_nitn):
                # Same process as before
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
        # This option is for no CVs, so don't save any for the CV
        else:
            # Only have an IS map if there are no CVs
            result = integrator(self.function._f, nitn=self.nitn, neval=self.map_neval)
            self._is_map = deepcopy(integrator.map)
            self.tot_neval += int(result.sum_neval)

    @timing
    def get_is_cv_values(self, jac_neval: Optional[int] = None) -> None:
        """
        Calculates the adapted function and the control variates from their maps.

        Parameters:
        jac_neval (default None) - The number of steps to split up `ys`, the unit
            hypercube, into. Defaults to `self.tot_neval`, the total number of
            iterations used when adapting the map.
        """
        self.jac_neval = self.tot_neval if jac_neval is None else jac_neval
        rng = RandomState(seed=self.rng_seed)

        # Uniformly distributed unit hypercube
        ys = rng.uniform(0, 1, (self.jac_neval, self.function.dim))
        # Find the Jacobian. If by importance sampling we transform f -> f/p, then
        # the Jacobian is 1/p
        xs = np.empty(ys.shape, float)
        is_jac = np.empty(ys.shape[0], float)
        self._is_map.map(ys, xs, is_jac)
        # The IS values
        self.weight_value = is_jac * self.function._f(xs)

        # Find the Jacobian(s) for the CV(s)
        self.cv_values, self.cv_jacs = [], []
        for cv_map in self._cv_maps:
            # Use inverse map for control variate to find CV Jacobian
            t_inv = np.empty(xs.shape, float)
            cv_jac = np.empty(xs.shape[0], float)
            cv_map.invmap(xs, t_inv, cv_jac)

            self.cv_values.append(is_jac / cv_jac)
            self.cv_jacs.append(cv_jac)

        if self.memory == "max":
            self.xs = xs
            self.is_jac = is_jac

    @timing
    def get_weight_prime(self) -> None:
        """
        Calculates the final CV function by finding the optimal coefficients
        for the control variates.
        """
        self._find_coefficients()
        self.weight_prime = self.weight_value + sum(
            [
                self.cs[ind] * (self.cv_values[ind] - self.cv_means[ind])
                for ind in range(self.num_cvs)
            ]
        )

    def _find_coefficients(self) -> None:
        """
        Finds the optimized values for the CV coefficients to minimize the variance via
        a matrix. Our equation to solve is of the form A=Bc where A and c are arrays and
        B is a matrix. We solve for c.

        The ith value of a control variate can be correlated to its coefficient, so
        removing said value when calculating the covariance would remove this
        correlation while still maintaining a good approximation of it. In effect,
        this would look like a covariance for each ith point. But this subtlety does
        not have a noticeable effect on the result but slows down the algorithm by
        multiple times so it isn't done.
        """
        # Populate the B matrix
        Bs = np.cov(self.cv_values)
        As = np.array([-np.cov(self.weight_value, cv)[0, 1] for cv in self.cv_values])

        # np.cov returns 0D array if there's only one element, so turn it into a matrix
        if self.num_cvs == 1:
            Bs = Bs.reshape(1, 1)
        # Solve the system of equations
        cs = np.linalg.solve(Bs, As)
        self.cs = cs.T

    @timing
    def integrate(
        self,
        map_neval: Optional[int] = None,
        jac_neval: Optional[int] = None,
        auto1_neval: Optional[int] = None,
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
        auto1_neval (defaut None) - Only used if `cv_iters` was set to 'auto1'.
            The number of iterations to use for the testing of each CV. Defaults
            to the value of `self.map_neval`.
        """
        self.create_maps(map_neval=map_neval, auto1_neval=auto1_neval)
        self.get_is_cv_values(jac_neval=jac_neval)
        if self.cv_values:
            # only run if we are using control variates
            self.get_weight_prime()
        self.garbage_collect()

    @timing
    def garbage_collect(self, memory: Optional[str] = None) -> None:
        """
        Deletes attributes depending on choices of self.memory to clear up memory.

        Parameters:
        memory (default None) - A measure of how many attributes to delete to clear up
            memory. Can be 'max', 'large', 'medium' or 'tiny'. More info in the class
            __init__ docstring.
        """
        memory = memory or self.memory

        # Initialize properties so result is saved before deleting arrays they need
        self.stdev
        self.mean
        self.w_stdev
        self.w_mean

        # Large arrays not used in anything
        if memory in {"large", "medium", "tiny"}:
            self._delete("xs", "is_jac")
        # Large arrays but used in properties below
        if memory in {"medium", "tiny"}:
            self._delete("weight_prime", "weight_value")
        # All bigger than a certain threshold
        if memory == "tiny":
            attr_items = list(self.__dict__.items())
            for attr, attr_val in attr_items:
                if getsizeof(attr_val) > self.TINY_THRESHOLD:
                    self.__delattr__(attr)

    def _delete(self, *attrs):
        """Delete attribute if it still exists as one."""
        for attr in attrs:
            if attr in self.__dir__():
                self.__delattr__(attr)

    @property
    @check_attrs("weight_prime", "jac_neval")
    def stdev(self) -> float:
        """Standard deviation of CV function"""
        self._stdev = np.std(self.weight_prime) / np.sqrt(self.jac_neval)
        return self._stdev

    @property
    @check_attrs("weight_value", "jac_neval")
    def w_stdev(self) -> float:
        """Standard deviation of IS function"""
        self._w_stdev = np.std(self.weight_value) / np.sqrt(self.jac_neval)
        return self._w_stdev

    @property
    @check_attrs("stdev")
    def var(self) -> float:
        """Variance of CV function"""
        self._var = self.stdev**2
        return self._var

    @property
    def w_var(self) -> float:
        """Variance of IS function"""
        self._w_var = self.w_stdev**2
        return self._w_var

    @property
    @check_attrs("weight_prime")
    def mean(self) -> float:
        """Mean of CV function"""
        self._mean = np.mean(self.weight_prime)
        return self._mean

    @property
    @check_attrs("weight_value")
    def w_mean(self) -> float:
        """Mean of IS function"""
        self._w_mean = np.mean(self.weight_value)
        return self._w_mean

    @property
    def vrp(self) -> float:
        """
        Variance reduction in percentage, i.e. by what percent was the variance
        reduced due to the CVs.
        """
        self._vrp = 1 - self.var / self.w_var
        return self._vrp

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
        vrp = f"{100 * self.vrp:.{rounding}{vtype(100 * self.vrp)}}%"

        plural = "s" if len(self.cv_nitn) > 1 else ""
        titles = f"No CV{plural}", f"With CV{plural}"
        print(f"{9*' '}|{titles[0]:^{8 + rounding}}|{titles[1]:^{8 + rounding}}")
        print(f"---------+{'-'*(8 + rounding)}+{'-'*(8 + rounding)}")
        print(f"Mean     |{w_mean:>{7 + rounding}} |{mean:>{7 + rounding}}")
        print(f"Variance |{w_var:>{7 + rounding}} |{var:>{7 + rounding}}")
        print(f"St Dev   |{w_stdev:>{7 + rounding}} |{stdev:>{7 + rounding}}")
        print(f"VRP      |{' ' * (7 + rounding)} |{vrp:>{7 + rounding}}")
