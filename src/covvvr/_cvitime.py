"""
File that holds the a version of the CVIntegrator class for granule timing. Don't take
this too seriously nor consider it even good coding practice. It was the simpliest way
to do thi :^)
"""
from copy import deepcopy
from datetime import datetime as dt
from itertools import product
from typing import Optional, Sequence, Union

import numpy as np
from nptyping import Float, NDArray, Shape
from numpy.random import RandomState
from vegas import Integrator

from ._wrappers import check_attrs, timing
from .cvintegrator import CVIntegrator
from .functions import Function


class CVITime(CVIntegrator):
    """
    This class is exactly the same as the CVIntegrator class. The only difference is
    that is has time measurements everywhere which are printed out. This is for
    benchmarks on the bottlenecks of the algorithm.

    The methods defined below are the same as in CVIntegrator except for the added
    lines for collecting the time data.
    """

    TIMING = True

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
        super().__init__(
            function=function,
            evals=evals,
            tot_iters=tot_iters,
            bounds=bounds,
            cv_iters=cv_iters,
            cv_means=cv_means,
            rng_seed=rng_seed,
            memory=memory,
        )

    @timing
    def create_maps(self, map_neval: Optional[int] = None, auto1_neval=None) -> None:
        timing_info = []
        self.map_neval = self.neval if map_neval is None else map_neval
        integrator = Integrator(self.bounds)
        self._cv_maps = []
        self.tot_neval = 0

        # Do this if there actually are CVs, otherwise you don't need to
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
                self.get_weight_prime(constant=True)
                vrps.append(self.vrp)
            # Find which index/map gives the maximum VRP and use that
            max_vrp_ind = np.argmax(vrps)
            print(vrps)
            self.cv_nitn = [max_vrp_ind + 1]
            self._cv_maps = [self._tmp_cv_maps[max_vrp_ind]]
        elif self.cv_nitn:
            dt0 = dt.now()
            # Run integrator for number of iterations until we reach first CV
            result = integrator(
                self.function._f, nitn=self.cv_nitn[0], neval=self.map_neval
            )
            dt1 = dt.now()
            timing_info.append(["First iteration", (dt1 - dt0).total_seconds()])
            # Save map for CV
            self._cv_maps.append(deepcopy(integrator.map))
            dt2 = dt.now()
            timing_info.append(["First deepcopy", (dt2 - dt1).total_seconds()])
            self.tot_neval += int(result.sum_neval)
            # For loop if there is more than 1 CV to save the others
            for cv_nitn_diff in np.diff(self.cv_nitn):
                # Same process as before
                result = integrator(
                    self.function._f, nitn=cv_nitn_diff, neval=self.map_neval
                )
                self._cv_maps.append(deepcopy(integrator.map))
                self.tot_neval += int(result.sum_neval)
            dt3 = dt.now()
            timing_info.append(
                ["Other iterations/deepcopies", (dt3 - dt2).total_seconds()]
            )

            # And save the final map as the IS map
            result = integrator(
                self.function._f,
                nitn=self.nitn - self.cv_nitn[-1],
                neval=self.map_neval,
            )
            dt4 = dt.now()
            timing_info.append(["Last iteration", (dt4 - dt3).total_seconds()])
            self._is_map = deepcopy(integrator.map)
            self.tot_neval += int(result.sum_neval)
            dt5 = dt.now()
            timing_info.append(["Last deepcopy", (dt5 - dt4).total_seconds()])
        else:
            # Only have an IS map if there are no CVs
            result = integrator(self.function._f, nitn=self.nitn, neval=self.map_neval)
            self._is_map = deepcopy(integrator.map)
            self.tot_neval += int(result.sum_neval)

        return timing_info

    @timing
    def get_is_cv_values(self, jac_neval: Optional[int] = None) -> None:
        timing_info = []
        self.jac_neval = self.tot_neval if jac_neval is None else jac_neval
        rng = RandomState(seed=self.rng_seed)

        # Uniformly distributed unit hypercube
        dt0 = dt.now()
        ys = rng.uniform(0, 1, (self.jac_neval, self.function.dim))
        dt1 = dt.now()
        timing_info.append(["Make hypercube", (dt1 - dt0).total_seconds()])
        # Find the Jacobian. If by importance sampling we transform f -> f/p, then
        # the Jacobian is 1/p
        xs = np.empty(ys.shape, float)
        is_jac = np.empty(ys.shape[0], float)
        self._is_map.map(ys, xs, is_jac)
        dt2 = dt.now()
        timing_info.append(["Find IS map", (dt2 - dt1).total_seconds()])
        # The IS values
        self.weight_value = is_jac * self.function._f(xs)
        dt3 = dt.now()
        timing_info.append(["Find vegas values", (dt3 - dt2).total_seconds()])

        # Find the Jacobian(s) for the CV(s)
        self.cv_values, self.cv_jacs = [], []
        for cv_map in self._cv_maps:
            # Use inverse map for control variate to find CV Jacobian
            t_inv = np.empty(xs.shape, float)
            cv_jac = np.empty(xs.shape[0], float)
            cv_map.invmap(xs, t_inv, cv_jac)

            self.cv_values.append(is_jac / cv_jac)
            self.cv_jacs.append(cv_jac)

        dt4 = dt.now()
        timing_info.append(["Find CV values", (dt4 - dt3).total_seconds()])

        if self.memory == "max":
            self.xs = xs
            self.is_jac = is_jac

        return timing_info

    @timing
    def get_weight_prime(self) -> None:
        timing_info = []
        dt0 = dt.now()
        coeff_timing_info = self._find_coefficients()
        dt1 = dt.now()
        timing_info.append(["Find coefficient(s)", (dt1 - dt0).total_seconds()])
        timing_info += coeff_timing_info
        self.weight_prime = self.weight_value + sum(
            [
                self.cs[ind] * (self.cv_values[ind] - self.cv_means[ind])
                for ind in range(self.num_cvs)
            ]
        )
        dt2 = dt.now()
        timing_info.append(["Calculate final values", (dt2 - dt1).total_seconds()])
        return timing_info

    def _find_coefficients(self) -> None:
        timing_info = []

        # Populate the B matrix
        dt0 = dt.now()
        Bs = np.cov(self.cv_values)
        dt1 = dt.now()
        timing_info.append(["Build B", (dt1 - dt0).total_seconds(), 2])
        As = np.array([-np.cov(self.weight_value, cv)[0, 1] for cv in self.cv_values])
        dt2 = dt.now()
        timing_info.append(["Build A", (dt2 - dt1).total_seconds(), 2])

        # Solve the system of equations
        cs = np.linalg.solve(Bs, As)
        self.cs = cs.T
        dt3 = dt.now()
        timing_info.append(["Solve inverse", (dt3 - dt2).total_seconds(), 2])
        return timing_info

    @staticmethod
    def _time_print(title, time, ind_level=1):
        spaces = 5 + 2 * (ind_level - 1)
        print(f"{' ' * spaces}{title} {'-' * (50 - len(title) - spaces)} {time:.3f}s")

    def _times_print(self, timing_info):
        for ti in timing_info:
            self._time_print(*ti)

    @timing
    def integrate(
        self,
        map_neval: Optional[int] = None,
        jac_neval: Optional[int] = None,
        auto1_neval: Optional[int] = None,
    ) -> None:
        self._times_print(
            self.create_maps(map_neval=map_neval, auto1_neval=auto1_neval)
        )
        self._times_print(self.get_is_cv_values(jac_neval=jac_neval))
        if self.cv_values:
            # only run if we are using control variates
            self._times_print(self.get_weight_prime())
        self.garbage_collect()

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
