"""
Calculates the variance reduction in percentage as defined by the `vrp` property
of the `CVIntegrator` class for every pair of possible CVs given a total
number of iterations. It also calculates the VRP for every single CV.
"""
from datetime import datetime as dt
from itertools import combinations
from multiprocessing.pool import Pool

import numpy as np
from constants import DATA_DIR
from covvvr import CVIntegrator
from covvvr.functions import ScalarTopLoop
from my_favorite_things import save


def time_units(time_val):
    """Convert from seconds to minutes to hours for printing"""
    unit = "seconds"
    if time_val >= 100:
        unit = "minutes"
        time_val /= 60
    elif time_val > 6000:
        unit = "hours  "
        time_val /= 3600
    return time_val, unit


class CalcVRP:
    """
    Calculate the percent reduction in variances for every possible pair of
    two control variates.

    Parameters:
    f - Function to integrate
    nitn - Number of iterations for Vegas
    neval - Max number of events per iteration for Vegas
    bounds - Integration bounds
    """

    def __init__(self, func, nitn, neval):
        self.func = func
        self.nitn = nitn
        self.N = self.nitn - 1
        self.neval = neval

    def _diag(self, cv_nitn):
        """
        Calculates values for a single CV. Used for the diagonal of the heatmap.

        Parameters:
        cv_nitn - Iteration used for CV

        Returns: (
            Iteration used for CV,
            Variance percentage reduction,
            Variance of CV result,
            Correlation coefficient IS function and CV
        )
        """
        # Create unique seed for each process
        rng = np.random.RandomState()
        cvi = CVIntegrator(
            function=self.func,
            evals=self.neval,
            tot_iters=self.nitn,
            cv_iters=cv_nitn,
            memory="large",
            rng=rng,
        )
        cvi.integrate()
        return (
            cv_nitn,
            cvi.vrp,
            cvi.var,
            np.corrcoef(cvi.weight_value, cvi.cv_values[0])[0, 1],
        )

    def _off_diag(self, cv_nitns):
        """
        Calculates variance percentage reduction for two CVs. Used for the
        off-diagonal of the heatmap.

        Parameters:
        cv_nitns - Tuple of two iterations to use for CVs

        Returns: (
            Tuple of iterations used for CVs,
            Varianec percentage reduction
        )
        """
        rng = np.random.RandomState()
        cvi = CVIntegrator(
            function=self.func,
            evals=self.neval,
            tot_iters=self.nitn,
            cv_iters=cv_nitns,
            memory="large",
            rng=rng,
        )
        cvi.integrate()
        return cv_nitns, cvi.vrp

    def calc(self):
        """
        returns (
            2D array representing the VRPs for every pair of 2 CVs,
            Array of variances for each choice of 1 CV,
            Correlation coefficient for each choice of 1 CV with IS function
        )
        """
        # Intialize array
        variances = np.zeros(self.N)
        corrcoefs = np.zeros(self.N)
        perc_diff = np.zeros((self.N, self.N))
        # Number used for printing
        tot_diags = self.N
        tot_off_diags = (self.N) * (self.N - 1) // 2

        # Create multiprocess pool to find diagonal terms and grab data
        print(
            "Number of diagonals:     " + f"{tot_diags:>{len(str(tot_off_diags))}} ",
            flush=True,
            end="",
        )
        t0 = dt.now()
        for cv_nitn, vrp, var, rho in Pool().map(self._diag, range(1, self.nitn)):
            perc_diff[cv_nitn - 1, cv_nitn - 1] = vrp
            variances[cv_nitn - 1] = var
            corrcoefs[cv_nitn - 1] = rho

        t1 = dt.now()
        tot_t, unit = time_units((t1 - t0).total_seconds())
        print(
            f"| Took {tot_t:>6.3f} {unit} | "
            + f"{tot_diags / tot_t:>8.3f} runs per {unit[:-1]}.",
            flush=True,
        )

        # Same thing but for the off diagonals
        print(
            "Number of off-diagonals: "
            + f"{tot_off_diags:>{len(str(tot_off_diags))}} ",
            flush=True,
            end="",
        )
        t0 = dt.now()
        for cv_nitns, vrp in Pool().map(
            self._off_diag, combinations(range(1, self.nitn), r=2)
        ):
            # Heatmap is symmetric
            perc_diff[cv_nitns[0] - 1, cv_nitns[1] - 1] = vrp
            perc_diff[cv_nitns[1] - 1, cv_nitns[0] - 1] = vrp
        t1 = dt.now()
        tot_t, unit = time_units((t1 - t0).total_seconds())
        print(
            f"| Took {tot_t:>6.3f} {unit} | "
            + f"{tot_off_diags / tot_t:>8.3f} runs per {unit[:-1]}.",
            flush=True,
        )

        return perc_diff, variances, corrcoefs


if __name__ == "__main__":
    # constant parameters
    N = 100
    nitn = 50
    neval = 5_000

    t0_tot = dt.now()
    # Info on function
    func = ScalarTopLoop()
    dim = func.dim
    name = func.name
    save_name = name.lower().replace(" ", "_")

    t0 = dt.now()
    perc_diffs, variancess, corrcoefss = [], [], []
    for n in range(N):
        # run it
        print(f"{name}: {n + 1:>4,}/{N:,}", flush=True)
        calcvrp = CalcVRP(func, nitn, neval)
        perc_diff, variances, corrcoefs = calcvrp.calc()
        perc_diffs.append(perc_diff)
        variancess.append(variances)
        corrcoefss.append(corrcoefs)
        print(flush=True)
    t1 = dt.now()
    print(
        f"Total time for {name}: " + f"{(t1 - t0).total_seconds():,.3f} seconds",
        flush=True,
    )

    # Save to data directory
    save(
        name=f"2cv_{save_name}_{nitn}x{neval}_avg{N}",
        savedir=DATA_DIR,
        absolute=True,
        # dryrun=True,
        perc_diff=perc_diffs,
        variances=variancess,
        corrcoefs=corrcoefss,
    )
