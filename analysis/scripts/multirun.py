"""
Runs each function N times using the best single CV.
"""
from datetime import datetime as dt

import numpy as np
from constants import DATA_DIR, FUNCTIONS
from my_favorite_things import format_ddict, nested_ddict, save
from utils import find_max_cvs

from covvvr import CVIntegrator


def run_n_times(N, func, cv_iters, iters, auto1_neval, evts, rng_seed):
    """For given parameters, run the integrator `N` times."""
    cv_means, vegas_means, max_cvs = [], [], []
    for n in range(N):
        print(f"\t Running {n + 1:>{len(str(N))}}/{N} ", end="", flush=True)
        t0 = dt.now()
        cvi = CVIntegrator(
            func, evals=evts, tot_iters=iters, cv_iters=cv_iters, rng_seed=rng_seed
        )
        cvi.integrate(auto1_neval=auto1_neval)
        t1 = dt.now()
        print(f"in {(t1 - t0).total_seconds():.3f} seconds", flush=True)
        cv_means.append(cvi.mean)
        vegas_means.append(cvi.w_mean)
        max_cvs.append(cvi.cv_nitn)

    return cv_means, vegas_means, max_cvs


if __name__ == "__main__":
    N = 1000
    auto1_neval = 1000
    evts = 5000
    iters = 50

    means = nested_ddict(1, list)
    metadata = {}

    metadata["N"] = N
    metadata["evts"] = evts
    metadata["iters"] = iters
    metadata["auto1_neval"] = auto1_neval
    metadata["seed"] = []
    metadata["functions"] = []
    metadata["true_values"] = []
    metadata["dims"] = []
    metadata["max_cv"] = []

    # The commented code uses the best CVs found by cv_compare
    # As is, it uses the 'auto1' choice parameterized by `auto1_neval`
    for func in FUNCTIONS:
        # Use previously found max CV
        max_cv, _ = find_max_cvs(func.name, iters, evts, N=100)
        line = f"Function: {func.name} "  # | Max CV: {max_cv} "
        print(line, end="", flush=True)
        print(f"{'-' * (50 - len(line))} [events: {evts}, iterations: {iters}]")
        seed = np.random.randint(0, 1_000_000)

        metadata["seed"].append(seed)
        metadata["functions"].append(func.name)
        metadata["true_values"].append(func.true_value)
        metadata["dims"].append(func.dim)
        # metadata["max_cv"].append(max_cv)

        # Change `"auto1"` to `int(max_cv)` to change it
        cv_means, vegas_means, max_cvs = run_n_times(
            # N=N, func=func, cv_iters=int(max_cv), iters=iters, evts=evts, rng=rng
            N=N,
            func=func,
            cv_iters="auto1",
            auto1_neval=auto1_neval,
            iters=iters,
            evts=evts,
            rng_seed=seed,
        )
        means[func.name]["cv"] = cv_means
        means[func.name]["vegas"] = vegas_means
        means[func.name]["max_cvs"] = max_cvs

    means = format_ddict(means)
    sname = f"means_{iters}x{evts}_avg{N}_a1n{auto1_neval}"
    save(
        name=sname,
        savedir=DATA_DIR,
        absolute=True,
        stype="pkl",
        means=means,
        metadata=metadata,
    )
