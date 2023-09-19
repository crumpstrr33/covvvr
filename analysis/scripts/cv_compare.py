"""
This file uses the data created by calc_percdiff.py to get the 1 CV and 2CV that
minimizes the variance. It runs the integrator for each function a number of times and
saves that data.
"""
from datetime import datetime as dt

import numpy as np
from constants import DATA_DIR, FUNCTIONS
from my_favorite_things import format_ddict, nested_ddict, save
from utils import find_max_cvs

from covvvr import CVIntegrator


def run(func, cv_iters, iters, evts, rng_seed):
    """
    Runs the integrator and returns data.
    """
    t0 = dt.now()
    cvi = CVIntegrator(
        func, evals=evts, tot_iters=iters, cv_iters=cv_iters, rng_seed=rng_seed
    )
    cvi.integrate()
    t1 = dt.now()

    tot_time = (t1 - t0).total_seconds()

    if not cv_iters:
        return cvi.w_var, tot_time, cvi.w_mean
    return cvi.var, cvi.vrp, tot_time, cvi.mean


def run_0cv(func, iters, evts, rng_seed):
    return run(func, [], iters, evts, rng_seed)


def run_1cv(func, cv, iters, evts, rng_seed):
    return run(func, [cv], iters, evts, rng_seed)


def run_2cv(func, cvs, iters, evts, rng_seed):
    return run(func, list(cvs), iters, evts, rng_seed)


def run_allcv(func, iters, evts, rng_seed):
    return run(func, "all", iters, evts, rng_seed)


if __name__ == "__main__":
    N = 100
    evts = 5000
    iters = 50

    vrps = nested_ddict(1, list)
    variances = nested_ddict(1, list)
    times = nested_ddict(1, list)
    means = nested_ddict(1, list)
    metadata = {}

    metadata["N"] = N
    metadata["evts"] = evts
    metadata["iters"] = iters
    metadata["rng_seed"] = []
    metadata["functions"] = []
    metadata["true_values"] = []
    metadata["dim"] = []

    # Cycle through all the functions
    for func in FUNCTIONS:
        f_pre = f"{func.name}"
        print(f"Function: {f_pre} | ", end="", flush=True)
        rng_seed = np.random.randint(0, 1_000_000)
        metadata["rng_seed"].append(rng_seed)
        metadata["functions"].append(func.name)
        metadata["true_values"].append(func.true_value)
        metadata["dim"].append(func.dim)

        max_1cv, max_2cv = find_max_cvs(func.name, iters, evts, N)
        print(f"Found max CVs: {max_1cv} and {max_2cv}", flush=True)
        # Run N times to take the average
        for n in range(N):
            n_pre = f"{n + 1:>{len(str(N))}}/{N}"
            print(f"\tRunning {n_pre}...", flush=True)
            # Run for no CVs
            var, time, mean = run_0cv(func, iters, evts, rng_seed)
            variances[func.name][0].append(var)
            times[func.name][0].append(time)
            means[func.name][0].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran 0 CVs in   {time:>8.3f} seconds.",
                flush=True,
            )

            # Run for 1 CV
            var, vrp, time, mean = run_1cv(func, max_1cv, iters, evts, rng_seed)
            variances[func.name][1].append(var)
            vrps[func.name][1].append(vrp)
            times[func.name][1].append(time)
            means[func.name][1].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran 1 CV  in   {time:>8.3f} seconds.",
                flush=True,
            )

            # Run for 2 CVs
            var, vrp, time, mean = run_2cv(func, max_2cv, iters, evts, rng_seed)
            variances[func.name][2].append(var)
            vrps[func.name][2].append(vrp)
            times[func.name][2].append(time)
            means[func.name][2].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran 2 CVs in   {time:>8.3f} seconds.",
                flush=True,
            )

            # Run for all CVs
            var, vrp, time, mean = run_allcv(func, iters, evts, rng_seed)
            variances[func.name]["all"].append(var)
            vrps[func.name]["all"].append(vrp)
            times[func.name]["all"].append(time)
            means[func.name]["all"].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran all CVs in {time:>8.3f} seconds.\n",
                flush=True,
            )

    variances = format_ddict(variances)
    vrps = format_ddict(vrps)
    times = format_ddict(times)
    means = format_ddict(means)

    print()
    sname = f"compare_{iters}x{evts}_avg{N}"
    save(
        name=sname,
        savedir=DATA_DIR,
        absolute=True,
        stype="pkl",
        variances=variances,
        vrps=vrps,
        times=times,
        means=means,
        metadata=metadata,
    )
