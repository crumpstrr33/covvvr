"""
This file uses the data created by calc_percdiff.py to get the 1 CV and 2CV that
minimizes the variance. It runs the integrator for each function a number of times and
saves that data.
"""
from datetime import datetime as dt

import numpy as np
from constants import DATA_DIR, FUNCTIONS
from covvvr import CVIntegrator
from my_favorite_things import format_ddict, nested_ddict, save


def find_max_cvs(name, iters, evts, N):
    """
    Finds which CVs give the maximum variance reduction for 1CV and 2CVs from the
    2CV data. The file is found by the given arguments.

    Parameters:
    name - Name of the function as specified by the Function class, i.e. Function.name
    iters - Number of iterations
    evts - Max number of events per iteration
    N - Number of runs averaged over
    """

    fpath = DATA_DIR / f"2cv_{name.lower().replace(' ', '_')}_{iters}x{evts}_avg{N}.npz"
    perc_diff = np.load(fpath)["perc_diff"]
    perc_diff = np.mean(perc_diff, axis=0)

    # Must shift by 1 bc index notation starts at 0
    max_1cv = np.argmax(np.diag(perc_diff)) + 1
    max_2cv = np.where(perc_diff == np.max(perc_diff))[0] + 1

    return max_1cv, max_2cv


def run(func, cv_iters, iters, evts, rng):
    """
    Runs the integrator and returns data.
    """
    t0 = dt.now()
    cvi = CVIntegrator(func, evals=evts, tot_iters=iters, cv_iters=cv_iters, rng=rng)
    cvi.integrate()
    t1 = dt.now()

    tot_time = (t1 - t0).total_seconds()

    if not cv_iters:
        return cvi.w_var, tot_time, cvi.w_mean
    return cvi.var, cvi.vrp, tot_time, cvi.mean


def run_0cv(func, iters, evts, rng):
    return run(func, [], iters, evts, rng)


def run_1cv(func, cv, iters, evts, rng):
    return run(func, [cv], iters, evts, rng)


def run_2cv(func, cvs, iters, evts, rng):
    return run(func, list(cvs), iters, evts, rng)


def run_allcv(func, iters, evts, rng):
    return run(func, "all", iters, evts, rng)


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
    metadata["functions"] = []
    metadata["true_values"] = []
    metadata["dims"] = []

    # Cycle through all the functions
    for func in FUNCTIONS:
        f_pre = f"{func.name}"
        print(f"Function: {f_pre} | ", end="", flush=True)
        seed = np.random.randint(0, 1_000_000)
        rng = np.random.RandomState(seed)
        metadata["seed"] = seed
        metadata["functions"].append(func.name)
        metadata["true_values"].append(func.true_value)
        metadata["dims"].append(func.dims)

        max_1cv, max_2cv = find_max_cvs(func.name, iters, evts, N)
        print(f"Found max CVs: {max_1cv} and {max_2cv}", flush=True)
        # Run N times to take the average
        for n in range(N):
            n_pre = f"{n + 1:>{len(str(N))}}/{N}"
            print(f"\tRunning {n_pre}...", flush=True)
            # Run for no CVs
            var, time, mean = run_0cv(func, iters, evts, rng)
            variances[func.name][0].append(var)
            times[func.name][0].append(time)
            means[func.name][0].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran 0 CVs in   {time:>8.3f} seconds.",
                flush=True,
            )

            # Run for 1 CV
            var, vrp, time, mean = run_1cv(func, max_1cv, iters, evts, rng)
            variances[func.name][1].append(var)
            vrps[func.name][1].append(vrp)
            times[func.name][1].append(time)
            means[func.name][1].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran 1 CV  in   {time:>8.3f} seconds.",
                flush=True,
            )

            # Run for 2 CVs
            var, vrp, time, mean = run_2cv(func, max_2cv, iters, evts, rng)
            variances[func.name][2].append(var)
            vrps[func.name][2].append(vrp)
            times[func.name][2].append(time)
            means[func.name][2].append(mean)
            print(
                f"\t\t{f_pre} [{n_pre}] -- Ran 2 CVs in   {time:>8.3f} seconds.",
                flush=True,
            )

            # Run for all CVs
            var, vrp, time, mean = run_allcv(func, iters, evts, rng)
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
        metadata=metadata,
    )
