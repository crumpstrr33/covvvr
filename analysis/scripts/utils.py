import numpy as np
from constants import DATA_DIR


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
