"""
Wrappers and functions used in different parts of the code.
"""
import pickle
from datetime import datetime as dt
from pathlib import Path

import numpy as np

from ._exceptions import ParameterBoundError


def timing(active: bool = True):
    """Decorator for timing with option to turn it off"""

    def decorator(f):
        def wrapper(*args, **kwargs):
            if active:
                dt0 = dt.now()
                output = f(*args, **kwargs)
                dt1 = dt.now()
                tot_time = (dt1 - dt0).total_seconds()
                print(f"{f.__name__}: {tot_time:>{30 - len(f.__name__)}.3f}s")
            else:
                output = f(*args, **kwargs)

            return output

        return wrapper

    return decorator


def check_value(f):
    """
    Decorator to return nan if there is no value otherwise. Used for the properties
    of CVIntegrator being passed as numbers, e.g. in self.compare.
    """

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AttributeError:
            return np.nan

    return wrapper


def save(
    name: str,
    savedir: str = "",
    savepath: str = "",
    stype: str = "npz",
    absolute: bool = False,
    parents: int = 0,
    overwrite: bool = False,
    dryrun: bool = False,
    **files: ...,
) -> None:
    """
    Saves data to a file. If saving numpy arrays, use `stype="npz"`, to save as a .npz
    file. For other objects such as dicts, use `stype="pkl"` to pickle it.

    Parameters:
    name - String representing name of file.
    savepath (default "") - Abosolute or relative path to save file in depending on
        value of `absolute`.
    savedir (default "") - Directory to save in. Can be a path. This is added on top of
        `savepath` and after `parents` is applied.  So `parents` moves up the tree and
        `savedir` can move down a different branch.
    stype (default "npz") - File type, can be either "npz" for saving numpy arrays or
        "pkl" for saving anything as a pickle file.
    absolute (default False) - If True, we start in the directory `savepath`, move up it
        `parents` number of times, then append `savedir` to it. If False, we start in
        the directory $CWD/`savepath` and do the same thing.
    parents (default 0) - Which parent directory to save in. If 0, saves in same
        directory as this file. If 1, saves in parent directory. If 2, saves in
        grandparent directory. And so on.
    overwrite (default False) - If a file with the same path and file name exists,
        overwrite it if `overwrite=True`. Otherwise, append `_1` to the end of the file.
        If that is already taken, then append `_2` instead. And so on, until a unique
        number is found.
    dryrun (default False) - If True, will not save anything but only print out where
        the save will be to.
    files - Kwargs for the python objects to save.
    """
    if not absolute:
        # Relative path
        path = Path.cwd() / savepath
    else:
        # Absolute path
        path = Path() / savepath

    # Create path for where to save data
    if parents > 0:
        try:
            path = path.parents[parents - 1]
        except IndexError as error:
            raise f"There is no grand^{parents}parent folder. for {path}." from error
    elif parents != 0:
        raise f"""
            `parents` must equal a nonnegative int but here, it is {parents}.
        """ from ParameterBoundError
    path = path / savedir
    path.mkdir(parents=True, exist_ok=True)

    # If file already exist, save with appending  number on the end
    if (path / (name + f".{stype}")).is_file() and not overwrite:
        print(
            "File of the same name already exists. Delete file or "
            + "set `overwrite` to `True`. Saving with appended integer."
        )
        ind = 1
        while True:
            if (path / (name + f"_{ind}.{stype}")).is_file():
                ind += 1
            else:
                name += f"_{ind}"
                break

    # Save as the appropriate type
    dr_txt = " This is a dryrun!" if dryrun else ""
    print(f"Saving to {path / (name + f'.{stype}')}.{dr_txt}")
    if not dryrun:
        if stype == "npz":
            np.savez(path / name, **files)
        elif stype == "pkl":
            with open(f"{path / name}.pkl", "wb") as savefile:
                pickle.dump(files, savefile)
