"""
Wrappers and functions used in different parts of the code.
"""
from datetime import datetime as dt


def timing(active: bool = True):
    """Decorator for timing with option to turn it off"""

    def decorator(func):
        def wrapper(inst, *args, **kwargs):
            if active:
                dt0 = dt.now()
                output = func(*args, **kwargs)
                dt1 = dt.now()
                tot_time = (dt1 - dt0).total_seconds()
                print(f"{func.__name__}: {tot_time:>{30 - len(func.__name__)}.3f}s")
            else:
                output = func(inst, *args, **kwargs)

            return output

        return wrapper

    return decorator


def check_attrs(*attrs):
    """
    Decorator that checks if the class instance has the attributes in `attrs`.
    If it doesn't, return the value as is rather than calculate it. So properties
    can "cache" their values if the largest arrays they rely on are deleted.
    """

    def decorator(func):
        def wrapper(inst, *args, **kwargs):
            for attr in attrs:
                if not hasattr(inst, attr):
                    # get hidden attribute name (has an underscore in the front)
                    return inst.__getattribute__(f"_{func.__name__}")
            return func(inst, *args, **kwargs)

        return wrapper

    return decorator
