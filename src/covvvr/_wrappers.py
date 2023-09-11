"""
Wrappers and functions used in different parts of the code.
"""
from datetime import datetime as dt


def timing(func):
    def wrapper(self, *args, **kwargs):
        if self.TIMING:
            dt0 = dt.now()
            output = func(self, *args, **kwargs)
            dt1 = dt.now()

            tot_time = (dt1 - dt0).total_seconds()
            fname = func.__name__
            print(
                f"{self.timing_count:>4}: {fname} "
                + f"{'-' * (50 - len(fname))} {tot_time:.3f}s"
            )
            self.timing_count += 1
        else:
            output = func(self, *args, **kwargs)

        return output

    return wrapper


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
