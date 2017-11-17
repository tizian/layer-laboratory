"""
Miscellaneous utility functions for tests (common fixtures,
test decorators, etc).
"""

import os

from functools import wraps
from inspect import getframeinfo, stack
from mitsuba.core import Thread, FileResolver


def fresolver_append_path(func):
    """Function decorator that adds the mitsuba project root
    to the FileResolver's search path. This is useful in particular
    for tests that e.g. load scenes, and need to specify paths to resources.

    The file resolver is restored to its previous state once the test's
    execution has finished.

    In a function
    """
    par = os.path.dirname

    # Get the path to the source file from which this function is
    # being called.
    # Source: https://stackoverflow.com/a/24439444/3792942
    caller = getframeinfo(stack()[1][0])
    caller_path = par(caller.filename)

    # Heuristic to find the project's root directory
    def is_root(path):
        children = os.listdir(path)
        return ('ext' in children) and ('include' in children) \
               and ('src' in children) and ('resources' in children)
    root_path = caller_path
    while not is_root(root_path) and (par(root_path) != root_path):
        root_path = par(root_path)


    # The @wraps decorator properly sets __name__ and other properties, so that
    # pytest-xdist can keep track of the original test function.
    @wraps(func)
    def f(*args, **kwargs):
        # New file resolver
        thread = Thread.thread()
        fres_old = thread.file_resolver()
        fres = FileResolver(fres_old)

        # Append current test directory and project root to the
        # search path.
        fres.append(caller_path)
        fres.append(root_path)

        thread.set_file_resolver(fres)

        # Run actual function
        res = func(*args, **kwargs)

        # Restore previous file resolver
        thread.set_file_resolver(fres_old)

        return res

    return f