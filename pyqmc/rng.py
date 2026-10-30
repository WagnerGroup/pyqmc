# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
"""
Random number generation for pyqmc.

pyqmc never touches numpy's global random state. All random numbers come from a
module-level numpy Generator, seeded from OS entropy at import, or from a
Generator passed in explicitly. Public functions take ``rng=None`` following
Scientific Python SPEC 7; ``get_rng`` normalizes that argument.

Code that draws random numbers deep inside an evaluation (e.g. ECP quadrature
rotations inside an accumulator) calls ``get_rng()`` with no argument. A worker
that has been handed its own Generator runs inside ``using(rng)``, so those
draws come from the worker's stream as well.

For reproducible runs, call ``pyqmc.rng.seed(s)`` or pass ``rng=s`` to an entry point.
"""
import contextlib
import functools
import numpy as np

_rng = np.random.default_rng()  # OS entropy at import


def get_rng(rng=None):
    """Normalize an ``rng`` argument to a Generator.

    :parameter rng: None returns pyqmc's module generator. An int or SeedSequence
        creates a new Generator. A Generator is returned unchanged, so passing it
        through several layers continues one stream.
    :returns: a numpy Generator
    """
    return _rng if rng is None else np.random.default_rng(rng)


def seed(s=None):
    """Replace pyqmc's module generator with ``default_rng(s)``.

    :parameter s: an int, SeedSequence, or Generator. None draws fresh OS entropy.
    """
    global _rng
    _rng = np.random.default_rng(s)


@contextlib.contextmanager
def using(rng):
    """Make ``rng`` the module generator inside the block, then restore the old one.

    Workers use this so that ``get_rng()`` calls in accumulators draw from the
    worker's stream. The module generator is shared by all threads in a process,
    so do not use this from concurrently running threads.

    :parameter rng: anything accepted by ``get_rng`` other than None
    """
    global _rng
    old = _rng
    _rng = np.random.default_rng(rng)
    try:
        yield _rng
    finally:
        _rng = old


def worker(f):
    """Decorator for a function that does the random work for one partition of walkers.

    The decorated function takes a keyword-only ``rng=None``. The wrapper resolves it
    with ``get_rng``, passes the Generator on, and runs the function inside
    ``using(rng)``, so accumulators called from it draw from the same stream as
    the function's own moves.
    """

    @functools.wraps(f)
    def wrapped(*args, rng=None, **kwargs):
        rng = get_rng(rng)
        with using(rng):
            return f(*args, rng=rng, **kwargs)

    return wrapped
