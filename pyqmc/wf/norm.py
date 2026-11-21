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

r"""A settable normalization for wave functions.

A wave function that can set its norm holds a parameter ``log_norm``, and its
``recompute()`` and ``value()`` return :math:`\ln|\psi| - {\rm log\_norm}`. It
is saved and read with the other parameters, so a set of wave functions
normalized relative to each other (for example by ensemble optimization) can be
read back and used directly, as in ``sample_overlap``.

Only the absolute value of the wave function changes. Ratios, gradients,
Laplacians, and parameter derivatives do not, so ``log_norm`` never enters the
parameter gradients, and it is not optimized: :class:`LinearTransform` leaves it
out when it is built without an explicit ``to_opt``.

The interface is three methods, ``can_set_norm()``, ``log_norm()``, and
``set_log_norm(value)``; a wave function without them cannot set its norm.
"""

import numpy as np

NORM_KEY = "log_norm"


def is_norm_key(key):
    """Whether a parameter name is a log_norm, including prefixed names such as wf1log_norm."""
    return key.endswith(NORM_KEY)


def can_set_norm(wf):
    """Whether wf can set its norm."""
    method = getattr(wf, "can_set_norm", None)
    return method is not None and method()


def new_norm_parameter():
    """The initial log_norm. Shape (1,) rather than a scalar, so that it is stored
    and updated in the checkpoint like the other parameters."""
    return np.zeros(1)


class SettableNorm:
    """Mixin for a wave function that stores ``log_norm`` in ``self.parameters``
    and subtracts it in ``recompute()`` and ``value()``."""

    def can_set_norm(self):
        return True

    def log_norm(self):
        return float(self.parameters[NORM_KEY][0])

    def set_log_norm(self, log_norm):
        """Divide the wave function by exp(log_norm), replacing the current factor."""
        self.parameters[NORM_KEY] = np.array([float(log_norm)])
