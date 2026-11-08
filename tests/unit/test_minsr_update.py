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

import numpy as np
import pytest

from pyqmc.method.cgsr import cgsr_direction
from pyqmc.method.minsr import minsr_direction, real_design_matrix
from pyqmc.observables.stochastic_reconfiguration import sr_averages, sr_direction


def make_data(seed, nsamples, nparams, complex_values=False):
    rng = np.random.default_rng(seed)
    dppsi = rng.normal(size=(nsamples, nparams))
    eloc = rng.normal(size=nsamples)
    if complex_values:
        dppsi = dppsi + 1j * rng.normal(size=(nsamples, nparams))
        eloc = eloc + 1j * rng.normal(size=nsamples)
    return dppsi, eloc


def sr_direction_from_samples(dppsi, eloc, eps, inverse_strategy="regularized_inverse", g=None):
    """The explicit-S direction, built through the same averages the accumulator uses."""
    return sr_direction(sr_averages({"total": eloc}, dppsi), eps, inverse_strategy, g=g)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("nsamples,nparams", [(20, 50), (50, 20), (30, 30)])
def test_solvers_agree(nsamples, nparams, complex_values):
    """Explicit S, minSR, and CG give the same direction, whether there are more
    samples than parameters or fewer, and for complex derivatives."""
    dppsi, eloc = make_data(12345, nsamples, nparams, complex_values)
    eps = 1e-2
    v_sr, report = sr_direction_from_samples(dppsi, eloc, eps)
    v_minsr, _ = minsr_direction(dppsi, eloc, eps)
    v_cg, _ = cgsr_direction(dppsi, eloc, eps, rtol=1e-12)
    assert np.allclose(v_sr, v_minsr, atol=1e-10), np.abs(v_sr - v_minsr).max()
    assert np.allclose(v_sr, v_cg, atol=1e-8), np.abs(v_sr - v_cg).max()
    assert report["SRdot"] > 0


def test_extra_gradient():
    """An extra gradient term g enters every solver the same way."""
    dppsi, eloc = make_data(3, 25, 40, complex_values=True)
    g = np.random.default_rng(4).normal(size=40)
    eps = 1e-2
    v_sr, _ = sr_direction_from_samples(dppsi, eloc, eps, g=g)
    v_minsr, _ = minsr_direction(dppsi, eloc, eps, g=g)
    v_cg, _ = cgsr_direction(dppsi, eloc, eps, rtol=1e-12, g=g)
    assert np.allclose(v_sr, v_minsr, atol=1e-8)
    assert np.allclose(v_sr, v_cg, atol=1e-8)


def test_minsr_solves_sr_equations():
    """The direction satisfies (S + eps) v = f with the conjugated definitions
    of S and f, for complex derivatives and local energies."""
    dppsi, eloc = make_data(54321, 25, 40, complex_values=True)
    eps = 1e-3
    v, _ = minsr_direction(dppsi, eloc, eps=eps)

    A, b = real_design_matrix(dppsi, eloc)
    S = A.T @ A
    f = 2 * A.T @ b
    residual = (S + eps * np.eye(S.shape[0])) @ v - f
    assert np.linalg.norm(residual) < 1e-10 * np.linalg.norm(f)
    assert v.dtype == np.dtype(float)


def test_minsr_pseudo_inverse():
    """Without regularization, minSR gives the minimum-norm solution of the SR
    equations, which is the same direction as the SR pseudo-inverse."""
    dppsi, eloc = make_data(99, 15, 60)
    eps = 1e-8
    v_sr, _ = sr_direction_from_samples(dppsi, eloc, eps, "pseudo_inverse")
    v_minsr, _ = minsr_direction(dppsi, eloc, eps=eps, inverse_strategy="pseudo_inverse")
    assert np.allclose(v_sr, v_minsr, atol=1e-6), np.abs(v_sr - v_minsr).max()
