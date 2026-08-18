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

from pyqmc.method.minsr import minsr_update, real_design_matrix
from pyqmc.observables.stochastic_reconfiguration import StochasticReconfiguration


def sr_data(dppsi, eloc):
    """Averages in the form that StochasticReconfiguration.delta_p expects."""
    return {
        "total": np.mean(eloc),
        "dppsi": np.mean(dppsi, axis=0),
        "dpH": np.mean(eloc[:, np.newaxis] * dppsi, axis=0),
        "dpidpj": np.einsum("ij,ik->jk", dppsi, dppsi) / dppsi.shape[0],
    }


@pytest.mark.parametrize("nsamples,nparams", [(20, 50), (50, 20), (30, 30)])
def test_minsr_equals_sr(nsamples, nparams):
    """minSR and regularized SR give the same update, whether there are more
    samples than parameters or fewer."""
    rng = np.random.default_rng(seed=12345)
    dppsi = rng.normal(size=(nsamples, nparams))
    eloc = rng.normal(size=nsamples)
    tstep = 0.13
    eps = 1e-2

    sr = StochasticReconfiguration(
        None, None, eps=eps, inverse_strategy="regularized_inverse"
    )
    dp_sr = sr.delta_p([tstep], sr_data(dppsi, eloc))[0][0]
    dp_minsr, report = minsr_update(dppsi, eloc, tstep, eps=eps)

    assert np.allclose(dp_sr, dp_minsr, atol=1e-10), np.abs(dp_sr - dp_minsr).max()
    assert report["SRdot"] > 0


def test_minsr_solves_sr_equations():
    """The update satisfies (S + eps) dp = -tstep * f with the conjugated
    definitions of S and f, for complex derivatives and local energies."""
    rng = np.random.default_rng(seed=54321)
    nsamples, nparams = 25, 40
    dppsi = rng.normal(size=(nsamples, nparams)) + 1j * rng.normal(
        size=(nsamples, nparams)
    )
    eloc = rng.normal(size=nsamples) + 1j * rng.normal(size=nsamples)
    tstep, eps = 0.05, 1e-3

    dp, _ = minsr_update(dppsi, eloc, tstep, eps=eps)

    A, b = real_design_matrix(dppsi, eloc)
    S = A.T @ A
    f = 2 * A.T @ b
    residual = (S + eps * np.eye(nparams)) @ dp + tstep * f
    assert np.linalg.norm(residual) < 1e-10 * np.linalg.norm(tstep * f)
    assert dp.dtype == np.dtype(float)


def test_minsr_pseudo_inverse():
    """Without regularization, minSR gives the minimum-norm solution of the SR
    equations, which is the same direction as the SR pseudo-inverse update."""
    rng = np.random.default_rng(seed=99)
    nsamples, nparams = 15, 60
    dppsi = rng.normal(size=(nsamples, nparams))
    eloc = rng.normal(size=nsamples)
    tstep, eps = 0.1, 1e-8

    sr = StochasticReconfiguration(
        None, None, eps=eps, inverse_strategy="pseudo_inverse"
    )
    dp_sr = sr.delta_p([tstep], sr_data(dppsi, eloc))[0][0]
    dp_minsr, _ = minsr_update(
        dppsi, eloc, tstep, eps=eps, inverse_strategy="pseudo_inverse"
    )
    assert np.allclose(dp_sr, dp_minsr, atol=1e-6), np.abs(dp_sr - dp_minsr).max()


def test_minsr_max_norm():
    rng = np.random.default_rng(seed=7)
    dppsi = rng.normal(size=(30, 30))
    eloc = rng.normal(size=30)
    dp_unclipped, _ = minsr_update(dppsi, eloc, 1.0, eps=1e-3)
    max_norm = 0.5 * np.linalg.norm(dp_unclipped)
    dp, report = minsr_update(dppsi, eloc, 1.0, eps=1e-3, max_norm=max_norm)
    assert np.isclose(np.linalg.norm(dp), max_norm)
    assert report["clipped"]
    # clipping only rescales; the direction is unchanged
    assert np.allclose(dp, dp_unclipped * max_norm / np.linalg.norm(dp_unclipped))
