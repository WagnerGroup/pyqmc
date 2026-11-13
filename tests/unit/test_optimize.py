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
import pyqmc.api as pyq
import pyqmc.method.mc
from pyqmc.method.optimize import optimize, MinSRUpdate
from pyqmc.observables.accumulators import LinearTransform
from pyqmc.observables.stochastic_reconfiguration import (
    StochasticReconfiguration,
    nodal_regularization,
)

SHORT = dict(
    warmup_options=dict(nblocks=1, nsteps_per_block=2),
    vmc_kwargs=dict(nblocks=1, nsteps_per_block=2),
)


@pytest.fixture
def lih(LiH_sto3g_rhf):
    mol, mf = LiH_sto3g_rhf
    wf, to_opt = pyq.generate_wf(mol, mf)
    configs = pyq.initial_guess(mol, 30)
    _, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1, nsteps_per_block=5)
    wf.recompute(configs)
    return mol, wf, to_opt, configs


def test_sr_accumulator_regularization(lih):
    """The SR accumulator uses its own nodal_cutoff, and S is the Gram matrix of
    the regularized derivatives, so f enters S squared."""
    mol, wf, to_opt, configs = lih
    transform = LinearTransform(wf.parameters, to_opt)
    enacc = pyq.EnergyAccumulator(mol)
    cutoff = 1.0  # large enough that most configurations are regularized
    avg = StochasticReconfiguration(enacc, transform, nodal_cutoff=cutoff).avg(configs, wf)

    den = enacc(configs, wf)
    _, f = nodal_regularization(den["grad2"], cutoff)
    assert np.any(f != 1.0)
    dp = transform.serialize_gradients(wf.pgradient()) * f[:, np.newaxis]
    nconf = dp.shape[0]
    assert np.allclose(avg["dppsi"], dp.mean(axis=0))
    assert np.allclose(avg["dpidpj"], np.conj(dp).T @ dp / nconf)
    assert np.allclose(avg["dpH"], den["total"] @ np.conj(dp) / nconf)


def test_max_norm_clips_the_step(lih):
    mol, wf, to_opt, configs = lih
    transform = LinearTransform(wf.parameters, to_opt)
    x0 = transform.serialize_parameters(wf.parameters)
    updater = MinSRUpdate(pyq.EnergyAccumulator(mol), transform)
    max_norm = 1e-4
    wf, df = optimize(wf, configs, updater, tstep=10.0, max_norm=max_norm, max_iterations=1, **SHORT)
    assert df[0]["clipped"]
    assert df[0]["step_norm"] == pytest.approx(max_norm)
    x1 = transform.serialize_parameters(wf.parameters)
    assert np.linalg.norm(x1 - x0) == pytest.approx(max_norm)


def test_line_minimization_is_deprecated(lih):
    mol, wf, to_opt, configs = lih
    acc = pyq.gradient_generator(mol, wf, to_opt)
    with pytest.warns(DeprecationWarning):
        wf, df = pyq.line_minimization(
            wf, configs, acc, max_iterations=1, npts=5,
            warmup_options=SHORT["warmup_options"], vmcoptions=SHORT["vmc_kwargs"],
        )
    assert "est_min" in df[0]
