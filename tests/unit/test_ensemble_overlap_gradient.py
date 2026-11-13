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

"""The overlap penalty gradient in EnsembleUpdater.delta_p is the derivative of
lambda |S_ij|^2, for real and complex wave functions.

On a fixed set of configurations with the sampling density frozen, the overlap
estimator is a deterministic function of the parameters, so its derivative can
be checked by finite differences."""

import copy

import numpy as np
import pytest

import pyqmc.api as pyq
import pyqmc.method.sample_many
from pyqmc.method.ensemble_updaters import EnsembleUpdater
from pyqmc.method.optimize import set_wf_params
from pyqmc.observables.accumulators import LinearTransform


class _GradientOnly:
    """Stand-in single-state updater whose solve returns the extra gradient."""

    def solve(self, data, g=None):
        return g, {"pgrad": 0.0, "SRdot": 0.0}


def make_states(mol, mf, mc, complex_orbitals):
    mc = copy.copy(mc)
    if complex_orbitals:
        # generate_slater takes the orbitals from mc
        rng = np.random.default_rng(7)
        mo = mc.mo_coeff * np.exp(1j * rng.uniform(0, 2 * np.pi, mc.mo_coeff.shape[1]))
        mc.mo_coeff = mo + 0.05j * rng.normal(size=mo.shape)
    wfs = []
    for i in range(2):
        mci = copy.copy(mc)
        mci.ci = mc.ci[i]
        wf, to_opt = pyq.generate_slater(
            mol, mf, mc=mci, optimize_determinants=True, optimize_orbitals=True, tol=1e-20
        )
        wfs.append(wf)
    return wfs, LinearTransform(wfs[-1].parameters, to_opt)


def psi_over_sqrt_rho(wfs, ref, rho):
    phase, logval = [np.array(x) for x in zip(*[wf.value() for wf in wfs])]
    return phase * np.exp(logval - ref) / np.sqrt(rho)


@pytest.mark.parametrize("complex_orbitals", [False, True])
def test_overlap_gradient_finite_difference(H2_casci, complex_orbitals):
    mol, mf, mc = H2_casci
    wfs, transform = make_states(mol, mf, mc, complex_orbitals)
    # equilibrated configurations; near-node walkers from the initial guess make
    # the finite differences inaccurate
    configs = pyq.initial_guess(mol, 200, rng=np.random.default_rng(3))
    _, configs = pyq.vmc(wfs[0], configs, nblocks=1, nsteps_per_block=20, rng=np.random.default_rng(4))
    for wf in wfs:
        wf.recompute(configs)
    assert np.iscomplexobj(wfs[-1].pgradient()["mo_coeff_alpha"]) == complex_orbitals

    # sampling density frozen at the base parameters
    logval = np.array([wf.value()[1] for wf in wfs])
    ref = np.max(logval, axis=0)
    rho = np.mean(np.exp(2 * (logval - ref)), axis=0)

    # analytic gradient from the code
    weights = pyqmc.method.sample_many.compute_weights(wfs)
    dp = transform.serialize_gradients(wfs[-1].pgradient())
    wtdp = np.einsum("cp,jkc->pjk", dp, weights) / weights.shape[-1]
    up = EnsembleUpdater(_GradientOnly())
    energy_data = {"energy": 0.0, "energy_error": 0.0}
    avg, _ = up.block_average(energy_data, {"wtdp": wtdp[None]}, weights.mean(axis=-1)[None])
    penalty = np.full((2, 2), 0.7)
    (g,), _ = up.delta_p([-1.0], avg, penalty)

    # finite differences of penalty * |S_10|^2 in this state's parameters
    x0 = transform.serialize_parameters(wfs[-1].parameters)

    def cost(x):
        set_wf_params(wfs[-1], x, transform)
        wfs[-1].recompute(configs)
        p = psi_over_sqrt_rho(wfs, ref, rho)
        S = p.conj() @ p.T / p.shape[1]
        return penalty[1, 0] * np.abs(S[1, 0]) ** 2 / np.real(S[0, 0] * S[1, 1])

    h = 1e-5
    fd = np.zeros_like(x0)
    for k in range(len(x0)):
        e = np.zeros_like(x0)
        e[k] = h
        fd[k] = (cost(x0 + e) - cost(x0 - e)) / (2 * h)
    set_wf_params(wfs[-1], x0, transform)

    scale = np.abs(fd).max()
    assert scale > 1e-6  # the states must overlap on these configurations
    assert np.abs(g - fd).max() / scale < 1e-5
