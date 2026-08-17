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
"""GeminalJastrowAccelerated must be the same wave function as GeminalJastrow.

The rewrite is algebraic, not an approximation, so every method is checked
against the original to machine precision rather than to a sampling tolerance.
"""

import numpy as np
import pytest
from pyscf import gto

import pyqmc.api as pyq
import pyqmc.wf.testwf as testwf
from pyqmc.wf.geminal_accelerated import GeminalJastrowAccelerated
from pyqmc.wf.geminaljastrow import GeminalJastrow

TOL = 1e-12


@pytest.fixture(scope="module")
def h2o():
    return gto.M(
        atom="O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587",
        ecp="ccecp",
        basis="ccecp-ccpvdz",
        unit="bohr",
        verbose=1,
    )


def make_pair(mol, nconfig=17, seed=0):
    """The two wave functions with identical parameters, on identical configs."""
    rng = np.random.default_rng(seed)
    old, new = GeminalJastrow(mol), GeminalJastrowAccelerated(mol)
    coeff = rng.normal(size=old.parameters["gcoeff"].shape) * 0.05
    old.parameters["gcoeff"] = coeff.copy()
    new.parameters["gcoeff"] = coeff.copy()
    np.random.seed(seed)
    configs = pyq.initial_guess(mol, nconfig)
    old.recompute(configs)
    new.recompute(configs)
    return old, new, configs, rng


def test_recompute_and_value(h2o):
    old, new, _, _ = make_pair(h2o)
    for a, b in zip(old.value(), new.value()):
        assert np.abs(a - b).max() < TOL


@pytest.mark.parametrize("e", [0, 3, 7])
def test_derivatives_match(h2o, e):
    """gradient, gradient_value, gradient_laplacian and testvalue all go through
    _compute_value, so each is checked at a displaced position."""
    old, new, configs, rng = make_pair(h2o)
    epos = configs.make_irreducible(
        e, configs.configs[:, e, :] + rng.normal(size=(len(configs.configs), 3)) * 0.3
    )

    assert np.abs(old.gradient(e, epos) - new.gradient(e, epos)).max() < TOL

    g_o, v_o, s_o = old.gradient_value(e, epos)
    g_n, v_n, s_n = new.gradient_value(e, epos)
    assert np.abs(g_o - g_n).max() < TOL
    assert np.abs(v_o - v_n).max() < TOL
    assert np.abs(s_o - s_n).max() < TOL

    gl_o = old.gradient_laplacian(e, epos)
    gl_n = new.gradient_laplacian(e, epos)
    for a, b in zip(gl_o, gl_n):
        assert np.abs(a - b).max() < TOL

    t_o, ao_o = old.testvalue(e, epos)
    t_n, ao_n = new.testvalue(e, epos)
    assert np.abs(t_o - t_n).max() < TOL
    assert np.abs(ao_o - ao_n).max() < TOL

    assert np.abs(old.pgradient()["gcoeff"] - new.pgradient()["gcoeff"]).max() < TOL


def test_testvalue_with_mask(h2o):
    old, new, configs, rng = make_pair(h2o)
    e = 2
    epos = configs.make_irreducible(
        e, configs.configs[:, e, :] + rng.normal(size=(len(configs.configs), 3)) * 0.3
    )
    mask = rng.integers(0, 2, len(configs.configs)).astype(bool)
    t_o, ao_o = old.testvalue(e, epos, mask)
    t_n, ao_n = new.testvalue(e, epos, mask)
    assert np.abs(t_o - t_n).max() < TOL
    assert np.abs(ao_o - ao_n).max() < TOL


def test_testvalue_many(h2o):
    old, new, configs, _ = make_pair(h2o)
    e_ = np.array([0, 2, 5])
    epos = configs.electron(1)
    assert np.abs(old.testvalue_many(e_, epos) - new.testvalue_many(e_, epos)).max() < TOL


def test_value_cache_survives_updates(h2o):
    """The cached value must track a sequence of partially accepted moves, and
    still agree with a from-scratch recompute."""
    old, new, configs, rng = make_pair(h2o)
    nconfig, nelec = configs.configs.shape[:2]

    for step in range(3):
        for e in range(nelec):
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] + rng.normal(size=(nconfig, 3)) * 0.2
            )
            accept = rng.integers(0, 2, nconfig).astype(bool)
            _, saved_o = old.testvalue(e, epos)
            _, saved_n = new.testvalue(e, epos)
            configs.move(e, epos, accept)
            old.updateinternals(e, epos, configs, mask=accept, saved_values=saved_o)
            new.updateinternals(e, epos, configs, mask=accept, saved_values=saved_n)

            v_o, v_n = old.value(), new.value()
            assert np.abs(v_o[1] - v_n[1]).max() < TOL, (step, e)

    # the incrementally maintained value equals a fresh recompute
    fresh = GeminalJastrowAccelerated(h2o)
    fresh.parameters["gcoeff"] = new.parameters["gcoeff"].copy()
    fresh.recompute(configs)
    assert np.abs(fresh.value()[1] - new.value()[1]).max() < 1e-10


def test_updateinternals_without_saved_values(h2o):
    """The saved_values=None path evaluates the orbitals itself."""
    old, new, configs, rng = make_pair(h2o)
    e = 4
    nconfig = len(configs.configs)
    epos = configs.make_irreducible(
        e, configs.configs[:, e, :] + rng.normal(size=(nconfig, 3)) * 0.2
    )
    configs.move(e, epos, np.ones(nconfig, dtype=bool))
    old.updateinternals(e, epos, configs)
    new.updateinternals(e, epos, configs)
    assert np.abs(old.value()[1] - new.value()[1]).max() < TOL


def test_generic_wf_checks(h2o):
    """The standard wave function test battery, as run for GeminalJastrow."""
    np.random.seed(0)
    wf = GeminalJastrowAccelerated(h2o)
    wf.parameters["gcoeff"] = (
        np.random.normal(size=wf.parameters["gcoeff"].shape) * 0.05
    )
    configs = pyq.initial_guess(h2o, 10)
    _, configs = pyq.vmc(wf, configs, nblocks=1, nsteps=2, tstep=1)

    for k, item in testwf.test_updateinternals(wf, configs).items():
        assert item < 1e-10, k
    testwf.test_mask(wf, 0, configs.electron(0))

    for func in [testwf.test_wf_gradient, testwf.test_wf_pgradient]:
        err = [func(wf, configs, delta) for delta in [1e-4, 1e-5, 1e-6]]
        assert min(err) < 1e-4, err
    assert testwf.test_wf_gradient_laplacian(wf, configs)["grad"] < 1e-8


def test_gcoeff_is_symmetric(h2o):
    """The rewrite merges the two terms of _compute_value using g_mn = g_nm, so
    pin the invariant that recompute establishes it."""
    _, new, _, _ = make_pair(h2o)
    assert np.abs(new.gcoeff - new.gcoeff.T).max() == 0.0


def test_value_result_is_owned_by_the_caller(h2o):
    """value() must not hand out the live cache: callers hold the array across
    updateinternals and compare against it, and gpu.asnumpy is a no-op on CPU."""
    _, new, configs, rng = make_pair(h2o)
    before = new.value()[1].copy()
    held = new.value()[1]
    e, nconfig = 0, len(configs.configs)
    epos = configs.make_irreducible(
        e, configs.configs[:, e, :] + rng.normal(size=(nconfig, 3)) * 0.3
    )
    _, saved = new.testvalue(e, epos)
    configs.move(e, epos, np.ones(nconfig, dtype=bool))
    new.updateinternals(e, epos, configs, saved_values=saved)
    assert np.array_equal(held, before)  # unchanged by the update
    assert not np.allclose(new.value()[1], before)  # but the wf did move
