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
import pyscf.gto
import pyscf.scf
import pytest
import pyqmc.api as pyq
import pyqmc.method.mc
from pyqmc.observables.ecp import ThresholdECP, FixedBudgetECP, default_naip


@pytest.fixture(scope="module")
def si_atom():
    """Si has s and p nonlocal channels, so P_l takes both signs."""
    mol = pyscf.gto.M(
        atom="Si 0. 0. 0.", basis="ccecpccpvdz", ecp="ccecp", spin=2, unit="bohr", verbose=0
    )
    mf = pyscf.scf.ROHF(mol).run()
    wf, _ = pyq.generate_wf(mol, mf)
    configs = pyq.initial_guess(mol, 20)
    _, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1, nsteps_per_block=5)
    wf.recompute(configs)
    return mol, wf, configs


def test_exact_agreement(si_atom):
    """With one ECP atom the fixed budget covers every quadrature point, and with
    threshold=-1 and no random rotation both methods evaluate the same points.
    Energies and T-moves must then agree exactly."""
    mol, wf, configs = si_atom
    threshold = ThresholdECP(mol, threshold=-1, stochastic_rotation=False)
    fixed = FixedBudgetECP(mol, stochastic_rotation=False)
    assert np.array_equal(threshold.naip, [12])
    assert np.allclose(threshold(configs, wf), fixed(configs, wf))
    for e in [0, 3]:
        t_threshold = threshold.nonlocal_tmoves(configs, wf, e, 0.1)
        t_fixed = fixed.nonlocal_tmoves(configs, wf, e, 0.1)
        assert np.allclose(t_threshold["ratio"], t_fixed["ratio"])
        assert np.allclose(t_threshold["weight"], t_fixed["weight"])
        assert np.any(t_fixed["weight"] != 0)


def test_default_naip():
    mol = pyscf.gto.M(
        atom="H 0 0 0; C 0 0 2; Si 0 0 4", basis="ccecpccpvdz", ecp="ccecp", spin=1, verbose=0
    )
    for cls in [ThresholdECP, FixedBudgetECP]:
        assert np.array_equal(cls(mol).naip, [0, 6, 12])
        assert np.array_equal(cls(mol, naip=12).naip, [0, 12, 12])
    with pytest.raises(ValueError):
        default_naip([{}, {}], naip=[6])


def test_fixed_selection_unbiased(diamond_primitive):
    """The fixed budget's random selection averages to the full evaluation."""
    mol, mf = diamond_primitive
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10)
    _, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)

    deterministic = FixedBudgetECP(mol, stochastic_rotation=False, nselect_deterministic=1000)
    stochastic = FixedBudgetECP(mol, stochastic_rotation=False)
    # the ECP energy is real; for PBC wave functions it comes back complex with
    # a roundoff-level imaginary part, so take the real part explicitly
    ref = deterministic(configs, wf).real
    samples = np.array([stochastic(configs, wf).real for _ in range(100)])
    assert np.mean(np.abs(samples.mean(axis=0) - ref)) < 0.001


def test_methods_agree_on_average(diamond_primitive):
    """With random rotations, the two methods agree within statistical error."""
    mol, mf = diamond_primitive
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10)
    _, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)

    nsample = 100
    methods = {"threshold": ThresholdECP(mol, threshold=-1), "fixed": FixedBudgetECP(mol)}
    samples = {k: np.array([m(configs, wf).real for _ in range(nsample)]) for k, m in methods.items()}
    avg = {k: s.mean(axis=0) for k, s in samples.items()}
    err = np.sqrt(sum(s.var(axis=0, ddof=1) / nsample for s in samples.values()))
    assert np.all(np.abs(avg["threshold"] - avg["fixed"]) < 5 * err + 1e-8)
