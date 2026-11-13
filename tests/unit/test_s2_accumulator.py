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
from pyscf import gto, scf
import pyqmc.api as pyq


def _slater(mol, mf):
    wf, _ = pyq.generate_slater(mol, mf)
    return wf


@pytest.mark.parametrize(
    "atom,spin,expected",
    [
        ("Li 0. 0. 0.; H 0. 0. 1.5", 0, 0.0),
        ("Li 0. 0. 0.", 1, 0.75),
    ],
)
def test_s2_eigenstate(atom, spin, expected):
    """A single restricted determinant is an S^2 eigenstate, so the local
    estimator equals s(s+1) at every configuration."""
    mol = gto.M(atom=atom, basis="sto-3g", unit="bohr", spin=spin, verbose=0)
    mf = scf.ROHF(mol).run()
    wf = _slater(mol, mf)
    configs = pyq.initial_guess(mol, 50)
    wf.recompute(configs)
    before = configs.configs.copy()

    s2 = pyq.S2Accumulator(mol.nelec)(configs, wf)["S2"]
    assert np.max(np.abs(s2 - expected)) < 1e-8
    # The accumulator must leave configs unchanged.
    assert np.allclose(configs.configs, before)


def test_s2_uhf_contamination():
    """For a spin-contaminated UHF determinant, the VMC average of S^2 should
    match pyscf's <S^2>."""
    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 4.0", basis="sto-3g", unit="bohr", verbose=0)
    mf = scf.UHF(mol)
    # Broken-symmetry guess: alpha density on the first atom, beta on the second.
    mf.kernel(dm0=np.array([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]))
    s2_ref, _ = mf.spin_square()
    assert s2_ref > 0.5, "UHF guess did not break spin symmetry"

    wf = _slater(mol, mf)
    configs = pyq.initial_guess(mol, 1000)
    df, _ = pyq.vmc(
        wf,
        configs,
        nblocks=20,
        nsteps_per_block=5,
        accumulators={"s2": pyq.S2Accumulator(mol.nelec)},
    )
    s2 = df["s2S2"][4:]
    err = np.std(s2) / np.sqrt(len(s2))
    assert abs(np.mean(s2) - s2_ref) < 5 * err + 1e-3
