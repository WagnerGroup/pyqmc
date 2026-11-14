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

"""Slater.recompute evaluates the AOs one spin at a time. Its determinants must equal those
built from one AO evaluation of all electrons."""
import numpy as np
import pyscf.gto
import pyscf.scf

import pyqmc.api as pyq
from pyqmc.wf.slater import Slater


def _reference_logdets(wf, configs):
    """log|det| and phase of every determinant, from the AOs of all electrons at once."""
    nconf, nelec, _ = configs.configs.shape
    aos = wf.orbitals.aos(wf._gtoval, configs)
    aos = aos.reshape(-1, nconf, nelec, aos.shape[-1])
    dets = []
    for s in [0, 1]:
        begin = wf._nelec[0] * s
        end = wf._nelec[0] + wf._nelec[1] * s
        mo = wf.orbitals.mos(aos[:, :, begin:end, :], s)
        mo_vals = np.swapaxes(mo[:, :, wf._det_occup[s]], 1, 2)
        dets.append(np.asarray(np.linalg.slogdet(mo_vals)))
    return dets


def _check(wf, configs):
    reference = _reference_logdets(wf, configs)
    for block in [None, 1, 3]:  # whole spin, one electron, blocks that do not divide nelec
        wf._ao_block = block
        value = wf.recompute(configs)
        for s in [0, 1]:
            assert np.allclose(wf._dets[s], reference[s], atol=1e-12), block
        assert np.all(np.isfinite(value[1]))


def test_recompute_pbc(H_pbc_sto3g_krks):
    mol, mf = H_pbc_sto3g_krks
    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    wf = Slater(supercell, mf, eval_gto_precision=1e-8)
    assert wf.dtype == complex
    _check(wf, pyq.initial_guess(supercell, 10))


def test_recompute_pbc_numba(H_pbc_sto3g_krks):
    """The numba AO evaluator, whose output also gets the wrap phase in place."""
    mol, mf = H_pbc_sto3g_krks
    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    configs = pyq.initial_guess(supercell, 10)
    wf = Slater(supercell, mf, eval_gto_precision=1e-8, evaluate_orbitals_with="numba")
    reference = Slater(supercell, mf, eval_gto_precision=1e-8)
    wf.recompute(configs)
    reference.recompute(configs)
    for s in [0, 1]:
        assert np.allclose(wf._dets[s], reference._dets[s], atol=1e-8)


def test_recompute_open(LiH_sto3g_rhf):
    mol, mf = LiH_sto3g_rhf
    wf = Slater(mol, mf)
    _check(wf, pyq.initial_guess(mol, 10))


def test_recompute_empty_spin():
    """A spin channel with no electrons."""
    mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", spin=2, unit="bohr", verbose=0)
    mf = pyscf.scf.ROHF(mol).run()
    wf = Slater(mol, mf)
    assert wf._nelec[1] == 0
    _check(wf, pyq.initial_guess(mol, 10))


def test_recompute_zero_determinant(LiH_sto3g_rhf):
    """A walker whose determinant is zero (two same-spin electrons at one point) takes the
    masked path: its inverse is left zero and the other walkers are unaffected."""
    import warnings

    mol, mf = LiH_sto3g_rhf
    wf = Slater(mol, mf)
    configs = pyq.initial_guess(mol, 6)
    configs.configs[0, 1] = configs.configs[0, 0]  # electrons 0 and 1 are both spin up
    reference = _reference_logdets(wf, configs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = wf.recompute(configs)
    assert not np.isfinite(value[1][0]) or value[0][0] == 0
    assert np.all(wf._inverse[0][0] == 0)
    assert np.all(np.isfinite(value[1][1:]))
    for s in [0, 1]:
        assert np.allclose(wf._dets[s][:, 1:], reference[s][:, 1:], atol=1e-12)
    # the inverses of the other walkers are still inverses
    nup = wf._nelec[0]
    aos = wf.orbitals.aos(wf._gtoval, configs.select_electrons(np.arange(nup)))
    mo = wf.orbitals.mos(aos.reshape(aos.shape[0], 6, nup, -1), 0)[:, :, wf._det_occup[0][0]]
    identity = np.einsum("cij,cjk->cik", mo[1:], wf._inverse[0][1:, 0])
    assert np.allclose(identity, np.eye(nup), atol=1e-10)
