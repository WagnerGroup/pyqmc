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
    value = wf.recompute(configs)
    reference = _reference_logdets(wf, configs)
    for s in [0, 1]:
        assert np.allclose(wf._dets[s], reference[s], atol=1e-12)
    assert np.all(np.isfinite(value[1]))


def test_recompute_pbc(H_pbc_sto3g_krks):
    mol, mf = H_pbc_sto3g_krks
    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    wf = Slater(supercell, mf, eval_gto_precision=1e-8)
    assert wf.dtype == complex
    _check(wf, pyq.initial_guess(supercell, 10))


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
