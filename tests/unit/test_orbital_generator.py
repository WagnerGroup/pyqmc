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
import pyqmc.testwf as testwf
from pyqmc.gpu import cp, asnumpy
from pyqmc.slater import Slater
from pyqmc.multiplywf import MultiplyWF
from pyqmc.addwf import AddWF
from pyqmc.geminaljastrow import GeminalJastrow
from pyqmc.wftools import generate_jastrow
import pyqmc.api as pyq
from pyqmc.three_body_jastrow import ThreeBodyJastrow
from pyqmc.wftools import default_jastrow_basis


def test_pbc_wfs_triplet(h_noncubic_sto3g_triplet, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """
    mol, mf = h_noncubic_sto3g_triplet

    # supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    supercell = pyq.get_supercell(mol, S=np.identity(3, dtype=int))
    epos = pyq.initial_guess(supercell, nconf)
    wf = Slater(supercell, mf, eval_gto_precision=1e-6)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2
    wf.recompute(epos)


def test_hci_wf(H2_ccecp_hci, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_hci
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2
    wf.recompute(configs)


def test_rohf(C_ccecp_rohf, epsilon=1e-5):
    mol, mf = C_ccecp_rohf
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf)
    wf.recompute(configs)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2


def test_casci_s0(H2_ccecp_casci_s0, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_casci_s0
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    wf.recompute(configs)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2


def test_casci_s2(H2_ccecp_casci_s2, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_casci_s2
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    wf.recompute(configs)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2


def test_casscf_s0(H2_ccecp_casscf_s0, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_casscf_s0
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    wf.recompute(configs)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2


def test_casscf_s2(H2_ccecp_casscf_s2, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_casscf_s2
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    wf.recompute(configs)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2


@pytest.mark.slow
def test_casscf_pbc(h_pbc_casscf, epsilon=1e-5):
    mol, mf, cisolver = h_pbc_casscf
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    wf.recompute(configs)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2



def test_manual_slater(H2_ccecp_rhf, epsilon=1e-5):
    mol, mf = H2_ccecp_rhf

    determinants = [(1.0, [[0], [0]]), (-0.2, [[1], [1]])]
    wf = Slater(mol, mf, determinants=determinants)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2
    configs = pyq.initial_guess(mol, 10)
    wf.recompute(configs)


def test_manual_pbcs_correct(H_pbc_sto3g_kuks, epsilon=1e-5, nconf=10):
    """
    This test makes sure that the number of k-points must match the number of k-points
    in the mf object.
    """
    from pyqmc.pyscftools import single_determinant_from_mf

    mol, mf = H_pbc_sto3g_kuks
    supercell = np.identity(3, dtype=int)
    supercell[0, 0] = 2
    mol = pyq.get_supercell(mol, supercell)

    determinants = [
        single_determinant_from_mf(mf, 1.0)[0],
        single_determinant_from_mf(mf, -0.2)[0],
    ]
    for s, ka, a, ki, i in [(0, 0, 0, 0, 1)]:
        determinants[1][1][s][ka].remove(a)
        determinants[1][1][s][ki].append(i)

    print(determinants[0])
    wf = Slater(mol, mf, determinants=determinants, eval_gto_precision=1e-6)
    for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
        assert len(wf.parameters[k].shape) == 2
    configs = pyq.initial_guess(mol, 10)
    wf.recompute(configs)


