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
from pyqmc.wftools import default_jastrow_basis
from pyqmc.wftools import generate_jastrow
from pyqmc.jastrowspin import JastrowSpin
from pyqmc.group_jastrow import GroupJastrowSpin
import pyqmc.api as pyq


# This test ensures that GroupJastrowSpin gives the same results as JastrowSpin
def run_tests(mol, configs, epsilon):
    a_basis, b_basis = default_jastrow_basis(mol)
    jastrow = JastrowSpin(mol, a_basis, b_basis)
    groupjastrow = GroupJastrowSpin(mol, a_basis, b_basis)

    configs = pyq.initial_guess(mol, 100)
    epos = configs.electron(0)
    epos.configs += 0.1

    jvals = call_methods(jastrow, configs, epos)
    gvals = call_methods(groupjastrow, configs, epos)

    for k, jitem in jvals.items():
        err = np.linalg.norm(jitem - gvals[k])
        print(k, err)
        assert err < epsilon, (k, err)


def call_methods(wf, configs, epos):
    vals = {}
    vals["recompute phase"], vals["recompute abs"] = wf.recompute(configs)
    vals["gradient"] = wf.gradient(0, epos)
    vals["gradient_value g"], vals["gradient_value v"], _ = wf.gradient_value(0, epos)
    vals["gradient_laplacian g"], vals["gradient_laplacian l"] = wf.gradient_laplacian(0, epos)
    vals["testvalue"] = wf.testvalue(0, epos)[0]
    return vals


def test_rohf(C_ccecp_rohf, epsilon=1e-5):
    mol, mf = C_ccecp_rohf
    configs = pyq.initial_guess(mol, 10)
    run_tests(mol, configs, epsilon)


def test_obc(LiH_sto3g_rhf, epsilon=1e-5, nconf=10):
    mol, mf = LiH_sto3g_rhf
    configs = pyq.initial_guess(mol, nconf)
    run_tests(mol, configs, epsilon)


def test_pbc_wfs(H_pbc_sto3g_krks, epsilon=1e-5, nconf=10):
    mol, mf = H_pbc_sto3g_krks
    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    configs = pyq.initial_guess(supercell, nconf)
    run_tests(mol, configs, epsilon)
