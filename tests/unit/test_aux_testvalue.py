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

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from pyqmc.api import (
    generate_wf,
    generate_jastrow,
    initial_guess,
)
from pyqmc.gpu import cp, asnumpy
import pyqmc.wf.testwf as testwf
import pytest
from pyqmc.wf.slater import Slater
from pyqmc.wf.multiplywf import MultiplyWF
from pyqmc.wf.addwf import AddWF
from pyqmc.wf.geminaljastrow import GeminalJastrow
from pyqmc.wftools import generate_jastrow,  default_jastrow_basis
from pyqmc.wf.three_body_jastrow import ThreeBodyJastrow


def test_testvalue_aux(LiH_ccecp_rhf, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    mol, mf = LiH_ccecp_rhf
    a_basis, b_basis = default_jastrow_basis(mol)
    for wf in [
        generate_jastrow(mol)[0],
        GeminalJastrow(mol),
        ThreeBodyJastrow(mol, a_basis, b_basis),
        MultiplyWF(
            Slater(mol, mf),
            generate_jastrow(mol)[0],
            ThreeBodyJastrow(mol, a_basis, b_basis),
        ),
        Slater(mol, mf),
    ]:
        for k in wf.parameters:
            if k != "mo_coeff":
                wf.parameters[k] = cp.asarray(np.random.rand(*wf.parameters[k].shape))

        naux = 6
        nelec = sum(mol.nelec)
        configs =  initial_guess(mol, nconf)
        aux =  initial_guess(mol, naux * (int(nconf / nelec) + 1))
        aux.reshape((-1, naux, 3))
        aux.resample(range(nconf))
        print(configs.configs.shape)
        print(aux.configs.shape)
        print(type(wf))
        testwf.test_testvalue_aux(wf, configs, aux)


