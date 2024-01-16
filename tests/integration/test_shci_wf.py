# MIT License
# 
# Copyright (c) 2019 Lucas K Wagner
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
import pyscf
import pyscf.hci
import pyqmc.api as pyq
from pyqmc.slater import Slater


def avg(vec):
    nblock = vec.shape[0]
    avg = np.mean(vec, axis=0)
    std = np.std(vec, axis=0)
    return avg, std / np.sqrt(nblock)


def test_shci_wf_is_better(H2_ccecp_hci):
    mol, mf, cisolver = H2_ccecp_hci

    configs = pyq.initial_guess(mol, 1000)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    data, configs = pyq.vmc(
        wf,
        configs,
        nblocks=40,
        verbose=True,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
    )
    en, err = avg(data["energytotal"][1:])
    nsigma = 4
    assert len(wf.parameters["det_coeff"]) == len(cisolver.ci)
    assert en - nsigma * err < mf.e_tot
    assert en + nsigma * err > cisolver.energy
