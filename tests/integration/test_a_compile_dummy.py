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
import pyqmc.api as pyq
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.slater import Slater


def test_a_compile_dummy(LiH_ccecp_rhf):

    """
    Checks that VMC energy matches energy calculated in PySCF
    """
    nsteps = 1

    mol, mf = LiH_ccecp_rhf
    wf = Slater(mol, mf)
    nconf = 10
    coords = pyq.initial_guess(mol, nconf)
    df, coords = pyq.vmc(
        wf, coords, nsteps=nsteps, accumulators={"energy": EnergyAccumulator(mol)}
    )

    assert df is not None
