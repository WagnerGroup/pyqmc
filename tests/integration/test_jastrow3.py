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
import pandas as pd
from pyqmc.api import (
    generate_wf,
    generate_jastrow,
    line_minimization,
    initial_guess,
    gradient_generator,
)
import pyqmc.wftools as wftools
import pytest


@pytest.mark.slow
def test_linemin_three_body(H2_ccecp_uhf):
    """Optimize a Slater-3bodyJastrow wave function and check that it's better than Hartree-Fock"""
    mol, mf = H2_ccecp_uhf
    mol.stdout = None
    mol.output = None
    wf, to_opt = generate_wf(
        mol,
        mf,
        jastrow=[generate_jastrow, wftools.generate_jastrow3],
        jastrow_kws=[{}, {}],
    )
    nconf = 100
    wf, dfgrad = line_minimization(
        wf, initial_guess(mol, nconf), gradient_generator(mol, wf, to_opt)
    )

    dfgrad = pd.DataFrame(dfgrad)
    mfen = mf.energy_tot()
    enfinal = dfgrad["energy"].values[-1]
    enfinal_err = dfgrad["energy_error"].values[-1]
    assert mfen > enfinal - enfinal_err


if __name__ == "__main__":
    test_linemin_three_body()
