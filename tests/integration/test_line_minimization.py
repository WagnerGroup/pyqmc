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
import pandas as pd
from pyqmc.api import generate_wf, line_minimization, initial_guess, gradient_generator
import pytest


@pytest.mark.slow
def test_linemin(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function and check that it's better than Hartree-Fock"""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    wf, to_opt = generate_wf(mol, mf)
    nconf = 1000
    wf, dfgrad = line_minimization(
        wf, initial_guess(mol, nconf), gradient_generator(mol, wf, to_opt), verbose=True
    )

    dfgrad = pd.DataFrame(dfgrad)
    mfen = mf.energy_tot()
    enfinal = dfgrad["energy"].values[-1]
    enfinal_err = dfgrad["energy_error"].values[-1]
    assert mfen > enfinal


if __name__ == "__main__":
    test_linemin()
