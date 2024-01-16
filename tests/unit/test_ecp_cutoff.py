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

import pandas as pd
from pyscf import gto, scf, mcscf
from pyqmc.mc import initial_guess
from pyqmc.multiplywf import MultiplyWF
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.slater import Slater
from pyqmc.wftools import generate_jastrow
import numpy as np
import time


def test_ecp_sj(C2_ccecp_rhf, nconf=10000):
    """test whether the cutoff saves us time without changing the energy too much.
    Because it's a stochastic evaluation, random choices can make a big difference, so we only require 10% agreement between these two.
    """
    mol, mf = C2_ccecp_rhf
    THRESHOLDS = [1e15, 10]

    np.random.seed(1234)
    coords = initial_guess(mol, nconf)
    wf = MultiplyWF(Slater(mol, mf), generate_jastrow(mol)[0])
    wf.recompute(coords)
    times = []
    energies = []
    for threshold in THRESHOLDS:
        np.random.seed(1234)
        eacc = EnergyAccumulator(mol, threshold)
        start = time.time()
        energy = eacc(coords, wf)
        end = time.time()
        times.append(end - start)
        energies.append(np.mean(energy["total"]))
    # print(times, energies)
    assert times[1] < times[0]
    assert (energies[1] - energies[0]) / energies[0] < 0.1


if __name__ == "__main__":
    mol = gto.M(
        atom="""C 0 0 0 
                C 1 0 0  """,
        ecp="ccecp",
        basis="ccecpccpvdz",
    )
    mf = scf.RHF(mol).run()

    test_ecp_sj((mol, mf))
