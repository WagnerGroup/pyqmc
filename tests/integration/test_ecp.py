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

from pyscf import gto, scf
from pyqmc.method.mc import vmc, initial_guess
from pyqmc.wf.slater import Slater
from pyqmc.observables.accumulators import EnergyAccumulator
import numpy as np
import pandas as pd


def test_ecp():
    mol = gto.M(atom="C 0. 0. 0.", ecp="bfd", basis="bfd_vtz")
    mf = scf.RHF(mol).run()
    nconf = 5000
    wf = Slater(mol, mf)
    coords = initial_guess(mol, nconf)
    df, coords = vmc(
        wf, coords, nsteps=100, accumulators={"energy": EnergyAccumulator(mol)}
    )
    df = pd.DataFrame(df)
    warmup = 30
    print(
        "mean field",
        mf.energy_tot(),
        "vmc estimation",
        np.mean(df["energytotal"][warmup:]),
        np.std(df["energytotal"][warmup:]),
    )

    assert abs(mf.energy_tot() - np.mean(df["energytotal"][warmup:])) <= np.std(
        df["energytotal"][warmup:]
    )


if __name__ == "__main__":
    test_ecp()
