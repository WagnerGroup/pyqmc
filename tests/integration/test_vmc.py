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
import pandas as pd
from pyqmc.mc import vmc, initial_guess
from pyscf import gto, scf
from pyqmc.reblock import reblock
from pyqmc.slater import Slater
from pyqmc.accumulators import EnergyAccumulator
import pytest


@pytest.mark.slow
def test_vmc(C2_ccecp_rhf):
    """
    Test that a VMC calculation of a Slater determinant matches Hartree-Fock within error bars.
    """
    mol, mf = C2_ccecp_rhf
    nconf = 500
    nsteps = 300
    warmup = 30

    wf = Slater(mol, mf)
    coords = initial_guess(mol, nconf)
    df, coords = vmc(
        wf,
        coords,
        nblocks=int(nsteps / 30),
        nsteps_per_block=30,
        accumulators={"energy": EnergyAccumulator(mol)},
    )

    df = pd.DataFrame(df)["energytotal"][int(warmup / 30) :]
    en = df.mean()
    err = df.sem()
    assert en - mf.energy_tot() < 5 * err, "pyscf {0}, vmc {1}, err {2}".format(
        mf.energy_tot(), en, err
    )


def test_accumulator(C2_ccecp_rhf):
    """Tests that the accumulator gets inserted into the data output correctly."""
    mol, mf = C2_ccecp_rhf
    nconf = 500
    wf = Slater(mol, mf)
    coords = initial_guess(mol, nconf)

    df, coords = vmc(
        wf, coords, nsteps=30, accumulators={"energy": EnergyAccumulator(mol)}
    )
    df = pd.DataFrame(df)
    eaccum = EnergyAccumulator(mol)
    eaccum_energy = eaccum(coords, wf)
    df = pd.DataFrame(df)
    print(df["energyke"][29] == np.average(eaccum_energy["ke"]))

    assert df["energyke"][29] == np.average(eaccum_energy["ke"])


if __name__ == "__main__":
    test_vmc()
    test_accumulator()
