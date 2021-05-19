import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pyscf import lib, gto, scf, mcscf
import pyqmc.testwf as testwf
import pyqmc.api as pyq
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.slater import Slater


def test_casci_energy(H2_ccecp_casci_s0):
    """
    Checks that VMC energy matches energy calculated in PySCF
    """
    nsteps = 200
    warmup = 10

    mol, mf, mc = H2_ccecp_casci_s0
    wf = Slater(mol, mf, mc)
    nconf = 1000
    coords = pyq.initial_guess(mol, nconf)
    df, coords = pyq.vmc(
        wf, coords, nsteps=nsteps, accumulators={"energy": EnergyAccumulator(mol)}
    )

    df = pd.DataFrame(df)
    df = pyq.avg_reblock(df["energytotal"][warmup:], 20)
    en = df.mean()
    err = df.sem()
    assert en - mc.e_tot < 5 * err

