#!/usr/bin/env python

import numpy as np
from pyscf.pbc import gto, scf
import pyqmc
from pyqmc.slaterpbc import PySCFSlaterPBC, get_pyscf_supercell


def read_nk():
    import sys

    args = sys.argv[1:]
    if len(args) != 1:
        print("usage: python driver.py nk-per-direction")
        return -1
    else:
        nk = int(args[0])
        return nk


def run_scf(nk):
    cell = gto.Cell()
    cell.atom = """
    He 0.000000000000   0.000000000000   0.000000000000
    """
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.a = """
    5.61, 0.00, 0.00
    0.00, 5.61, 0.00
    0.00, 0.00, 5.61"""
    cell.unit = "B"
    cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts([nk, nk, nk])
    kmf = scf.KRHF(cell, exxdiv=None).density_fit()
    kmf.kpts = kpts
    ehf = kmf.kernel()

    print("EHF")
    print(ehf)
    return cell, kmf


if __name__ == "__main__":
    import pandas as pd

    nconfigs = 100
    for nk in [2]:
        # Run SCF
        cell, kmf = run_scf(nk)

        # Set up wf and configs
        S = np.eye(3) * nk
        supercell = get_pyscf_supercell(cell, S)
        wf = PySCFSlaterPBC(cell, kmf, S=S)
        configs = pyqmc.initial_guess(supercell, nconfigs)

        # Warm up VMC
        df, configs = pyqmc.vmc(wf, configs, nsteps=40, verbose=True)

        # Initialize energy accumulator (and Ewald)
        enacc = pyqmc.EnergyAccumulator(supercell)

        # Run VMC
        df, configs = pyqmc.vmc(
            wf, configs, nsteps=20, accumulators={"energy": enacc}, verbose=True
        )

        df = pd.DataFrame(df)
        print(df)
        df.to_csv("pbc_he_nk{0}.csv".format(nk))
