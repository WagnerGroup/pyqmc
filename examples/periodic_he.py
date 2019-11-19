#!/usr/bin/env python

import numpy as np
from pyscf.pbc import gto, scf
import pyqmc
from pyqmc import JastrowSpin, MultiplyWF
from pyqmc.slaterpbc import PySCFSlaterPBC, get_supercell
from pyqmc.func3d import PolyPadeFunction, CutoffCuspFunction
from pyqmc.linemin import line_minimization


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

    nconfig = 100
    for nk in [2]:
        # Run SCF
        cell, kmf = run_scf(nk)

        # Set up wf and configs
        S = np.eye(3) * nk
        supercell = get_supercell(cell, S)
        slater = PySCFSlaterPBC(supercell, kmf)
        configs = pyqmc.initial_guess(supercell, nconfig)

        # Jastrow
        abasis = [PolyPadeFunction(beta=0.8 * 2 ** i, rcut=7.5) for i in range(3)]
        bbasis = [CutoffCuspFunction(gamma=8, rcut=7.5)]
        bbasis += [PolyPadeFunction(beta=0.8 * 2 ** i, rcut=7.5) for i in range(3)]
        jastrow = JastrowSpin(supercell, a_basis=abasis, b_basis=bbasis)
        jastrow.parameters["bcoeff"][0, [0, 1, 2]] = np.array([-0.25, -0.50, -0.25])
        freeze = {}
        freeze["wf2acoeff"] = np.zeros(jastrow.parameters["acoeff"].shape).astype(bool)
        freeze["wf2bcoeff"] = np.zeros(jastrow.parameters["bcoeff"].shape).astype(bool)
        freeze["wf2bcoeff"][0, [0, 1, 2]] = True  # Cusp conditions
        to_opt = ["wf2acoeff", "wf2bcoeff"]

        # Multiply
        wf = MultiplyWF(slater, jastrow)
        configs = pyqmc.initial_guess(supercell, nconfig)

        # Warm up VMC
        df, configs = pyqmc.vmc(wf, configs, nsteps=5, verbose=True)

        # Initialize energy accumulator (and Ewald)
        pgrad = pyqmc.gradient_generator(supercell, wf, to_opt=to_opt, freeze=freeze)

        # Optimize jastrow
        hdf_file = "linemin_nk{0}.hdf".format(nk)
        wf, lm_df = line_minimization(
            wf, configs, pgrad, hdf_file=hdf_file, verbose=True
        )
        jastrow = wf.wf2

        # Run VMC
        df, configs = pyqmc.vmc(
            wf, configs, nsteps=20, accumulators={"energy": pgrad.enacc}, verbose=True
        )

        df = pd.DataFrame(df)
        print(df)
        df.to_csv("pbc_he_nk{0}.csv".format(nk))
