#!/usr/bin/env python

import numpy as np
from pyscf.pbc import gto, scf
import pyqmc.api as pyq
from pyqmc.supercell import get_supercell


def run_scf(nk):
    cell = gto.Cell()
    cell.atom = """
    He 0.000000000000   0.000000000000   0.000000000000
    """
    cell.basis = "ccecp-ccpvdz"
    cell.a = """
    5.61, 0.00, 0.00
    0.00, 5.61, 0.00
    0.00, 0.00, 5.61"""
    cell.unit = "B"
    cell.verbose = 5
    cell.build()

    kmf = scf.KRHF(cell, exxdiv=None).density_fit()
    kmf.kpts = cell.make_kpts([nk, nk, nk])
    ehf = kmf.kernel()
    print("EHF", ehf)
    return cell, kmf


if __name__ == "__main__":
    # Run SCF
    cell, kmf = run_scf(nk=2)

    # Set up wf and configs
    nconfig = 100
    S = np.eye(3) * 2  # 2x2x2 supercell
    supercell = get_supercell(cell, S)
    wf, to_opt = pyq.generate_wf(supercell, kmf)
    configs = pyq.initial_guess(supercell, nconfig)

    # Initialize energy accumulator (and Ewald)
    pgrad = pyq.gradient_generator(supercell, wf, to_opt=to_opt)

    # Optimize jastrow
    wf, lm_df = pyq.line_minimization(
        wf, configs, pgrad, hdf_file="pbc_he_linemin.hdf", verbose=True
    )

    # Run VMC
    df, configs = pyq.vmc(
        wf,
        configs,
        nblocks=100,
        accumulators={"energy": pgrad.enacc},
        hdf_file="pbc_he_vmc.hdf",
        verbose=True,
    )

    # Run DMC
    pyq.rundmc(
        wf,
        configs,
        nblocks=1000,
        accumulators={"energy": pgrad.enacc},
        hdf_file="pbc_he_dmc.hdf",
        verbose=True,
    )
