#!/usr/bin/env python

import numpy as np
from pyscf.pbc import gto, scf
import pyqmc.api as pyq
from pyqmc.supercell import get_supercell
from pyqmc.geminal_jastrow import GeminalJastrow
import copy


def make_cell(basis="ccecp-ccpvdz"):
    cell = gto.Cell()
    cell.atom = """
    He 0.000000000000   0.000000000000   0.000000000000
    """
    cell.basis = basis
    cell.a = """
    5.61, 0.00, 0.00
    0.00, 5.61, 0.00
    0.00, 0.00, 5.61"""
    cell.unit = "B"
    cell.verbose = 0
    cell.build()
    return cell


def run_scf(nk):
    cell = make_cell()
    kmf = scf.KRHF(cell, exxdiv=None).density_fit()
    kmf.kpts = cell.make_kpts([nk, nk, nk])
    kmf.chkfile = "he.chkfile"
    ehf = kmf.kernel()
    print("EHF", ehf)
    return cell, kmf


def run_qmc(cell, kmf):
    # Set up wf and configs
    nconfig = 100
    S = np.eye(3) * 2  # 2x2x2 supercell
    supercell = get_supercell(cell, S)
    to_opts = [None] * 3
    slater, to_opts[0] = pyq.generate_slater(supercell, kmf)
    cusp, to_opts[1] = pyq.generate_jastrow(supercell, na=0, nb=0)

    geminal = GeminalJastrow(supercell)
    to_opts[2] = {"gcoeff":np.ones(geminal.parameters["gcoeff"].shape).astype(bool)}
    to_opt = {}
    for i, t_o in enumerate(to_opts):
        to_opt.update({f"wf{i+1}" + k: v for k, v in t_o.items()})
    print("to_opt", to_opt["wf3gcoeff"].shape)

    wf = pyq.MultiplyWF(slater, cusp, geminal)
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


if __name__ == "__main__":
    # Run SCF
    #cell, kmf = run_scf(nk=2)
    cell, kmf = pyq.recover_pyscf("he.chkfile")

    run_qmc(cell, kmf)
