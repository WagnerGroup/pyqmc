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

import numpy as np
import pandas as pd
import pyqmc.api as pyq
from pyqmc.slater import Slater
from pyscf.pbc import gto, scf
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.scf.addons import remove_linear_dep_
import time
import uuid


def test_energy_li(li_cubic_ccecp):
    cell, mf = li_cubic_ccecp
    runtest(cell, mf, kind=0)


def runtest(mol, mf, kind=0):
    kpt = mf.kpts[kind]
    dm = mf.make_rdm1()
    print("original dm shape", dm.shape)
    if len(dm.shape) == 4:
        dm = np.sum(dm, axis=0)
    dm = dm[kind]

    #####################################
    ## evaluate KE in PySCF
    #####################################
    ke_mat = mol.pbc_intor("int1e_kin", hermi=1, kpts=np.array(kpt))
    print("ke_mat", ke_mat.shape)
    print("dm", dm.shape)
    pyscfke = np.real(np.einsum("ij,ji->", ke_mat, dm))
    print("PySCF kinetic energy: {0}".format(pyscfke))

    #####################################
    ## evaluate KE integral with VMC
    #####################################
    wf = Slater(mol, mf)
    coords = pyq.initial_guess(mol, 1200, 0.7)
    warmup = 10
    start = time.time()
    df, coords = pyq.vmc(
        wf,
        coords,
        nsteps=100 + warmup,
        tstep=1,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
        verbose=False,
        hdf_file=str(uuid.uuid4()),
    )
    print("VMC time", time.time() - start)

    df = pd.DataFrame(df)
    dfke = pyq.avg_reblock(df["energyke"][warmup:], 10)
    vmcke, err = dfke.mean(), dfke.sem()
    print("VMC kinetic energy: {0} +- {1}".format(vmcke, err), df["energyke"])

    assert (
        np.abs(vmcke - pyscfke) < 5 * err
    ), "energy diff not within 5 sigma ({0:.6f}): energies \n{1} \n{2}".format(
        5 * err, vmcke, pyscfke
    )
