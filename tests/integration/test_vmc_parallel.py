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
import pyqmc.recipes
import concurrent.futures
import os
import pytest

ncore = 2
nconfig = ncore * 400


def run_scf(chkfile):
    from pyscf import gto, scf

    mol = gto.M(
        atom="H 0 0 0; H 0 0. 1.4", basis="ccecpccpvdz", ecp="ccecp", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()


@pytest.mark.slow
def test_parallel():
    run_scf("h2.hdf5")
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
        pyqmc.recipes.OPTIMIZE(
            "h2.hdf5", "linemin.hdf5", nconfig=50, client=client, npartitions=ncore
        )
    assert os.path.isfile("linemin.hdf5")
    os.remove("h2.hdf5")
    os.remove("linemin.hdf5")
