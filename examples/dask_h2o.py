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

from dask.distributed import Client, LocalCluster
import pyqmc

ncore = 2
nconfig = ncore * 400


def run_scf():
    from pyscf import gto, scf

    mol = gto.M(
        atom="O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587", basis="bfd_vtz", ecp="bfd"
    )
    mf = scf.RHF(mol).run()
    return mol, mf


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    client = Client(cluster)
    mol, mf = run_scf()
    from pyqmc import vmc, line_minimization, rundmc

    wf, to_opt = pyqmc.default_sj(mol, mf)
    pgrad_acc = pyqmc.gradient_generator(mol, wf, to_opt)
    configs = pyqmc.initial_guess(mol, nconfig)
    line_minimization(
        wf,
        configs,
        pgrad_acc,
        hdf_file="h2o_opt.hdf",
        client=client,
        npartitions=ncore,
        verbose=True,
    )
    df, configs = vmc(
        wf,
        configs,
        hdf_file="h2o_vmc.hdf",
        accumulators={"energy": pgrad_acc.enacc},
        client=client,
        npartitions=ncore,
        verbose=True,
    )
    dfdmc, configs, weights = rundmc(
        wf,
        configs,
        hdf_file="h2o_dmc.hdf",
        nblocks=1000,
        accumulators={"energy": pgrad_acc.enacc},
        ekey=("energy", "total"),
        tstep=0.02,
        verbose=True,
        client=client,
        npartitions=ncore,
    )
