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

"""
Here we demonstrate how to create and optimize an ensemble of wave functions,
using CASCI to generate the initial wave functions.

Pass a client to run the states in parallel; without one the sampling runs
serially, which is fine for a quick check but slow for real work. The optimizer
is selected with `method`: "sr" builds the (nparameters, nparameters) S matrix,
"minsr" solves the same equations in sample space and never builds it, which is
what you want when there are more parameters than samples.
"""

from pyscf import gto, scf, mcscf
import h5py
import pyqmc.api as pyq
import pyqmc.observables.accumulators
from pyqmc.method.ensemble_optimization import optimize_ensemble
from rich import print
import os
import copy
from concurrent.futures import ProcessPoolExecutor


def run_scf(atoms, scf_checkfile):
    mol = gto.M(atom=atoms, basis="ccecpccpvtz", ecp="ccecp", unit="bohr")
    mf = scf.RHF(mol)
    mf.chkfile = scf_checkfile
    dm = mf.init_guess_by_atom()
    mf.kernel(dm)


def run_casci(scf_checkfile, ci_checkfile):
    cell, mf = pyq.recover_pyscf(scf_checkfile, cancel_outputs=False)
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 4
    mc.kernel()

    print(mc.__dict__.keys())
    with h5py.File(ci_checkfile, "a") as f:
        f.create_group("ci")
        f["ci/ncas"] = mc.ncas
        f["ci/nelecas"] = list(mc.nelecas)
        f["ci/ci"] = mc.ci
        f["ci/mo_coeff"] = mc.mo_coeff
    return mc


def run_pyscf_h2(scf_checkfile, ci_checkfile):
    run_scf("H 0. 0. 0.0; H 0. 0. 1.4", scf_checkfile)
    run_casci(scf_checkfile, ci_checkfile)


def run_ensemble(
    scf_checkfile,
    ci_checkfile,
    jastrow_checkfile,
    hdf_file,
    max_iterations,
    client=None,
    npartitions=None,
    nstates=3,
    tau=0.1,
    nconfig=800,
    method="sr",
):
    """ """
    mol, mf, mc = pyq.recover_pyscf(scf_checkfile, ci_checkfile, cancel_outputs=False)

    mcs = [copy.copy(mc) for i in range(nstates)]
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]

    wfs = []
    transforms = []

    for i in range(nstates):
        wf, to_opt = pyq.generate_wf(
            mol, mf, mc=mcs[i], slater_kws=dict(optimize_determinants=True)
        )
        with h5py.File(jastrow_checkfile, "r") as f:
            for k in wf.parameters.keys():
                if "wf2" in k:
                    wf.parameters[k] = f["wf"][k][()]
        wfs.append(wf)
        transforms.append(
            pyqmc.observables.accumulators.LinearTransform(wf.parameters, to_opt)
        )

    configs = pyq.initial_guess(mol, nconfig)

    # minSR solves in sample space, and the kernel it solves is square in the
    # number of samples, nconfig * nblocks, so take a single block there. SR
    # averages over blocks instead, and its cost does not depend on the count.
    vmc_kwargs = {"nblocks": 1} if method == "minsr" else None

    return optimize_ensemble(
        wfs,
        configs,
        transforms,
        hdf_file=hdf_file,
        enacc=pyq.EnergyAccumulator(mol),
        method=method,
        vmc_kwargs=vmc_kwargs,
        max_iterations=max_iterations,
        client=client,
        npartitions=npartitions,
        verbose=True,
        tau=tau,
    )


if __name__ == "__main__":
    scf_checkfile = f"{__file__}.scf.hdf5"
    ci_checkfile = f"{__file__}.ci.hdf5"
    if not os.path.isfile(scf_checkfile) or not os.path.isfile(ci_checkfile):
        run_pyscf_h2(scf_checkfile, ci_checkfile)

    jastrow_checkfile = f"{__file__}.jastrow.hdf5"
    if not os.path.isfile(jastrow_checkfile):
        pyq.OPTIMIZE(
            dft_checkfile=scf_checkfile,
            ci_checkfile=ci_checkfile,
            output=jastrow_checkfile,
            verbose=True,
        )
    ensemble_checkfile = f"{__file__}.ensemble.hdf5"
    # Drop the client to run this serially, e.g. while trying something out.
    ncores = 9
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        run_ensemble(
            scf_checkfile,
            ci_checkfile,
            jastrow_checkfile,
            ensemble_checkfile,
            max_iterations=50,
            client=executor,
            npartitions=ncores,
            method="sr",
        )
