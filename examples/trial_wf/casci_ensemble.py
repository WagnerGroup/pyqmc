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
"""

from pyscf import gto, scf, mcscf
import h5py
import pyqmc.api as pyq
import pyqmc.accumulators
from rich import print
import os
import copy


def run_scf(atoms, scf_checkfile):
    mol = gto.M(atom=atoms, basis='ccecpccpvtz', ecp='ccecp', unit='bohr')
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
    nconfig=800
):
    """
    """
    from pyqmc.ensemble_optimization_wfbywf import optimize_ensemble, StochasticReconfigurationWfbyWf

    mol, mf, mc = pyq.recover_pyscf(
        scf_checkfile, ci_checkfile, cancel_outputs=False
    )

    mcs = [copy.copy(mc) for i in range(nstates)]
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]

    wfs = []
    energy = pyq.EnergyAccumulator(mol)
    sr_accumulator = []

    for i in range(nstates):
        wf, to_opt = pyq.generate_wf(
            mol, mf, mc=mcs[i], slater_kws=dict(optimize_determinants=True)
        )
        with h5py.File(jastrow_checkfile, "r") as f:
            for k in wf.parameters.keys():
                if 'wf2' in k:
                    wf.parameters[k] = f['wf'][k][()]
        wfs.append(wf)
        sr_accumulator.append([StochasticReconfigurationWfbyWf(energy, pyqmc.accumulators.LinearTransform(wf.parameters, to_opt))])
                
    configs = pyq.initial_guess(mol, nconfig)
    
    return optimize_ensemble(
        wfs,
        configs,
        sr_accumulator,
        hdf_file=hdf_file,
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
        pyq.OPTIMIZE(dft_checkfile=scf_checkfile, 
                     ci_checkfile=ci_checkfile,
                     output=jastrow_checkfile, 
                     verbose=True)
    ensemble_checkfile = f"{__file__}.ensemble.hdf5"
    run_ensemble(
        scf_checkfile, ci_checkfile, jastrow_checkfile, ensemble_checkfile, max_iterations=50
    )
