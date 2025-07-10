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

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import copy
import h5py
import time
import concurrent.futures
from pyscf import gto, scf, mcscf
import pyqmc.api as pyq
import pyqmc.observables.accumulators
from pyqmc.method.ensemble_optimization_wfbywf import (
    StochasticReconfigurationWfbyWf,
    evaluate_gradients,
    evaluate_gradients_threaded,
)


def run_scf(atoms, scf_checkfile):
    mol = gto.M(atom=atoms, basis="ccecpccpvtz", ecp="ccecp", unit="bohr")
    mf = scf.RHF(mol)
    mf.chkfile = scf_checkfile
    dm = mf.init_guess_by_atom()
    mf.kernel(dm)


def run_casci(scf_checkfile, ci_checkfile):
    _, mf = pyq.recover_pyscf(scf_checkfile, cancel_outputs=False)
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 2
    mc.kernel()
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


def generate_updater(mol, wfs, to_opts):
    updater = []
    energy = pyq.EnergyAccumulator(mol)
    for i in range(len(wfs)):
        updater.append(
            [
                StochasticReconfigurationWfbyWf(
                    energy,
                    pyqmc.observables.accumulators.LinearTransform(
                        wfs[i].parameters, to_opts[i]
                    ),
                )
            ]
        )
    return updater


def test_threading(
    scf_checkfile,
    ci_checkfile,
    client=None,
    npartitions=1,
):
    """We test the evaluate_gradients_threaded function in the ensemble_optimization_wfbywf module
    evaluate_gradients_threaded runs vmc and sample_overlap for multiple states asynchronously using threading
    Using two CASCI wave functions for H2, we test whether
    (1) evaluate_gradients_threaded takes less runtime than evaluate_gradients for the same nconfig and npartitions
    (2) evaluate_gradients and evaluate_gradients_threaded return the same ground and excited state energies within error
    (3) evaluate_gradients_threaded returns an overlap matrix with unit determinant

    :parameter str scf_checkfile: hdf5 file containing the mean-field calculation data for H2
    :parameter str ci_checkfile: hdf5 file containing the CASCI calculation data for H2
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    """
    nwf = 2
    nconfig = 1000
    mol, mf, mc = pyq.recover_pyscf(scf_checkfile, ci_checkfile, cancel_outputs=False)
    mcs = [copy.copy(mc) for _ in range(nwf)]
    wfs = []
    to_opts = []
    for i in range(nwf):
        mcs[i].ci = mc.ci[i]
        wf, to_opt = pyq.generate_slater(
            mol,
            mf,
            mc=mcs[i],
            optimize_determinants=True,
        )
        wfs.append(wf)
        to_opts.append(to_opt)
    updater = generate_updater(mol, wfs, to_opts)
    configs = pyq.initial_guess(mol, nconfig)
    configs_ensemble = [copy.deepcopy(configs) for _ in range(nwf)]
    runtime = {}
    energy0 = {}
    energy_error0 = {}
    energy1 = {}
    energy_error1 = {}
    overlap_determinant = {}
    choose_gradient_function = {
        "False": evaluate_gradients,
        "True": evaluate_gradients_threaded,
    }
    for use_threader in [False, True]:
        start_time = time.perf_counter()
        mc_data = choose_gradient_function[str(use_threader)](
            wfs,
            configs_ensemble,
            updater,
            range(nwf),
            sub_iteration=0,
            client=client,
            npartitions=npartitions,
        )
        end_time = time.perf_counter()
        runtime[f"threader_{use_threader}"] = end_time - start_time
        (
            data_sample1_ensemble,
            data_weighted_ensemble,
            data_unweighted_ensemble,
            configs_ensemble,
        ) = mc_data
        avg0, error0 = updater[0][0].block_average(
            data_sample1_ensemble[0],
            data_weighted_ensemble[0],
            data_unweighted_ensemble[0]["overlap"],
        )
        energy0[f"threader_{use_threader}"] = avg0["total"]
        energy_error0[f"threader_{use_threader}"] = error0["total"]
        avg1, error1 = updater[1][0].block_average(
            data_sample1_ensemble[1],
            data_weighted_ensemble[1],
            data_unweighted_ensemble[1]["overlap"],
        )
        energy1[f"threader_{use_threader}"] = avg1["total"]
        energy_error1[f"threader_{use_threader}"] = error1["total"]
        overlap_determinant[f"threader_{use_threader}"] = np.linalg.det(avg1["overlap"])

    print("Evaluation time without threader: ", runtime["threader_False"], " s")
    print("Evaluation time with threader: ", runtime["threader_True"], " s")
    assert runtime["threader_False"] > runtime["threader_True"], "Threaded run doesn't save time relative to no threader"
    stderr_energy0 = np.sqrt(energy_error0["threader_False"] ** 2 + energy_error0["threader_True"] ** 2)
    chi2_energy0 = (np.abs(energy0["threader_False"] - energy0["threader_True"]) / stderr_energy0)
    assert chi2_energy0 < 5, "Ground state energy computed with threading deviates too much from that without threader"
    stderr_energy1 = np.sqrt(energy_error1["threader_False"] ** 2 + energy_error1["threader_True"] ** 2)
    chi2_energy1 = (np.abs(energy1["threader_False"] - energy1["threader_True"]) / stderr_energy1)
    assert chi2_energy1 < 5, "Excited state energy computed with threading deviates too much from that without threader"
    assert np.abs(overlap_determinant["threader_True"] - 1) < 0.01, "CASCI wf overlap determinant deviates too much from 1"


if __name__ == "__main__":
    scf_checkfile = f"{__file__}.scf.hdf5"
    ci_checkfile = f"{__file__}.ci.hdf5"
    if not os.path.isfile(scf_checkfile) or not os.path.isfile(ci_checkfile):
        run_pyscf_h2(scf_checkfile, ci_checkfile)
    npartitions = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=npartitions) as client:
        test_threading(
            scf_checkfile,
            ci_checkfile,
            client=client,
            npartitions=npartitions,
        )
    os.remove(scf_checkfile)
    os.remove(ci_checkfile)
