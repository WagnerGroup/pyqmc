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
import concurrent.futures
import pyqmc.api as pyq
import pyqmc.observables.accumulators
from pyqmc.method.ensemble_optimization_wfbywf import (
    StochasticReconfigurationWfbyWf,
    evaluate_gradients,
    evaluate_gradients_threaded,
)


def test_threading(H2_casci):
    """We test the evaluate_gradients_threaded function in ensemble_optimization_wfbywf
    evaluate_gradients_threaded runs vmc and sample_overlap for multiple states asynchronously using threading
    Using two CASCI wave functions for H2 and two sub iterations per wave function, we test whether
    (1) evaluate_gradients and evaluate_gradients_threaded return the same ground and excited state energies within error
    (2) evaluate_gradients_threaded returns an overlap matrix with unit determinant
    (3) the functions support cases of multiple sub iterations
    """
    nwf = 2
    nconfig = 1000
    npartitions = 8
    max_sub_iterations = 2
    mol, mf, mc = H2_casci
    mcs = [copy.copy(mc) for _ in range(nwf)]
    energy = pyq.EnergyAccumulator(mol)
    wfs = []
    updater = []
    for i in range(nwf):
        mcs[i].ci = mc.ci[i]
        wf, to_opt = pyq.generate_slater(
            mol,
            mf,
            mc=mcs[i],
            optimize_determinants=True,
        )
        wfs.append(wf)
        transform_list = [] 
        half_batch_index = int(np.floor(0.5 * len(to_opt["det_coeff"])))
        for sub_iteration in range(max_sub_iterations):
            to_opt_new = copy.deepcopy(to_opt)
            to_opt_new["det_coeff"][sub_iteration * half_batch_index : (sub_iteration + 1) * half_batch_index] = False
            transform_list.append(
                StochasticReconfigurationWfbyWf(
                    energy,
                    pyqmc.observables.accumulators.LinearTransform(wf.parameters, to_opt_new),
                )
            )
        updater.append(transform_list)
    configs = pyq.initial_guess(mol, nconfig)
    configs_ensemble = [[copy.deepcopy(configs) for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=npartitions) as client:
            mc_data = choose_gradient_function[str(use_threader)](
                wfs,
                configs_ensemble,
                updater,
                wf_start=0,
                sub_iteration_offset=0, 
                client=client,
                npartitions=npartitions,
            )
        (
            data_sample1_ensemble,
            data_weighted_ensemble,
            data_unweighted_ensemble,
            configs_ensemble,
        ) = mc_data
        avg0, error0 = updater[0][0].block_average(
            data_sample1_ensemble[0][0],
            data_weighted_ensemble[0][0],
            data_unweighted_ensemble[0][0]["overlap"],
        )
        energy0[f"threader_{use_threader}"] = avg0["total"]
        energy_error0[f"threader_{use_threader}"] = error0["total"]
        avg1, error1 = updater[1][0].block_average(
            data_sample1_ensemble[1][0],
            data_weighted_ensemble[1][0],
            data_unweighted_ensemble[1][0]["overlap"],
        )
        energy1[f"threader_{use_threader}"] = avg1["total"]
        energy_error1[f"threader_{use_threader}"] = error1["total"]
        overlap_determinant[f"threader_{use_threader}"] = np.linalg.det(avg1["overlap"])

    stderr_energy0 = np.sqrt(energy_error0["threader_False"] ** 2 + energy_error0["threader_True"] ** 2)
    chi2_energy0 = np.abs(energy0["threader_False"] - energy0["threader_True"]) / stderr_energy0
    assert chi2_energy0 < 5, "Ground state energy computed with threader deviates too much from that without threader"
    stderr_energy1 = np.sqrt(energy_error1["threader_False"] ** 2 + energy_error1["threader_True"] ** 2)
    chi2_energy1 = np.abs(energy1["threader_False"] - energy1["threader_True"]) / stderr_energy1
    assert chi2_energy1 < 5, "Excited state energy computed with threader deviates too much from that without threader"
    assert np.abs(overlap_determinant["threader_True"] - 1) < 0.01, "CASCI wf overlap determinant deviates too much from 1"