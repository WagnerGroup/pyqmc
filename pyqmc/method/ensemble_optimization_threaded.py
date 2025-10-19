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



from pyqmc.method.ensemble_optimization_wfbywf import StochasticReconfigurationWfbyWf, hdf_save, set_wf_params, renormalize
import numpy as np
import copy
import pyqmc
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import pyqmc.gpu as gpu
import os

def round_to_fixed_sum(x: np.ndarray, target_sum: int) -> np.ndarray:
    """
    Approximate an array of floats with integers such that:
      - The integer array sums to `target_sum`
      - The result is as close as possible to the original array

    Parameters
    ----------
    x : np.ndarray
        Input array of floats.
    target_sum : int
        Desired sum of the integer approximation.

    Returns
    -------
    np.ndarray
        Integer array with the same shape as x.
    """
    # Floor all elements first
    x = np.asarray(x)
    x = x*target_sum/np.sum(x)
    y = np.floor(x).astype(int)

    # Compute how many units we still need to add
    diff = target_sum - np.sum(y)

    if diff < 0 or diff > len(x):
        raise ValueError("Target sum is not achievable with integer rounding.")

    # Compute fractional parts
    frac = x - y

    # Get indices of the largest fractional parts
    idx = np.argsort(frac)[::-1]
    y[idx[:diff]] += 1

    return y


def evaluate_gradients_threaded(
    wfs,
    configs_ensemble,
    updater,
    client=None,
    npartitions=1,
    vmc_kwargs=None,
    overlap_kwargs=None,
    verbose=True,
    overlap_thread_weight=None,
):
    """Evaluate parameter gradients for each state with threader
    It runs Monte Carlo evaluations for all the states asynchronously and gathers the results as they complete
    Each state's vmc evaluations get parallelized using an even share of npartitions

    :parameter list wfs: list of optimized wave functions
    :parameter list configs_ensemble: nested list of initial configurations indexed by state then by sub-iteration then by thread
    :parameter list updater: nested list of StochasticReconfigurationWfbyWf accumulators indexed by state then by sub-iteration
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :parameter dict vmc_kwargs: a dictionary of options for the vmc method
    :parameter dict overlap_kwargs: a dictionary of options for the sample_overlap method

    :return dict data_sample1_ensemble: nested list of vmc outputs indexed by state then by sub-iteration
    :return dict data_weighted_ensemble: nested list of weighted sample_overlap outputs indexed by state then by sub-iteration
    :return dict data_unweighted_ensemble: nested list of unweighted sample_overlap outputs indexed by state then by sub-iteration
    :return list configs_ensemble: nested list of updated configurations indexed by state then by sub-iteration then by thread
    """
    if vmc_kwargs is None:
        vmc_kwargs = {}
    if overlap_kwargs is None:
        overlap_kwargs = {}
    nwf = len(wfs)
    nthreads = 2 * sum([len(updater[wfi]) for wfi in range(nwf)])
    data_sample1_ensemble = [[0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
    data_weighted_ensemble = [[0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
    data_unweighted_ensemble = [[0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
    energy_workers = {}
    overlap_workers = {}
    if nthreads == 0:
        return data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble

    #if npartitions // nthreads == 0:
    #    print("Warning: there are more threads than worker processes", flush=True)
    #    print("Each thread will be given 1 worker process", flush=True)
    #npartitions_per_thread = max(1, int(npartitions // nthreads))

    weights = np.zeros(nthreads)
    threadcount=0
    # Energy
    for transform in updater:
        for _ in transform:
            weights[threadcount] = 1.0
            threadcount += 1
    # overlap: the estimate is that the energy costs about the same
    # as sampling one wave function. So we add nwf/2.0 to the weight
    # because we are sampling wfi+1 wave functions
    for wfi, transform in enumerate(updater):
        if overlap_thread_weight is None:
            for trans in transform:
                weights[threadcount] = (1+wfi)/2.0
                threadcount += 1
        else:
            for trans in transform:
                weights[threadcount] = overlap_thread_weight[wfi]
                threadcount += 1

    npartitions_by_thread = round_to_fixed_sum(weights, npartitions)

    print("nthreads", nthreads, "npartitions", npartitions_by_thread, flush=True)
    start_time = time.perf_counter()
    threadcount = 0
    with ThreadPoolExecutor(max_workers=nthreads) as threader:
        for wfi, wf in enumerate(wfs):
            transform_list = updater[wfi]
            for sub_iteration in range(len(transform_list)):
                transform = transform_list[sub_iteration]
                energy_workers_thread = threader.submit(
                    pyqmc.method.mc.vmc,
                    wf,
                    configs_ensemble[wfi][sub_iteration][0],
                    accumulators={"": transform.onewf()},
                    verbose=True,
                    client=client,
                    npartitions=npartitions_by_thread[threadcount],
                    **vmc_kwargs,
                )
                energy_workers[energy_workers_thread] = (wfi, sub_iteration)
                threadcount += 1
        for wfi, wf in enumerate(wfs):
            print("wfi", wfi, threadcount, npartitions_by_thread[threadcount], flush=True)
            transform_list = updater[wfi]
            for sub_iteration in range(len(transform_list)):
                overlap_workers_thread = threader.submit(
                    pyqmc.method.sample_many.sample_overlap,
                    wfs[0 : wfi + 1],
                    configs_ensemble[wfi][sub_iteration][1],
                    transform.allwfs(),
                    client=client,
                    npartitions=npartitions_by_thread[threadcount],
                    **overlap_kwargs,
                )
                overlap_workers[overlap_workers_thread] = (wfi, sub_iteration)
                threadcount += 1
        all_workers = {**energy_workers, **overlap_workers}

        middle_time = time.perf_counter()
        times = []
        for future in as_completed(all_workers):
            wfi, sub_iteration = all_workers[future]
            if future in energy_workers:
                times.append( {'time': time.perf_counter() - middle_time, 'type':'energy', 'wfi':wfi, 'sub_iteration':sub_iteration} )
                data_sample1_ensemble[wfi][sub_iteration], configs_ensemble[wfi][sub_iteration][0] = future.result()
            elif future in overlap_workers: #overlap worker
                times.append( {'time': time.perf_counter() - middle_time, 'type':'overlap', 'wfi':wfi, 'sub_iteration':sub_iteration} )
                (
                    data_weighted_ensemble[wfi][sub_iteration],
                    data_unweighted_ensemble[wfi][sub_iteration],
                    configs_ensemble[wfi][sub_iteration][1],
                ) = future.result()
            else:
                raise ValueError("Received unknown future")
    if verbose:
        print("time to submit", middle_time-start_time, flush=True)
        print(pd.DataFrame(times))
    return data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble


def optimize_ensemble(
    wfs,
    configs,
    updater,
    hdf_file,
    client,
    tau=1,
    max_iterations=100,
    overlap_penalty=None,
    npartitions=None,
    verbose=True,
    overlap_thread_weight = None,
    warmup_kwargs=None,
    vmc_kwargs=None,
    overlap_kwargs=None,
):
    """Optimize a set of wave functions using ensemble VMC.


    Returns
    -------

    wfs : list of optimized wave functions
    """

    if warmup_kwargs is None or len(warmup_kwargs) == 0:
        warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if vmc_kwargs is None or len(vmc_kwargs) == 0:
        vmc_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if overlap_kwargs is None or len(overlap_kwargs) == 0:
        overlap_kwargs = dict(nblocks=10, nsteps=10)
    nwf = len(wfs)
    if overlap_penalty is None:
        overlap_penalty = np.ones((nwf, nwf)) * 0.5

    iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                for wfi, wf in enumerate(wfs):
                    grp = hdf[f"wf/{wfi}"]
                    for k in grp.keys():
                        wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf.keys():
                iteration_offset = np.max(hdf["iteration"][...]) + 1
            configs.load_hdf(hdf)
    else:
        _, configs = pyqmc.method.mc.vmc(
            wfs[0],
            configs,
            verbose=True,
            client=client,
            npartitions=npartitions,
            **warmup_kwargs,
        )

    configs_ensemble = [
        [[copy.deepcopy(configs) for _ in range(2)] for _ in range(len(updater[wfi]))]
        for wfi in range(nwf)
    ]
    for i in range(iteration_offset, max_iterations):
        _, data_unweighted, configs = pyqmc.method.sample_many.sample_overlap(
            wfs,
            configs_ensemble[0][0][0],
            None,
            client=client,
            npartitions=npartitions,
            **overlap_kwargs,
        )
        norm = np.mean(data_unweighted["overlap"], axis=0)
        if verbose:
            print("Normalization step", norm.diagonal())
        renormalize(wfs, norm.diagonal(), pivot=0)
        data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble = evaluate_gradients_threaded(
            wfs,
            configs_ensemble,
            updater,
            client=client,
            npartitions=npartitions,
            vmc_kwargs=vmc_kwargs,
            overlap_kwargs=overlap_kwargs,
        )
        for wfi, wf in enumerate(wfs):
            transform_list = updater[wfi]
            for sub_iteration, transform in enumerate(transform_list):
                transform = transform_list[sub_iteration]
                avg, error = transform.block_average(
                    data_sample1_ensemble[wfi][sub_iteration],
                    data_weighted_ensemble[wfi][sub_iteration],
                    data_unweighted_ensemble[wfi][sub_iteration]["overlap"],
                )
                if verbose:
                    print("Iteration", i, "wf ", wfi, " sub iteration ", sub_iteration, "Energy", avg["total"], "Overlap", avg["overlap"][wfi, :])
                dp, report = transform.delta_p([tau], avg, overlap_penalty, verbose=True)
                x = transform.transform.serialize_parameters(wf.parameters)
                x = x + dp[0]
                set_wf_params(wf, x, transform)

                save_data = {
                    f"energy{wfi}": avg["total"],
                    f"energy_error{wfi}": error["total"],
                    f"overlap{wfi}": avg["overlap"],
                    "iteration": i,
                    "wavefunction": wfi,
                    "sub_iteration": sub_iteration,
                }
                hdf_save(hdf_file, save_data, {"tau": tau}, wfs, configs_ensemble[wfi][sub_iteration][0])

    return wfs
