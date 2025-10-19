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


import pyqmc.method.sample_many
import numpy as np
import h5py
from pyqmc.method import hdftools
import pyqmc.gpu as gpu
import os
from pyqmc.observables.stochastic_reconfiguration import StochasticReconfiguration
import scipy.stats
import copy
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class StochasticReconfigurationWfbyWf:
    """
    This class works as an accumulator, but has an extra method that computes the change in parameters
    given the averages given by avg()
    """

    def __init__(self, enacc, transform, eps=1e-3):
        """
        """
        self.enacc = enacc
        self.transform = transform
        self.eps = eps
        self._onewf = StochasticReconfiguration(enacc, transform, eps)

    def onewf(self):
        return self._onewf

    def allwfs(self):
        return self

    def avg(self, configs, wfs, weights=None):
        """
        Compute (weighted) average
        """
        wfi = len(wfs) - 1
        dp = self.transform.serialize_gradients(wfs[wfi].pgradient())
        nconfig = weights.shape[-1]
        # here we assume that you are evaluating the derivative wrt the last wave function
        d = {}
        d["wtdp"] = np.einsum("cp,jkc->pjk", dp, weights, optimize=True) / nconfig
        return d

    def keys(self):
        return self.enacc.keys().union(["dpH", "dppsi", "dpidpj"])

    def shapes(self):
        nparms = self.transform.nparams()
        d = {"dppsi": (nparms,)}
        d.update(self.enacc.shapes())
        return d

    def update_state(self, hdf_file: h5py.File):
        """
        Update the state of the accumulator from a restart file.
        StochasticReconfiguration does not keep a state.

        hdf_file: h5py.File object
        """
        pass

    def block_average(self, data_sample1, data, weights):
        """
        This is meant to be called to create correctly weighted average after a number of blocks have
        been performed.
        weights are block, wf, wf
        data is a dictionary, with each entry being a numpy array of shape (block, ...) (i.e., block is added to the front of what's returned from avg())
        """
        weight_avg = np.mean(weights, axis=0)

        N = np.abs(weight_avg.diagonal())
        Nij = np.sqrt(np.outer(N, N))

        avg = {}
        error = {}
        wfi = Nij.shape[0] - 1
        for k in ["wtdp"]:
            it = data[k]
            avg[k] = np.mean(it, axis=0) / Nij[wfi]
            error[k] = scipy.stats.sem(it, axis=0) / Nij[wfi]

        avg["overlap"] = weight_avg

        for k in ['total', 'dppsi', 'dpH', "dpidpj"]:
            it = data_sample1[k]
            avg[k] = np.mean(it, axis=0) 
            error[k] = scipy.stats.sem(it, axis=0)
        return avg, error

    def _collect_terms(self, avg, error):
        ret = {}
        nwf = avg["overlap"].shape[0]
        N = np.abs(avg["overlap"].diagonal())
        Nij = np.sqrt(np.outer(N, N))

        # bits that don't depend on the overlap
        ret["dp_energy"] = np.real(avg["dpH"] - avg["total"] * avg["dppsi"])
        ret["dpidpj"] = np.real(
            avg["dpidpj"] - np.einsum("i,j->ij", avg["dppsi"], avg["dppsi"])
        )

        # overlap gradient
        fac = np.ones((nwf, nwf)) + np.identity(nwf)
        wfi = nwf-1
        ret["norm"] = N
        ret["overlap"] = avg["overlap"] / Nij
        ret["dp_norm"] = 2.0 * np.real(avg["wtdp"][:, wfi, wfi])
        norm_part = (
            np.einsum("i,p->pi", avg["overlap"][wfi, :], ret["dp_norm"])
            / N
        )
        ret["dp_overlap"] = (
            fac[wfi] * (avg["wtdp"][:, wfi, :] - 0.5 * norm_part) / Nij[wfi]
        )
        ret["energy"] = avg["total"]
        return ret

    def delta_p(
        self, steps: np.ndarray, data: dict, overlap_penalty: np.ndarray, verbose=False
    ):
        """
        steps: a list/numpy array of timesteps
        data: averaged data from avg() or __call__. Note that if you use VMC to compute this with
        an accumulator with a name, you'll need to remove that name from the keys.
        That is, the keys should be equal to the ones returned by keys().

        Compute the change in parameters given the data from a stochastic reconfiguration step.
        Return the change in parameters, and data that we may want to use for diagnostics.
        """
        data = self._collect_terms(data, None)
        nwf = data["overlap"].shape[0]
        wfi = nwf - 1
        overlap_cost = 0.0
        for i in range(wfi):
            overlap_cost += overlap_penalty[wfi, i] * data["overlap"][wfi, i]

        Sij = np.real(data["dpidpj"])
        invSij = np.linalg.inv(Sij + self.eps * np.eye(Sij.shape[0]))

        ovlp = 0.0

        for i in range(wfi):
            ovlp += (
                2.0
                * data["dp_overlap"][:, i]
                * overlap_penalty[wfi, i]
                * data["overlap"][wfi, i]
            )
        #print("dp_energy", data["dp_energy"])
        pgrad = data["dp_energy"] + ovlp

        v = np.einsum("ij,j->i", invSij, pgrad)
        dp = [-step * v for step in steps]
        report = {
            "pgrad": np.linalg.norm(pgrad),
            "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
            'overlap gradient norm': np.linalg.norm(ovlp),
            'Gradient norm': np.linalg.norm(pgrad),
        }
        return dp, report


def hdf_save(hdf_file, data, attr, wfs, configs):
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            for wfi, wf in enumerate(wfs):
                if f"wf/{wfi}" not in hdf.keys():
                    hdf.create_group(f"wf/{wfi}")
                    for k, it in wf.parameters.items():
                        hdf[f"wf/{wfi}/" + k] = it.copy()

            hdftools.append_hdf(hdf, data)
            if 'configs' not in hdf.keys():
                configs.initialize_hdf(hdf)
            configs.to_hdf(hdf)
            for wfi, wf in enumerate(wfs):
                for k, it in wf.parameters.items():
                    hdf[f"wf/{wfi}/" + k][:] = it.copy()


def set_wf_params(wf, params, updater):
    newparms = updater.transform.deserialize(wf, params)
    for k in newparms.keys():
        wf.parameters[k] = newparms[k]


def renormalize(wfs, norms, pivot=0, N=1):
    """
    Renormalize the wave functions so that they have the same normalization as the pivot wave function.

    .. math::

    """
    for i, wf in enumerate(wfs):
        if i == pivot:
            continue
        renorm = np.sqrt(norms[pivot] / norms[i] * N)
        if "wf1det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["wf1det_coeff"] = wf.parameters["wf1det_coeff"] * renorm
        elif "det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["det_coeff"] = wf.parameters["det_coeff"] * renorm
        else:
            raise NotImplementedError("need wf1det_coeff or det_coeff in parameters")


def evaluate_gradients(
    wfs, 
    configs_ensemble, 
    updater, 
    client=None, 
    npartitions=1, 
    vmc_kwargs={},
    overlap_kwargs={},
):
    """Evaluate parameter gradients for each state without threader
    It loops over states and runs Monte Carlo evaluations for them in sequence

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
    nwf = len(wfs)
    data_sample1_ensemble = [[0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
    data_weighted_ensemble = [[0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
    data_unweighted_ensemble = [[0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)]
    for wfi in range(nwf):
        wf = wfs[wfi]
        wf_copies = [copy.deepcopy(wf) for wf in wfs]
        transform_list = updater[wfi]
        for sub_iteration in range(len(transform_list)):
            transform = transform_list[sub_iteration]
            data_sample1_ensemble[wfi][sub_iteration], configs_ensemble[wfi][sub_iteration][0] = pyqmc.method.mc.vmc(
                wf,
                configs_ensemble[wfi][sub_iteration][0],
                accumulators={"": transform.onewf()},
                verbose=True,
                client=client,
                npartitions=npartitions,
                **vmc_kwargs,
            )
            (
                data_weighted_ensemble[wfi][sub_iteration], 
                data_unweighted_ensemble[wfi][sub_iteration], 
                configs_ensemble[wfi][sub_iteration][1],
            ) = pyqmc.method.sample_many.sample_overlap(
                wf_copies[0 : wfi + 1],
                configs_ensemble[wfi][sub_iteration][1],
                transform.allwfs(),
                client=client,
                npartitions=npartitions,
                **overlap_kwargs,
            )
    return data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble

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

    # Add 1 to the top `diff` indices
    y[idx[:diff]] += 1

    return y



def evaluate_gradients_threaded(
    wfs, 
    configs_ensemble, 
    updater,
    client=None, 
    npartitions=1, 
    vmc_kwargs={},
    overlap_kwargs={},
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
        for trans in transform:
            weights[threadcount] = 1.0
            threadcount += 1
    # overlap: the estimate is that the energy costs about the same
    # as sampling one wave function. So we add nwf/2.0 to the weight
    # because we are sampling wfi+1 wave functions
    for wfi, transform in enumerate(updater):
        if wfi == 0:
            threadcount+=1
            continue
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
            if wfi==0:
                threadcount+=1
                continue
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
    # we know for sure that the overlap for wf1 is just 1
    for sub_iteration in range(len(updater[wfi])):
        data_unweighted_ensemble[0][sub_iteration] = {'overlap':np.ones((10,1,1))}
        data_weighted_ensemble[0][sub_iteration] = {'overlap':np.ones((10,1,1))}
    if verbose:
        print("time to submit", middle_time-start_time, flush=True)
        print(pd.DataFrame(times))
    return data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble


def optimize_ensemble(
    wfs,
    configs,
    updater,
    hdf_file,
    tau=1,
    max_iterations=100,
    overlap_penalty=None,
    npartitions=None,
    client=None,
    verbose=False,
    use_threader=True,
    warmup_kwargs={},
    vmc_kwargs={},
    overlap_kwargs={},
):
    """Optimize a set of wave functions using ensemble VMC.


    Returns
    -------

    wfs : list of optimized wave functions
    """

    if len(warmup_kwargs) == 0:
        warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if len(vmc_kwargs) == 0:
        vmc_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if len(overlap_kwargs) == 0:
        overlap_kwargs = dict(nblocks=10, nsteps=10)
    if client is None:
        use_threader = False
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
        if not use_threader:
            data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble = evaluate_gradients(
                wfs, 
                configs_ensemble, 
                updater,
                client=client, 
                npartitions=npartitions, 
                vmc_kwargs=vmc_kwargs,
                overlap_kwargs=overlap_kwargs,
            )
        else:
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
