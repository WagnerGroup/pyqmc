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
import pyqmc
import h5py
from pyqmc.method import hdftools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import pyqmc.gpu as gpu
import os
from pyqmc.observables.stochastic_reconfiguration import StochasticReconfiguration
import scipy.stats


class StochasticReconfigurationWfbyWf:
    """
    This class works as an accumulator, but has an extra method that computes the change in parameters
    given the averages given by avg()
    """

    def __init__(self, enacc, transform, eps=1e-3):
        """ """
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

        for k in ["total", "dppsi", "dpH", "dpidpj"]:
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
        wfi = nwf - 1
        ret["norm"] = N
        ret["overlap"] = avg["overlap"] / Nij
        ret["dp_norm"] = 2.0 * np.real(avg["wtdp"][:, wfi, wfi])
        norm_part = np.einsum("i,p->pi", avg["overlap"][wfi, :], ret["dp_norm"]) / N
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
        if verbose:
            print("Overlap cost", overlap_cost)

        Sij = np.real(data["dpidpj"])
        invSij = np.linalg.inv(Sij + self.eps * np.eye(Sij.shape[0]))

        ovlp = 0.0
        #print("dp overlap", data["dp_overlap"].shape)

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
        }
        if verbose:
            print("overlap gradient norm", np.linalg.norm(ovlp))
            print("Gradient norm: ", np.linalg.norm(pgrad))
            print("Dot product between gradient and SR step: ", report["SRdot"])
        return dp, report


def _configs_to_hdf(hdf, configs):
    if "configs" not in hdf:
        configs.initialize_hdf(hdf)
    configs.to_hdf(hdf)


def load_all_configs(hdf, configs, updater):
    """
    Load all optimizer configs, or return `None` for legacy checkpoints.

    Args:
        hdf: the HDF5 object to load the configurations from
        configs: configs that supply the configuration type and shape
        updater: nested list of StochasticReconfigurationWfbyWf accumulators indexed by state then by sub-iteration

    Return:
        norm_configs: configs used for normalization sampling
        gradient_configs (list[list[dict]]): nested list of configs used for gradient sampling,
            indexed by state, sub-iteration, then kind ("energy" or "overlap")
    """
    if "all_configs" not in hdf:
        return None

    gradient_configs = []
    norm_configs = configs.copy()
    norm_configs.load_hdf(hdf["all_configs/normalization"])
    for wfi, transform_list in enumerate(updater):
        state_configs = []
        for sub_iteration in range(len(transform_list)):
            configs_by_kind = {}
            for kind in ("energy", "overlap"):
                cfg = configs.copy()
                cfg.load_hdf(
                    hdf[f"all_configs/gradient/{wfi}/{sub_iteration}/{kind}"]
                )
                configs_by_kind[kind] = cfg
            state_configs.append(configs_by_kind)
        gradient_configs.append(state_configs)
    return norm_configs, gradient_configs


def hdf_save(hdf_file, data, attr, wfs, norm_configs, gradient_configs):
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            for wfi, wf in enumerate(wfs):
                if f"wf/{wfi}" not in hdf.keys():
                    hdf.create_group(f"wf/{wfi}")
                    for k, it in wf.parameters.items():
                        hdf[f"wf/{wfi}/" + k] = it.copy()

            hdftools.append_hdf(hdf, data)
            all_configs_group = hdf.require_group("all_configs")
            _configs_to_hdf(all_configs_group.require_group("normalization"), norm_configs)
            gradient_group = all_configs_group.require_group("gradient")
            for wfi, state_configs in enumerate(gradient_configs):
                for sub_iteration, configs_by_kind in enumerate(state_configs):
                    for kind, cfg in configs_by_kind.items():
                        group = gradient_group.require_group(f"{wfi}/{sub_iteration}/{kind}")
                        _configs_to_hdf(group, cfg)
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
    x = x * target_sum / np.sum(x)
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
    y[y < 1] = 1

    return y


def _warmup_overlap(wfs, configs, client, npartitions, kwargs):
    if not kwargs or kwargs.get("nblocks", 1) <= 0:
        return configs
    _, _, configs = pyqmc.method.sample_many.sample_overlap(
        wfs,
        configs,
        None,
        client=client,
        npartitions=npartitions,
        **kwargs,
    )
    return configs


def _sampling_requested(kwargs):
    return bool(kwargs) and kwargs.get("nblocks", 1) > 0


def sample_gradient_configs_threaded(
    wfs,
    gradient_configs,
    updater=None,
    client=None,
    npartitions=1,
    vmc_kwargs=None,
    overlap_kwargs=None,
    verbose=True,
    overlap_thread_weight=None,
):
    """
    Sample all vmc and prefix-overlap configurations. It is a warmup run when `updater` is `None`
    (no gradient accumulators and all measurements are discarded). Otherwise, the same threaded
    scheduler performs the measured gradient sampling.

    Sampling jobs are assigned partitions according to their weights. Every active sampling job
    receives at least one partition, so when ``npartitions`` is smaller than the number of active
    jobs, the total assigned partitions is the number of active jobs.
    """
    run_energy = _sampling_requested(vmc_kwargs)
    run_overlap = _sampling_requested(overlap_kwargs)

    energy_data = [
        [None for _ in state_configs] for state_configs in gradient_configs
    ]
    overlap_data_weighted = [
        [None for _ in state_configs] for state_configs in gradient_configs
    ]
    overlap_data_unweighted = [
        [None for _ in state_configs] for state_configs in gradient_configs
    ]

    jobs = []
    if run_energy:
        for wfi, state_configs in enumerate(gradient_configs):
            for sub_iteration in range(len(state_configs)):
                jobs.append(("energy", wfi, sub_iteration, 1.0))
    if run_overlap:
        for wfi, state_configs in enumerate(gradient_configs):
            for sub_iteration in range(len(state_configs)):
                if overlap_thread_weight is None:
                    weight = (1 + wfi) / 2.0
                else:
                    weight = overlap_thread_weight[wfi]
                jobs.append(("overlap", wfi, sub_iteration, weight))

    if not jobs:
        return (
            energy_data,
            overlap_data_weighted,
            overlap_data_unweighted,
            gradient_configs,
        )

    available_partitions = len(jobs) if npartitions is None else npartitions
    npartitions_by_thread = round_to_fixed_sum(
        np.array([job[-1] for job in jobs]), available_partitions
    )
    if verbose:
        print("nthreads", len(jobs), "npartitions", npartitions_by_thread, flush=True)

    workers = {}
    start_time = time.perf_counter()

    # Without separate workers, sampling tasks share wave-function state, so run them one at a time
    max_workers = len(jobs) if client is not None else 1
    with ThreadPoolExecutor(max_workers=max_workers) as threader:
        for threadcount, (kind, wfi, sub_iteration, _) in enumerate(jobs):
            transform = None if updater is None else updater[wfi][sub_iteration]
            # vmc sampling
            if kind == "energy":
                accumulators = None if transform is None else {"": transform.onewf()}
                future = threader.submit(
                    pyqmc.method.mc.vmc,
                    wfs[wfi],
                    gradient_configs[wfi][sub_iteration]["energy"],
                    accumulators=accumulators,
                    verbose=verbose if updater is not None else False,
                    client=client,
                    npartitions=npartitions_by_thread[threadcount],
                    **vmc_kwargs,
                )
            # prefix-overlap sampling
            else:
                accumulator = None if transform is None else transform.allwfs()
                future = threader.submit(
                    pyqmc.method.sample_many.sample_overlap,
                    wfs[0:wfi + 1],
                    gradient_configs[wfi][sub_iteration]["overlap"],
                    accumulator,
                    client=client,
                    npartitions=npartitions_by_thread[threadcount],
                    **overlap_kwargs,
                )
            workers[future] = (kind, wfi, sub_iteration)

        middle_time = time.perf_counter()
        times = []
        for future in as_completed(workers):
            kind, wfi, sub_iteration = workers[future]
            times.append(
                {
                    "time": time.perf_counter() - middle_time,
                    "type": kind,
                    "wfi": wfi,
                    "sub_iteration": sub_iteration,
                }
            )
            if kind == "energy":
                (
                    energy_data[wfi][sub_iteration],
                    gradient_configs[wfi][sub_iteration]["energy"],
                ) = future.result()
            else:
                (
                    overlap_data_weighted[wfi][sub_iteration],
                    overlap_data_unweighted[wfi][sub_iteration],
                    gradient_configs[wfi][sub_iteration]["overlap"],
                ) = future.result()

    if verbose:
        print("time to submit", middle_time - start_time, flush=True)
        print(pd.DataFrame(times))
    return (
        energy_data,
        overlap_data_weighted,
        overlap_data_unweighted,
        gradient_configs,
    )


def evaluate_gradients_threaded(
    wfs,
    gradient_configs,
    updater,
    client=None,
    npartitions=1,
    vmc_kwargs=None,
    overlap_kwargs=None,
    refresh_vmc_warmup_kwargs=None,
    refresh_overlap_warmup_kwargs=None,
    verbose=True,
    overlap_thread_weight=None,
):
    """
    Evaluate parameter gradients for each state and sub-iteration using threads.
    It runs Monte Carlo evaluations for all the states asynchronously and gathers the results as they complete.
    Optional refresh sampling is performed first and its measurements are discarded.
    Samplings get parallelized using weighted partitions, which can be overridden by `overlap_thread_weight`.

    Args:
        gradient_configs (list[list[dict]]): nested list of configs indexed by state, sub-iteration, then kind ("energy" or "overlap")
        See ``optimize_ensemble`` for details of other parameters.

    Return:
        energy_data (list): nested list of vmc outputs indexed by state then by sub-iteration
        overlap_data_weighted (list): nested list of weighted sample_overlap outputs indexed by state then by sub-iteration
        overlap_data_unweighted (list): nested list of unweighted sample_overlap outputs indexed by state then by sub-iteration
        gradient_configs (list): updated configurations with the same structure
    """
    if not vmc_kwargs:
        vmc_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if not overlap_kwargs:
        overlap_kwargs = dict(nblocks=10, nsteps_per_block=10)

    # Refresh warmup
    if _sampling_requested(refresh_vmc_warmup_kwargs) or _sampling_requested(refresh_overlap_warmup_kwargs):
        _, _, _, gradient_configs = sample_gradient_configs_threaded(
            wfs,
            gradient_configs,
            updater=None,
            client=client,
            npartitions=npartitions,
            vmc_kwargs=refresh_vmc_warmup_kwargs,
            overlap_kwargs=refresh_overlap_warmup_kwargs,
            verbose=verbose,
            overlap_thread_weight=overlap_thread_weight,
        )

    # Gradient measurement
    return sample_gradient_configs_threaded(
        wfs,
        gradient_configs,
        updater=updater,
        client=client,
        npartitions=npartitions,
        vmc_kwargs=vmc_kwargs,
        overlap_kwargs=overlap_kwargs,
        verbose=verbose,
        overlap_thread_weight=overlap_thread_weight,
    )


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
    verbose=True,
    overlap_thread_weight=None,
    vmc_kwargs=None,
    overlap_kwargs=None,
    initial_vmc_warmup_kwargs=None,
    initial_overlap_warmup_kwargs=None,
    refresh_vmc_warmup_kwargs=None,
    refresh_overlap_warmup_kwargs=None,
    all_configs=None,
):
    """
    Optimize a set of wave functions using ensemble VMC.

    Separate configurations are maintained for the all-wf overlap, the vmc, and the prefix-overlap distributions.
    Warmups are performed by default. Empty individual warmup dictionaries disable the corresponding warmups.
    Initial warmups are skipped when all configurations can be restored from a checkpoint.
    Refresh warmups are run before measurements in each iteration, unless disabled.
    Starting configs precedence:
        1. If `hdf_file` exists, restart from its configs, initial warmups are skipped
        2. Otherwise, use `(norm_configs, gradient_configs)` from the supplied `all_configs` (optional)
        3. Otherwise, construct all configs from `configs`

    Args:
        wfs (list): list of wave functions to be optimized
        configs: initial configs before warmup if not loaded from a checkpoint
        updater (list[list]): nested list of StochasticReconfigurationWfbyWf accumulators indexed by state then by sub-iteration
        hdf_file (str): path for the checkpoint file
        tau (float): optimization step size
        max_iterations (int): maximum number of optimization iterations
        overlap_penalty (np.ndarray): overlap penalty matrix with shape (nwf, nwf)
        npartitions (int): total number of partitions distributed among the threads
        client: an object with submit() functions that return futures
        overlap_thread_weight (list): a list of float that overrides the default thread weights (1 + wfi) / 2.0
        vmc_kwargs (dict): options for measurement `vmc`
        overlap_kwargs (dict): options for measurement `sample_overlap`
        initial_vmc_warmup_kwargs (dict): options for initial warmup `vmc`; an empty dictionary disables it
        initial_overlap_warmup_kwargs (dict): options for initial warmup `sample_overlap`; an empty dictionary disables it
        refresh_vmc_warmup_kwargs (dict): options for refresh warmup `vmc`; an empty dictionary disables it
        refresh_overlap_warmup_kwargs (dict): options for refresh warmup `sample_overlap`; an empty dictionary disables it
        all_configs (tuple): `(norm_configs, gradient_configs)`, a full set of configs to start the optimization

    Return:
        wfs (list): list of optimized wave functions
    """

    if initial_vmc_warmup_kwargs is None:
        initial_vmc_warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if initial_overlap_warmup_kwargs is None:
        initial_overlap_warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if refresh_vmc_warmup_kwargs is None:
        refresh_vmc_warmup_kwargs = dict(nblocks=1, nsteps_per_block=30)
    if refresh_overlap_warmup_kwargs is None:
        refresh_overlap_warmup_kwargs = dict(nblocks=1, nsteps_per_block=30)
    if not vmc_kwargs:
        vmc_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if not overlap_kwargs:
        overlap_kwargs = dict(nblocks=10, nsteps_per_block=10)
    nwf = len(wfs)
    if overlap_penalty is None:
        overlap_penalty = np.ones((nwf, nwf)) * 0.5

    iteration_offset = 0
    checkpoint_configs = None
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                for wfi, wf in enumerate(wfs):
                    grp = hdf[f"wf/{wfi}"]
                    for k in grp.keys():
                        wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf.keys():
                iteration_offset = np.max(hdf["iteration"][...]) + 1
            checkpoint_configs = load_all_configs(hdf, configs, updater)
            if checkpoint_configs is None and "configs" in hdf:
                # Legacy checkpoints contain only one set of configs
                configs.load_hdf(hdf)

    restored_all_configs = checkpoint_configs is not None
    if restored_all_configs:
        norm_configs, gradient_configs = checkpoint_configs
    else:
        if all_configs is None:
            norm_configs = configs.copy()
            gradient_configs = [
                [
                    {"energy": configs.copy(), "overlap": configs.copy()}
                    for _ in transform_list
                ]
                for transform_list in updater
            ]
        else:
            norm_configs, gradient_configs = all_configs

        # Initial warmup: equilibrate every set of configs under its own target distribution
        norm_configs = _warmup_overlap(
            wfs,
            norm_configs,
            client=client,
            npartitions=npartitions,
            kwargs=initial_overlap_warmup_kwargs,
        )
        _, _, _, gradient_configs = sample_gradient_configs_threaded(
            wfs,
            gradient_configs,
            updater=None,
            client=client,
            npartitions=npartitions,
            vmc_kwargs=initial_vmc_warmup_kwargs,
            overlap_kwargs=initial_overlap_warmup_kwargs,
            verbose=verbose,
            overlap_thread_weight=overlap_thread_weight,
        )

    for i in range(iteration_offset, max_iterations):
        # Refresh warmup for the normalization sampling
        norm_configs = _warmup_overlap(
            wfs,
            norm_configs,
            client=client,
            npartitions=npartitions,
            kwargs=refresh_overlap_warmup_kwargs,
        )
        # Norm measurement
        _, data_unweighted, norm_configs = pyqmc.method.sample_many.sample_overlap(
            wfs,
            norm_configs,
            None,
            client=client,
            npartitions=npartitions,
            **overlap_kwargs,
        )
        norm = np.mean(data_unweighted["overlap"], axis=0)
        if verbose:
            print("Normalization step", norm.diagonal())
        renormalize(wfs, norm.diagonal(), pivot=0)

        # Refresh warmup and gradient measurement
        (
            energy_data,
            overlap_data_weighted,
            overlap_data_unweighted,
            gradient_configs,
        ) = evaluate_gradients_threaded(
            wfs,
            gradient_configs,
            updater,
            client=client,
            npartitions=npartitions,
            vmc_kwargs=vmc_kwargs,
            overlap_kwargs=overlap_kwargs,
            refresh_vmc_warmup_kwargs=refresh_vmc_warmup_kwargs,
            refresh_overlap_warmup_kwargs=refresh_overlap_warmup_kwargs,
            overlap_thread_weight=overlap_thread_weight,
        )

        for wfi, wf in enumerate(wfs):
            transform_list = updater[wfi]
            for sub_iteration, transform in enumerate(transform_list):
                transform = transform_list[sub_iteration]
                avg, error = transform.block_average(
                    energy_data[wfi][sub_iteration],
                    overlap_data_weighted[wfi][sub_iteration],
                    overlap_data_unweighted[wfi][sub_iteration]["overlap"],
                )
                if verbose:
                    print(
                        "Iteration",
                        i,
                        "wf ",
                        wfi,
                        " sub iteration ",
                        sub_iteration,
                        "Energy",
                        avg["total"],
                        "Overlap",
                        avg["overlap"][wfi, :],
                    )
                dp, report = transform.delta_p(
                    [tau], avg, overlap_penalty, verbose=True
                )
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
                hdf_save(
                    hdf_file,
                    save_data,
                    {"tau": tau},
                    wfs,
                    norm_configs,
                    gradient_configs,
                )

    return wfs
