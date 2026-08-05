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
Ensemble (excited state) optimization with minSR.

This is the minSR counterpart of
:mod:`pyqmc.method.ensemble_optimization_threaded`. The algorithm is unchanged:
each state is sampled separately in its own thread, each state above the ground
state carries an overlap penalty against the lower states, and the update is a
stochastic reconfiguration step. The only difference is how the SR equations are
solved.

The gradient here has two pieces,

.. math:: f = f_{\\rm energy} + f_{\\rm overlap}

The energy piece comes from per-sample derivatives, so
:math:`f_{\\rm energy} = A^T b` and the overlap matrix is :math:`S = A^T A`,
which is exactly the minSR structure. The overlap penalty piece comes from the
separate overlap sampling and is just a vector, with no reason to lie in the row
space of A. :func:`pyqmc.method.minsr.sr_solve` applies the same regularized
inverse to both using only the (nsamples, nsamples) kernel, so the
(nparameters, nparameters) S matrix is never built for either piece.
"""

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import scipy.stats

import pyqmc.gpu as gpu
import pyqmc.method.mc
import pyqmc.method.sample_many
from pyqmc.method.ensemble_optimization_threaded import round_to_fixed_sum
from pyqmc.method.ensemble_optimization_wfbywf import (
    hdf_save,
    renormalize,
    set_wf_params,
)
from pyqmc.method.minsr import real_design_matrix, sample_minsr_data, sr_solve


class MinSRWfbyWf:
    """Updater for one state of an ensemble, the minSR analogue of
    :class:`pyqmc.method.ensemble_optimization_wfbywf.StochasticReconfigurationWfbyWf`.

    It is used in two places: as the accumulator passed to
    :func:`pyqmc.method.sample_many.sample_overlap`, where `avg` accumulates the
    weighted derivatives that give the overlap gradient, and as the object that
    turns the sampled data into a parameter change in `delta_p`.

    Unlike the SR version there is no `onewf` accumulator, because the energy
    part is sampled per configuration by
    :func:`pyqmc.method.minsr.sample_minsr_data` rather than averaged into
    dpH/dppsi/dpidpj.

    :parameter enacc: an EnergyAccumulator-like object
    :parameter transform: a LinearTransform for this state's parameters
    :parameter float eps: regularization of the SR equations
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    """

    def __init__(self, enacc, transform, eps=1e-2, nodal_cutoff=1e-3):
        self.enacc = enacc
        self.transform = transform
        self.eps = eps
        self.nodal_cutoff = nodal_cutoff

    def allwfs(self):
        return self

    def avg(self, configs, wfs, weights):
        """Weighted derivatives of the last wave function in `wfs`, averaged over
        configurations. Identical to the StochasticReconfigurationWfbyWf version.
        """
        wfi = len(wfs) - 1
        dp = self.transform.serialize_gradients(wfs[wfi].pgradient())
        nconfig = weights.shape[-1]
        return {"wtdp": np.einsum("cp,jkc->pjk", dp, weights, optimize=True) / nconfig}

    def keys(self):
        return self.enacc.keys().union(["wtdp"])

    def shapes(self):
        d = {"wtdp": (self.transform.nparams,)}
        d.update(self.enacc.shapes())
        return d

    def update_state(self, hdf_file: h5py.File):
        """This accumulator keeps no state."""
        pass

    def block_average(self, data_sample1, data, weights):
        """Average the sampled data, with the same signature and normalization as
        StochasticReconfigurationWfbyWf.block_average.

        `data_sample1` is the per-sample output of sample_minsr_data rather than
        block-averaged SR data, so the derivative rows are passed through
        unaveraged for delta_p to build the design matrix from.
        """
        weight_avg = np.mean(weights, axis=0)
        N = np.abs(weight_avg.diagonal())
        Nij = np.sqrt(np.outer(N, N))
        wfi = Nij.shape[0] - 1

        avg = {}
        error = {}
        for k in ["wtdp"]:
            it = data[k]
            avg[k] = np.mean(it, axis=0) / Nij[wfi]
            error[k] = scipy.stats.sem(it, axis=0) / Nij[wfi]
        avg["overlap"] = weight_avg

        eloc = data_sample1["total"]
        avg["dppsi_samples"] = data_sample1["dppsi"]
        avg["eloc_samples"] = eloc
        avg["total"] = np.mean(eloc).real
        block_energy = data_sample1["block_energy"].real
        if len(block_energy) > 1:
            error["total"] = scipy.stats.sem(block_energy)
        else:
            error["total"] = np.std(eloc.real) / np.sqrt(len(eloc))
        return avg, error

    def _collect_terms(self, avg):
        """Normalized overlaps and their derivatives. This is the overlap part of
        StochasticReconfigurationWfbyWf._collect_terms; the energy part is
        replaced by the design matrix built in delta_p.
        """
        ret = {}
        nwf = avg["overlap"].shape[0]
        N = np.abs(avg["overlap"].diagonal())
        Nij = np.sqrt(np.outer(N, N))

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

    def delta_p(self, steps, data, overlap_penalty, verbose=False):
        """Compute the change in parameters for this state.

        Solves :math:`(S + \\epsilon I) v = f_{\\rm energy} + f_{\\rm overlap}`
        in sample space. The energy gradient convention matches
        StochasticReconfigurationWfbyWf, so `steps` and `overlap_penalty` mean
        the same thing as they do there.

        :parameter steps: list of timesteps
        :parameter dict data: output of block_average
        :parameter overlap_penalty: (nwf, nwf) penalty matrix
        :returns: (list of parameter changes, report dictionary)
        """
        terms = self._collect_terms(data)
        nwf = terms["overlap"].shape[0]
        wfi = nwf - 1

        overlap_cost = 0.0
        ovlp = 0.0
        for i in range(wfi):
            overlap_cost += overlap_penalty[wfi, i] * terms["overlap"][wfi, i]
            ovlp += (
                2.0
                * terms["dp_overlap"][:, i]
                * overlap_penalty[wfi, i]
                * terms["overlap"][wfi, i]
            )

        A, b = real_design_matrix(data["dppsi_samples"], data["eloc_samples"])
        g = None if wfi == 0 else np.real(ovlp)
        v = sr_solve(A, b, g=g, eps=self.eps)
        dp = [-step * v for step in steps]

        pgrad = A.T @ b if g is None else A.T @ b + g
        report = {
            "pgrad": np.linalg.norm(pgrad),
            "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
            "overlap_cost": overlap_cost,
        }
        if verbose:
            print("Overlap cost", overlap_cost)
            print("overlap gradient norm", np.linalg.norm(ovlp))
            print("Gradient norm: ", report["pgrad"])
            print("Dot product between gradient and SR step: ", report["SRdot"])
        return dp, report


def evaluate_gradients_threaded(
    wfs,
    configs_ensemble,
    updater,
    client=None,
    npartitions=1,
    minsr_kwargs=None,
    overlap_kwargs=None,
    verbose=True,
    overlap_thread_weight=None,
):
    """Sample the energy derivatives and the overlaps for every state, threaded.

    Same structure as
    :func:`pyqmc.method.ensemble_optimization_threaded.evaluate_gradients_threaded`:
    two threads per (state, sub-iteration), one sampling that state alone and one
    sampling the combined distribution of the states up to it, with the client's
    workers divided between the threads by estimated cost. The difference is that
    the first thread stores per-configuration derivatives instead of accumulating
    dpidpj.

    :parameter list wfs: list of wave functions
    :parameter list configs_ensemble: nested list of configurations indexed by state, sub-iteration, then thread
    :parameter list updater: nested list of MinSRWfbyWf objects indexed by state then sub-iteration
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :parameter dict minsr_kwargs: options for sample_minsr_data
    :parameter dict overlap_kwargs: options for sample_overlap

    :return: (data_sample1_ensemble, data_weighted_ensemble, data_unweighted_ensemble, configs_ensemble)

    With a client the sampling runs in worker processes, so the threads only
    read the wave functions. Without one it runs in this process, where the
    samplers mutate wave function internal state, so each thread is given its own
    copy. That is meant for testing; real runs should pass a client.
    """
    if minsr_kwargs is None:
        minsr_kwargs = {}
    if overlap_kwargs is None:
        overlap_kwargs = {}
    nwf = len(wfs)
    nthreads = 2 * sum([len(updater[wfi]) for wfi in range(nwf)])
    data_sample1_ensemble = [
        [0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)
    ]
    data_weighted_ensemble = [
        [0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)
    ]
    data_unweighted_ensemble = [
        [0 for _ in range(len(updater[wfi]))] for wfi in range(nwf)
    ]
    energy_workers = {}
    overlap_workers = {}
    if nthreads == 0:
        return (
            data_sample1_ensemble,
            data_weighted_ensemble,
            data_unweighted_ensemble,
            configs_ensemble,
        )

    weights = np.zeros(nthreads)
    threadcount = 0
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
            for _ in transform:
                weights[threadcount] = (1 + wfi) / 2.0
                threadcount += 1
        else:
            for _ in transform:
                weights[threadcount] = overlap_thread_weight[wfi]
                threadcount += 1

    npartitions_by_thread = round_to_fixed_sum(weights, npartitions)

    if verbose:
        print("nthreads", nthreads, "npartitions", npartitions_by_thread, flush=True)
    start_time = time.perf_counter()
    threadcount = 0
    with ThreadPoolExecutor(max_workers=nthreads) as threader:
        for wfi, wf in enumerate(wfs):
            for sub_iteration, transform in enumerate(updater[wfi]):
                energy_workers_thread = threader.submit(
                    sample_minsr_data,
                    wf if client is not None else copy.deepcopy(wf),
                    configs_ensemble[wfi][sub_iteration][0],
                    transform.transform,
                    transform.enacc,
                    transform.nodal_cutoff,
                    client=client,
                    npartitions=npartitions_by_thread[threadcount],
                    **minsr_kwargs,
                )
                energy_workers[energy_workers_thread] = (wfi, sub_iteration)
                threadcount += 1
        for wfi, wf in enumerate(wfs):
            for sub_iteration, transform in enumerate(updater[wfi]):
                sampled_wfs = wfs[0 : wfi + 1]
                if client is None:
                    sampled_wfs = [copy.deepcopy(w) for w in sampled_wfs]
                overlap_workers_thread = threader.submit(
                    pyqmc.method.sample_many.sample_overlap,
                    sampled_wfs,
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
                times.append(
                    {
                        "time": time.perf_counter() - middle_time,
                        "type": "energy",
                        "wfi": wfi,
                        "sub_iteration": sub_iteration,
                    }
                )
                (
                    data_sample1_ensemble[wfi][sub_iteration],
                    configs_ensemble[wfi][sub_iteration][0],
                ) = future.result()
            elif future in overlap_workers:
                times.append(
                    {
                        "time": time.perf_counter() - middle_time,
                        "type": "overlap",
                        "wfi": wfi,
                        "sub_iteration": sub_iteration,
                    }
                )
                (
                    data_weighted_ensemble[wfi][sub_iteration],
                    data_unweighted_ensemble[wfi][sub_iteration],
                    configs_ensemble[wfi][sub_iteration][1],
                ) = future.result()
            else:
                raise ValueError("Received unknown future")
    if verbose:
        print("time to submit", middle_time - start_time, flush=True)
        print(pd.DataFrame(times))
    return (
        data_sample1_ensemble,
        data_weighted_ensemble,
        data_unweighted_ensemble,
        configs_ensemble,
    )


def optimize_ensemble(
    wfs,
    configs,
    updater,
    hdf_file,
    client=None,
    tau=0.1,
    max_iterations=100,
    overlap_penalty=None,
    npartitions=None,
    verbose=True,
    overlap_thread_weight=None,
    warmup_kwargs=None,
    minsr_kwargs=None,
    overlap_kwargs=None,
):
    """Optimize a set of wave functions using ensemble VMC with minSR.

    This is a drop-in replacement for
    :func:`pyqmc.method.ensemble_optimization_threaded.optimize_ensemble` that
    takes MinSRWfbyWf updaters instead of StochasticReconfigurationWfbyWf ones
    and never builds the S matrix. `tau` and `overlap_penalty` have the same
    meaning in both.

    The number of samples per state per iteration is
    nconfig * minsr_kwargs['nblocks'], and the kernel that gets solved is square
    in that number, so nblocks defaults to 1 here where the SR version uses 10
    blocks of averaging. Raise it for better statistics at quadratic memory cost.

    :parameter list wfs: list of wave functions, ordered from the ground state up
    :parameter configs: initial configurations
    :parameter list updater: nested list of MinSRWfbyWf objects indexed by state then sub-iteration
    :parameter str hdf_file: file to store output; the format matches the SR version
    :parameter client: an object with submit() functions that return futures
    :parameter float tau: step size in parameter space
    :parameter int max_iterations: total number of iterations, including any read from hdf_file
    :parameter overlap_penalty: (nwf, nwf) penalty matrix, default 0.5 everywhere
    :parameter int npartitions: the number of workers to submit at a time
    :parameter dict warmup_kwargs: options for the initial vmc warmup
    :parameter dict minsr_kwargs: options for sample_minsr_data
    :parameter dict overlap_kwargs: options for sample_overlap
    :returns: list of optimized wave functions
    """
    if warmup_kwargs is None or len(warmup_kwargs) == 0:
        warmup_kwargs = {"nblocks": 1, "nsteps_per_block": 100}
    if minsr_kwargs is None or len(minsr_kwargs) == 0:
        minsr_kwargs = {"nblocks": 1, "nsteps_per_block": 10}
    if overlap_kwargs is None or len(overlap_kwargs) == 0:
        overlap_kwargs = {"nblocks": 10, "nsteps": 10}
    nwf = len(wfs)
    if overlap_penalty is None:
        overlap_penalty = np.ones((nwf, nwf)) * 0.5

    iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf:
                for wfi, wf in enumerate(wfs):
                    grp = hdf[f"wf/{wfi}"]
                    for k in grp:
                        wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf:
                iteration_offset = np.max(hdf["iteration"][...]) + 1
            configs.load_hdf(hdf)
    else:
        _, configs = pyqmc.method.mc.vmc(
            wfs[0],
            configs,
            verbose=verbose,
            client=client,
            npartitions=npartitions,
            **warmup_kwargs,
        )

    configs_ensemble = [
        [[copy.deepcopy(configs) for _ in range(2)] for _ in range(len(updater[wfi]))]
        for wfi in range(nwf)
    ]
    for i in range(iteration_offset, max_iterations):
        # renormalize so that the overlap matrix is in terms of normalized states
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

        (
            data_sample1_ensemble,
            data_weighted_ensemble,
            data_unweighted_ensemble,
            configs_ensemble,
        ) = evaluate_gradients_threaded(
            wfs,
            configs_ensemble,
            updater,
            client=client,
            npartitions=npartitions,
            minsr_kwargs=minsr_kwargs,
            overlap_kwargs=overlap_kwargs,
            verbose=verbose,
            overlap_thread_weight=overlap_thread_weight,
        )

        for wfi, wf in enumerate(wfs):
            for sub_iteration, transform in enumerate(updater[wfi]):
                avg, error = transform.block_average(
                    data_sample1_ensemble[wfi][sub_iteration],
                    data_weighted_ensemble[wfi][sub_iteration],
                    data_unweighted_ensemble[wfi][sub_iteration]["overlap"],
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
                    [tau], avg, overlap_penalty, verbose=verbose
                )
                x = transform.transform.serialize_parameters(wf.parameters)
                set_wf_params(wf, x + dp[0], transform)

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
                    {"tau": tau, "eps": transform.eps},
                    wfs,
                    configs_ensemble[wfi][sub_iteration][0],
                )

    return wfs
