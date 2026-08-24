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


import importlib

import pyqmc.method.sample_many
import numpy as np
import pyqmc
import h5py
from pyqmc.method import hdftools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple
import time
import pyqmc.gpu as gpu
import os
from pyqmc.observables.stochastic_reconfiguration import StochasticReconfiguration
import scipy.stats


class StochasticReconfigurationWfbyWf:
    """
    This class works as an accumulator, but has an extra method that computes the change in parameters
    given the averages given by avg()
    """

    def __init__(self, enacc, transform, eps=1e-3, nodal_cutoff=1e-3):
        """ """
        self.enacc = enacc
        self.transform = transform
        self.eps = eps
        self.nodal_cutoff = nodal_cutoff
        # eps used to be passed positionally here, which landed it in
        # nodal_cutoff; the two defaults coincide, so only a non-default eps was
        # affected, and it retuned the nodal regularization rather than the solve
        self._onewf = StochasticReconfiguration(
            enacc, transform, nodal_cutoff=nodal_cutoff, eps=eps
        )

    def onewf(self):
        return self._onewf

    def allwfs(self):
        return self

    def sample_energy(self, wf, configs, client=None, npartitions=None, verbose=True, **kwargs):
        """Sample this state on its own and accumulate whatever delta_p needs
        from that distribution, here the SR averages dpH, dppsi, and dpidpj.

        The driver calls this rather than running vmc itself, so that an updater
        that needs something else from the single-state sampling -- minSR needs
        the derivatives per configuration, not averaged into dpidpj -- can be
        dropped in without a separate driver.

        :returns: (data, configs) with data in the form block_average expects
        """
        return pyqmc.method.mc.vmc(
            wf,
            configs,
            accumulators={"": self.onewf()},
            verbose=verbose,
            client=client,
            npartitions=npartitions,
            **kwargs,
        )

    def sample_overlap(self, wfs, configs, client=None, npartitions=None, **kwargs):
        """Sample the overlap distribution and accumulate whatever delta_p needs
        from it, here the weighted derivatives averaged over every step.

        The driver calls this rather than passing an accumulator itself, so that
        an updater that wants something else from the overlap sampling -- minSR
        evaluates the derivatives on one snapshot per block rather than at every
        step -- can be dropped in without a separate driver.

        :returns: (weighted, unweighted, configs)
        """
        return pyqmc.method.sample_many.sample_overlap(
            wfs,
            configs,
            self.allwfs(),
            client=client,
            npartitions=npartitions,
            **kwargs,
        )

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
            error[k] = _block_sem(it) / Nij[wfi]

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
            "overlap_cost": overlap_cost,
        }
        if verbose:
            print("overlap gradient norm", np.linalg.norm(ovlp))
            print("Gradient norm: ", np.linalg.norm(pgrad))
            print("Dot product between gradient and SR step: ", report["SRdot"])
        return dp, report


def _block_sem(blocks):
    """Standard error over the leading (block) axis.

    scipy.stats.sem warns and returns nan for a single block, which is a common
    case here since the per-sample methods default to one. Return the nan
    without the warning.
    """
    if blocks.shape[0] < 2:
        return np.full(blocks.shape[1:], np.nan)
    return scipy.stats.sem(blocks, axis=0)


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
            for k, it in attr.items():
                if k not in hdf.attrs:
                    hdf.attrs[k] = it
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


#: algorithms optimize_ensemble can be asked for by name, and the regularization
#: each one wants by default
UPDATERS = {
    "sr": (StochasticReconfigurationWfbyWf, 1e-3),
    # given as (module, class name) so that the import happens on demand and
    # those modules are free to build on this one; see make_updater
    "minsr": (("pyqmc.method.ensemble_minsr", "MinSRWfbyWf"), 1e-2),
    "cgsr": (("pyqmc.method.ensemble_cgsr", "CGSRWfbyWf"), 1e-2),
}


def make_updater(transform, enacc, method="sr", eps=None, nodal_cutoff=1e-3):
    """Build the per-state updater for one algorithm.

    :parameter transform: a LinearTransform for this state's parameters
    :parameter enacc: an EnergyAccumulator-like object
    :parameter str method: 'sr' for stochastic reconfiguration, which builds the
        (nparameters, nparameters) S matrix, 'minsr', which solves the same
        equations in sample space and never builds it, or 'cgsr', which solves
        them iteratively and builds neither that matrix nor the sample-space
        kernel
    :parameter float eps: regularization of the solve; defaults to what the
        method wants, 1e-3 for sr and 1e-2 for minsr and cgsr
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    """
    if method not in UPDATERS:
        raise ValueError(
            f"Unknown method {method!r}; choose one of {sorted(UPDATERS)}."
        )
    cls, default_eps = UPDATERS[method]
    if isinstance(cls, tuple):  # deferred import
        module_name, class_name = cls
        cls = getattr(importlib.import_module(module_name), class_name)
    return cls(
        enacc,
        transform,
        eps=default_eps if eps is None else eps,
        nodal_cutoff=nodal_cutoff,
    )


def build_updaters(transforms, enacc=None, method="sr", eps=None, nodal_cutoff=1e-3):
    """Turn parameter transforms into the nested list of updaters the driver uses.

    :parameter transforms: one transform per state, or a nested list indexed by
        state then sub-iteration. Objects that are already updaters (anything
        with delta_p) are passed through, so a hand-built updater still works.
    :parameter enacc: an EnergyAccumulator-like object shared by every state, or
        a list with one per state
    :returns: nested list of updaters indexed by state then sub-iteration
    """
    nested = [t if isinstance(t, (list, tuple)) else [t] for t in transforms]
    if all(hasattr(t, "delta_p") for state in nested for t in state):
        return [list(state) for state in nested]
    if enacc is None:
        raise ValueError(
            "enacc is required to build updaters from transforms; pass an "
            "EnergyAccumulator, or pass updater objects instead of transforms."
        )
    enaccs = enacc if isinstance(enacc, (list, tuple)) else [enacc] * len(nested)
    if len(enaccs) != len(nested):
        raise ValueError(
            f"got {len(enaccs)} energy accumulators for {len(nested)} states"
        )
    return [
        [make_updater(t, e, method, eps, nodal_cutoff) for t in state]
        for state, e in zip(nested, enaccs)
    ]


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


class _SamplingJob(NamedTuple):
    """One sampling task: which distribution, for which state and sub-iteration.

    `weight` is this job's share of the partitions. `cost` is an estimate of its
    wall time, used only to decide submission order.
    """

    kind: str  # "energy" or "overlap"
    wfi: int
    sub_iteration: int
    weight: float
    cost: float

    def label(self, with_sub_iteration=False):
        sub = f".{self.sub_iteration}" if with_sub_iteration else ""
        return f"{self.kind} wf{self.wfi}{sub}"


def _sweeps(kwargs):
    """Metropolis sweeps one sampling job will run."""
    return kwargs.get("nblocks", 1) * kwargs.get("nsteps_per_block", 10)


def _build_sampling_jobs(
    gradient_configs, vmc_kwargs, overlap_kwargs, overlap_thread_weight=None
):
    """List the sampling jobs for one round, longest first.

    Ordering matters when there are fewer partitions than jobs: the client then
    has fewer workers than tasks and starts them in submission order, so
    submitting a long job last leaves it running alone at the end and sets the
    makespan. The prefix-overlap sampling for the highest state propagates the
    most wave functions and is usually the longest job of the round, so it goes
    first.

    The cost estimate is sweeps times the number of wave functions propagated,
    which is why overlap sampling for state `wfi` counts `wfi + 1`. It is only
    used for ordering; `weight`, which `overlap_thread_weight` overrides, is what
    divides up the partitions.
    """
    jobs = []
    if _sampling_requested(vmc_kwargs):
        sweeps = _sweeps(vmc_kwargs)
        for wfi, state_configs in enumerate(gradient_configs):
            for sub_iteration in range(len(state_configs)):
                jobs.append(_SamplingJob("energy", wfi, sub_iteration, 1.0, sweeps))
    if _sampling_requested(overlap_kwargs):
        sweeps = _sweeps(overlap_kwargs)
        for wfi, state_configs in enumerate(gradient_configs):
            for sub_iteration in range(len(state_configs)):
                if overlap_thread_weight is None:
                    weight = (1 + wfi) / 2.0
                else:
                    weight = overlap_thread_weight[wfi]
                jobs.append(
                    _SamplingJob(
                        "overlap", wfi, sub_iteration, weight, sweeps * (wfi + 1)
                    )
                )
    # ties go to overlap, which does more work per sweep than vmc, then to the
    # higher state
    jobs.sort(key=lambda job: (-job.cost, job.kind == "energy", -job.wfi))
    return jobs


def _timed(function, *args, **kwargs):
    """Run `function` and report how long it took, so that the summary reports
    durations rather than completion times."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def _submit_sampling_job(
    threader, job, wfs, gradient_configs, updater, client, npartitions,
    vmc_kwargs, overlap_kwargs,
):
    """Submit one job. `updater` of None means a warmup: propagate only, measure
    nothing.

    :parameter npartitions: partitions for this job alone, not the total
    :returns: a future whose result is (sampling output, seconds)
    """
    configs = gradient_configs[job.wfi][job.sub_iteration][job.kind]
    transform = None if updater is None else updater[job.wfi][job.sub_iteration]
    # the samplers print progress per block, which is unreadable interleaved
    # across threads; the summary from _format_sampling_summary replaces it
    common = dict(client=client, npartitions=npartitions)

    if job.kind == "energy":
        if transform is None:
            return threader.submit(
                _timed, pyqmc.method.mc.vmc, wfs[job.wfi], configs,
                accumulators=None, verbose=False, **common, **vmc_kwargs,
            )
        # the updater decides what to collect from this sampling
        return threader.submit(
            _timed, transform.sample_energy, wfs[job.wfi], configs,
            verbose=False, **common, **vmc_kwargs,
        )

    prefix = wfs[: job.wfi + 1]
    if transform is None:
        return threader.submit(
            _timed, pyqmc.method.sample_many.sample_overlap, prefix, configs, None,
            **common, **overlap_kwargs,
        )
    return threader.submit(
        _timed, transform.sample_overlap, prefix, configs, **common, **overlap_kwargs
    )


def _format_step_report(report):
    """The diagnostics an updater returns from delta_p, on one line.

    pgrad and SRdot are common to every updater; anything else it reports --
    the overlap penalty cost, CG iteration counts, how long the solve took -- is
    appended as it comes, so that a new updater's diagnostics show up without
    touching the driver. Keys ending in `_seconds` are printed as times, and
    flags that are True are left out, having nothing to report.
    """
    named = {"pgrad": "|grad|", "SRdot": "grad.step"}
    parts = [
        f"{named[key]} = {float(np.real(report[key])):.4g}"
        for key in named
        if key in report
    ]
    for key, value in report.items():
        if key in named or np.ndim(value) != 0 or isinstance(value, str):
            continue
        if isinstance(value, (bool, np.bool_)):
            if not value:  # a flag is worth printing only when it is a problem
                parts.append(f"{key} = False")
            continue
        if key.endswith("_seconds"):  # any updater can report a timing this way
            parts.append(f"{key[: -len('_seconds')]} = {float(value):.3f}s")
            continue
        parts.append(f"{key} = {float(np.real(value)):.4g}")
    return "   ".join(parts)


def _format_sampling_summary(jobs, partitions, durations, submit_time):
    """One block of text for a round of sampling: what ran, on how many
    partitions, and how long each took."""
    with_sub = any(job.sub_iteration for job in jobs)
    order = sorted(range(len(jobs)), key=lambda i: -durations[i])
    lines = [
        f"  sampling: {len(jobs)} jobs on {sum(partitions)} partitions"
        f" (submitted in {submit_time:.3f}s)"
    ]
    for i in order:
        lines.append(
            f"      {jobs[i].label(with_sub):<16s} {durations[i]:7.2f}s"
            f"  [{partitions[i]} partition{'s' if partitions[i] != 1 else ''}]"
        )
    return "\n".join(lines)


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
    jobs, the total assigned partitions is the number of active jobs. Jobs are submitted longest
    first; see :func:`_build_sampling_jobs`.
    """
    energy_data = [[None for _ in state] for state in gradient_configs]
    overlap_data_weighted = [[None for _ in state] for state in gradient_configs]
    overlap_data_unweighted = [[None for _ in state] for state in gradient_configs]

    jobs = _build_sampling_jobs(
        gradient_configs, vmc_kwargs, overlap_kwargs, overlap_thread_weight
    )
    if not jobs:
        return (
            energy_data,
            overlap_data_weighted,
            overlap_data_unweighted,
            gradient_configs,
        )

    available_partitions = len(jobs) if npartitions is None else npartitions
    partitions = round_to_fixed_sum(
        np.array([job.weight for job in jobs]), available_partitions
    )

    durations = [0.0] * len(jobs)
    # Without separate workers, sampling tasks share wave-function state, so run them one at a time
    max_workers = len(jobs) if client is not None else 1
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as threader:
        futures = {
            _submit_sampling_job(
                threader, job, wfs, gradient_configs, updater, client,
                partitions[i], vmc_kwargs, overlap_kwargs,
            ): i
            for i, job in enumerate(jobs)
        }
        submit_time = time.perf_counter() - start_time

        for future in as_completed(futures):
            i = futures[future]
            job = jobs[i]
            result, durations[i] = future.result()
            if job.kind == "energy":
                (
                    energy_data[job.wfi][job.sub_iteration],
                    gradient_configs[job.wfi][job.sub_iteration]["energy"],
                ) = result
            else:
                (
                    overlap_data_weighted[job.wfi][job.sub_iteration],
                    overlap_data_unweighted[job.wfi][job.sub_iteration],
                    gradient_configs[job.wfi][job.sub_iteration]["overlap"],
                ) = result

    if verbose:
        print(
            _format_sampling_summary(jobs, partitions, durations, submit_time),
            flush=True,
        )
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
    transforms,
    hdf_file,
    enacc=None,
    method="sr",
    eps=None,
    nodal_cutoff=1e-3,
    tau=.02,
    max_iterations=100,
    overlap_penalty=None,
    npartitions=None,
    client=None,
    verbose=True,
    overlap_thread_weight=None,
    vmc_kwargs=None,
    overlap_kwargs=None,
    norm_kwargs=None,
    initial_vmc_warmup_kwargs=None,
    initial_overlap_warmup_kwargs=None,
    refresh_vmc_warmup_kwargs=None,
    refresh_overlap_warmup_kwargs=None,
    all_configs=None,
):
    """
    Optimize a set of wave functions using ensemble VMC.

    Separate configurations are maintained for the all-wf overlap, the vmc, and the prefix-overlap distributions.
    Initial warmups are performed by default, and are skipped when all configurations can be restored from a checkpoint.
    Empty individual warmup dictionaries disable the corresponding warmups.
    Refresh warmups before each iteration's measurements are off by default: every set of configs is already
    propagated by its own measurement each iteration, and the measurement sampling is doing almost exactly what
    a refresh warmup would do. Enable them by passing the corresponding dictionaries if a case needs it.
    Starting configs precedence:
        1. If `hdf_file` exists, restart from its configs, initial warmups are skipped
        2. Otherwise, use `(norm_configs, gradient_configs)` from the supplied `all_configs` (optional)
        3. Otherwise, construct all configs from `configs`

    Args:
        wfs (list): list of wave functions to be optimized
        configs: initial configs before warmup if not loaded from a checkpoint
        transforms (list): one LinearTransform per state, or a nested list indexed
            by state then by sub-iteration. Already-built updaters are accepted
            here too, in which case enacc, method, eps, and nodal_cutoff are unused.
        hdf_file (str): path for the checkpoint file
        enacc: an EnergyAccumulator-like object shared by every state, or a list
            with one per state. Required unless transforms are already updaters.
        method (str): 'sr' for stochastic reconfiguration, which builds the
            (nparameters, nparameters) S matrix; 'minsr', which solves the same
            equations in sample space without building it; or 'cgsr', which
            solves them by conjugate gradient and builds neither that matrix nor
            the sample-space kernel.
            minSR is worth it when there are more parameters than samples. Note
            that the number of samples per state is
            nconfig * vmc_kwargs['nblocks'], and the kernel it solves is square
            in that, which is why it defaults to one block rather than ten.
            cgsr shares that default because it shares the sampling, but it does
            not share the reason: its cost and memory are both linear in the
            sample count, so it is the one to raise nblocks on if the statistics
            need it. See pyqmc.method.cgsr for where the crossover sits.
        eps (float): regularization of the solve; defaults to what the method
            wants, 1e-3 for sr and 1e-2 for minsr and cgsr
        nodal_cutoff (float): regularization distance for the nodal divergence of
            the parameter derivatives
        tau (float): optimization step size
        max_iterations (int): maximum number of optimization iterations
        overlap_penalty (np.ndarray): overlap penalty matrix with shape (nwf, nwf)
        npartitions (int): total number of partitions distributed among the threads
        client: an object with submit() functions that return futures
        overlap_thread_weight (list): a list of float that overrides the default thread weights (1 + wfi) / 2.0
        vmc_kwargs (dict): options for measurement `vmc`
        overlap_kwargs (dict): options for measurement `sample_overlap`
        norm_kwargs (dict): options for the normalization `sample_overlap`; defaults to
            overlap_kwargs with nblocks=2, since the norms only set a rescaling and
            only matter to within a factor of two or so
        initial_vmc_warmup_kwargs (dict): options for initial warmup `vmc`; an empty dictionary disables it
        initial_overlap_warmup_kwargs (dict): options for initial warmup `sample_overlap`; an empty dictionary disables it
        refresh_vmc_warmup_kwargs (dict): options for refresh warmup `vmc`; defaults to off, pass a dictionary to enable
        refresh_overlap_warmup_kwargs (dict): options for refresh warmup `sample_overlap`; defaults to off, pass a dictionary to enable
        all_configs (tuple): `(norm_configs, gradient_configs)`, a full set of configs to start the optimization

    Return:
        wfs (list): list of optimized wave functions
    """

    if initial_vmc_warmup_kwargs is None:
        initial_vmc_warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if initial_overlap_warmup_kwargs is None:
        initial_overlap_warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if refresh_vmc_warmup_kwargs is None:
        refresh_vmc_warmup_kwargs = {}
    if refresh_overlap_warmup_kwargs is None:
        refresh_overlap_warmup_kwargs = {}
    # minsr and cgsr both keep the derivatives per configuration, so they share
    # these defaults; sr averages them into dpidpj and wants more blocks
    per_sample = method in ("minsr", "cgsr")
    if  vmc_kwargs is None:
        if per_sample:
            vmc_kwargs = dict(nblocks=1, nsteps_per_block=10)
        else:
            vmc_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if not overlap_kwargs:
        if per_sample:
            overlap_kwargs = dict(nblocks=1, nsteps_per_block=10)
        else:
            overlap_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if norm_kwargs is None:
        norm_kwargs = dict(overlap_kwargs, nblocks=2)
    updater = build_updaters(transforms, enacc, method, eps, nodal_cutoff)
    if len(updater) != len(wfs):
        raise ValueError(
            f"got transforms for {len(updater)} states but {len(wfs)} wave functions"
        )
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
        if verbose:
            print(f"\n=== Iteration {i} ===", flush=True)
        # Refresh warmup for the normalization sampling
        norm_configs = _warmup_overlap(
            wfs,
            norm_configs,
            client=client,
            npartitions=npartitions,
            kwargs=refresh_overlap_warmup_kwargs,
        )
        # Norm measurement. The norms only set a rescaling of the wave functions,
        # so a couple of blocks is plenty; see norm_kwargs.
        _, data_unweighted, norm_configs = pyqmc.method.sample_many.sample_overlap(
            wfs,
            norm_configs,
            None,
            client=client,
            npartitions=npartitions,
            **norm_kwargs,
        )
        norm = np.mean(data_unweighted["overlap"], axis=0)
        if verbose:
            diag = np.array2string(
                np.real(norm.diagonal()), precision=4, suppress_small=True
            )
            print(f"  normalization: {diag}", flush=True)
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
                # the driver formats the diagnostics uniformly from the report,
                # so the updater does not print its own
                delta_p_start = time.perf_counter()
                dp, report = transform.delta_p(
                    [tau], avg, overlap_penalty, verbose=False
                )
                # the solve does not parallelize the way the sampling does, so
                # it is worth seeing next to the per-job sampling times
                report["delta_p_seconds"] = time.perf_counter() - delta_p_start
                if verbose:
                    label = f"wf {wfi}"
                    if len(transform_list) > 1:
                        label += f" sub {sub_iteration}"
                    line = (
                        f"  {label}: E = {float(np.real(avg['total'])):.6f}"
                        f" +/- {float(np.real(error['total'])):.6f}"
                    )
                    if wfi > 0:  # overlap with the states below this one
                        lower = np.array2string(
                            np.abs(avg["overlap"][wfi, :wfi]),
                            precision=4,
                            suppress_small=True,
                        )
                        line += f"   overlap with lower states {lower}"
                    print(line, flush=True)
                    print(f"      {_format_step_report(report)}", flush=True)
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
