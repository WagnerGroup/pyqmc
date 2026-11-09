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


r"""Scheduling the samplings of one ensemble optimization round.

Each round samples the state on its own (an "energy" job) for every state and
sub-iteration, and the overlap distribution of the state with the states below
it (an "overlap" job) for every state but the lowest, which has no overlap
penalty. The jobs are independent, so they run at the
same time on threads, each submitting its own partitions to the client. The
partitions are divided among the jobs by weight (:func:`round_to_fixed_sum`),
jobs are submitted longest first (:func:`_build_sampling_jobs`), and each job
gets its own child of the random generator, spawned before any thread starts.

Without a client the jobs share wave-function state, so they run one at a time.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import numpy as np

import pyqmc.method.mc
import pyqmc.method.sample_many
import pyqmc.rng


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


def warmup_overlap(wfs, configs, client, npartitions, kwargs, rng):
    if not kwargs or kwargs.get("nblocks", 1) <= 0:
        return configs
    _, _, configs = pyqmc.method.sample_many.sample_overlap(
        wfs,
        configs,
        None,
        client=client,
        npartitions=npartitions,
        rng=rng,
        **kwargs,
    )
    return configs


def sampling_kinds(wfi):
    """The samplings state `wfi` needs: state 0 has no lower states to overlap."""
    return ("energy",) if wfi == 0 else ("energy", "overlap")


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
    gradient_configs, vmc_kwargs, overlap_kwargs, overlap_thread_weight=None, measure=False
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

    A warmup (`measure` False) skips a sampling whose kwargs are empty or have
    nblocks <= 0. A measurement always samples; empty kwargs mean the sampler's
    defaults.
    """
    jobs = []
    if measure or _sampling_requested(vmc_kwargs):
        sweeps = _sweeps(vmc_kwargs)
        for wfi, state_configs in enumerate(gradient_configs):
            for sub_iteration in range(len(state_configs)):
                jobs.append(_SamplingJob("energy", wfi, sub_iteration, 1.0, sweeps))
    if measure or _sampling_requested(overlap_kwargs):
        sweeps = _sweeps(overlap_kwargs)
        for wfi, state_configs in enumerate(gradient_configs):
            if "overlap" not in sampling_kinds(wfi):
                continue
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
    vmc_kwargs, overlap_kwargs, rng,
):
    """Submit one job. `updater` of None means a warmup: propagate only, measure
    nothing.

    :parameter npartitions: partitions for this job alone, not the total
    :parameter rng: a Generator for this job alone, not shared with other threads
    :returns: a future whose result is (sampling output, seconds)
    """
    configs = gradient_configs[job.wfi][job.sub_iteration][job.kind]
    transform = None if updater is None else updater[job.wfi][job.sub_iteration]
    # the samplers print progress per block, which is unreadable interleaved
    # across threads; the summary from _format_sampling_summary replaces it
    common = dict(client=client, npartitions=npartitions, rng=rng)

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
    rng=None,
):
    """
    Sample all vmc and prefix-overlap configurations. It is a warmup run when `updater` is `None`
    (no gradient accumulators and all measurements are discarded). Otherwise, the same threaded
    scheduler performs the measured gradient sampling.

    Sampling jobs are assigned partitions according to their weights. Every active sampling job
    receives at least one partition, so when ``npartitions`` is smaller than the number of active
    jobs, the total assigned partitions is the number of active jobs. Jobs are submitted longest
    first; see :func:`_build_sampling_jobs`.

    Each job gets its own child of `rng`, spawned here before any thread starts, so
    the streams do not depend on the order in which the threads run.
    """
    energy_data = [[None for _ in state] for state in gradient_configs]
    overlap_data_weighted = [[None for _ in state] for state in gradient_configs]
    overlap_data_unweighted = [[None for _ in state] for state in gradient_configs]

    jobs = _build_sampling_jobs(
        gradient_configs, vmc_kwargs, overlap_kwargs, overlap_thread_weight,
        measure=updater is not None,
    )
    if not jobs:
        return (
            energy_data,
            overlap_data_weighted,
            overlap_data_unweighted,
            gradient_configs,
        )

    children = pyqmc.rng.get_rng(rng).spawn(len(jobs))
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
                partitions[i], vmc_kwargs, overlap_kwargs, children[i],
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
    rng=None,
):
    """
    Evaluate parameter gradients for each state and sub-iteration using threads.
    It runs Monte Carlo evaluations for all the states asynchronously and gathers the results as they complete.
    Optional refresh sampling is performed first and its measurements are discarded.
    Samplings get parallelized using weighted partitions, which can be overridden by `overlap_thread_weight`.

    Args:
        gradient_configs (list[list[dict]]): nested list of configs indexed by state, sub-iteration, then kind
            ("energy", and "overlap" for states above 0; see sampling_kinds)
        See ``optimize_ensemble`` for details of other parameters.

    Return:
        energy_data (list): nested list of vmc outputs indexed by state then by sub-iteration
        overlap_data_weighted (list): nested list of weighted sample_overlap outputs indexed by state then by
            sub-iteration; None for state 0
        overlap_data_unweighted (list): the same for the unweighted outputs
        gradient_configs (list): updated configurations with the same structure
    """
    rng = pyqmc.rng.get_rng(rng)
    if vmc_kwargs is None:
        vmc_kwargs = dict(nblocks=10, nsteps_per_block=10)
    if overlap_kwargs is None:
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
            rng=rng,
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
        rng=rng,
    )
