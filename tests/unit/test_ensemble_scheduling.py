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
"""Scheduling and reporting helpers for the ensemble driver."""

import warnings

import numpy as np
import pytest

from pyqmc.method.ensemble_optimization import (
    _block_sem,
    _build_sampling_jobs,
    _format_step_report,
    _sweeps,
    round_to_fixed_sum,
)

VMC = {"nblocks": 1, "nsteps_per_block": 10}
OVERLAP = {"nblocks": 1, "nsteps_per_block": 10}


def configs_for(nstates, nsub=1):
    """The shape _build_sampling_jobs reads: state, then sub-iteration."""
    return [[{"energy": None, "overlap": None} for _ in range(nsub)] for _ in range(nstates)]


def test_longest_job_is_submitted_first():
    """The prefix overlap for the highest state propagates the most wave
    functions, so it must not be left to start last."""
    jobs = _build_sampling_jobs(configs_for(3), VMC, OVERLAP)
    assert (jobs[0].kind, jobs[0].wfi) == ("overlap", 2)
    # overlap jobs descend by state, and the cheap energy jobs come after
    kinds = [(j.kind, j.wfi) for j in jobs]
    assert kinds[:3] == [("overlap", 2), ("overlap", 1), ("overlap", 0)]
    assert all(kind == "energy" for kind, _ in kinds[3:])


def test_jobs_are_sorted_by_descending_cost():
    jobs = _build_sampling_jobs(configs_for(4), VMC, OVERLAP)
    costs = [j.cost for j in jobs]
    assert costs == sorted(costs, reverse=True)


def test_every_job_is_present_exactly_once():
    """Reordering must not drop or duplicate work."""
    nstates, nsub = 3, 2
    jobs = _build_sampling_jobs(configs_for(nstates, nsub), VMC, OVERLAP)
    seen = {(j.kind, j.wfi, j.sub_iteration) for j in jobs}
    assert len(jobs) == 2 * nstates * nsub == len(seen)


def test_cost_counts_the_wave_functions_propagated():
    """Overlap sampling for state wfi moves wfi+1 wave functions at once."""
    jobs = _build_sampling_jobs(configs_for(3), VMC, {"nblocks": 2, "nsteps_per_block": 5})
    overlap = {j.wfi: j.cost for j in jobs if j.kind == "overlap"}
    assert overlap == {0: 10, 1: 20, 2: 30}
    energy = {j.cost for j in jobs if j.kind == "energy"}
    assert energy == {_sweeps(VMC)}


def test_a_longer_overlap_still_outranks_a_longer_energy_sampling():
    """The cost estimate uses the actual kwargs, so it tracks whichever sampling
    was configured to be the expensive one."""
    jobs = _build_sampling_jobs(
        configs_for(1),
        {"nblocks": 20, "nsteps_per_block": 10},  # a long vmc sampling
        {"nblocks": 1, "nsteps_per_block": 10},  # a short overlap sampling
    )
    assert jobs[0].kind == "energy"


def test_only_the_requested_samplings_are_scheduled():
    assert all(j.kind == "overlap" for j in _build_sampling_jobs(configs_for(2), {}, OVERLAP))
    assert all(j.kind == "energy" for j in _build_sampling_jobs(configs_for(2), VMC, {}))
    assert _build_sampling_jobs(configs_for(2), {}, {}) == []


def test_partitions_follow_the_weights_in_submission_order():
    """round_to_fixed_sum is indexed positionally, so the weights it is handed
    have to be the reordered ones."""
    jobs = _build_sampling_jobs(configs_for(3), VMC, OVERLAP)
    partitions = round_to_fixed_sum(np.array([j.weight for j in jobs]), 12)
    assert sum(partitions) == 12
    by_job = {(j.kind, j.wfi): p for j, p in zip(jobs, partitions)}
    # overlap weight rises with the state, so its partition count must not fall
    assert by_job[("overlap", 0)] <= by_job[("overlap", 1)] <= by_job[("overlap", 2)]


def test_overlap_thread_weight_overrides_the_partition_share():
    jobs = _build_sampling_jobs(configs_for(3), VMC, OVERLAP, overlap_thread_weight=[5.0, 1.0, 1.0])
    weights = {j.wfi: j.weight for j in jobs if j.kind == "overlap"}
    assert weights == {0: 5.0, 1: 1.0, 2: 1.0}
    # it changes partitions, not the ordering, which still follows cost
    assert (jobs[0].kind, jobs[0].wfi) == ("overlap", 2)


def test_format_step_report_hides_flags_that_are_fine():
    line = _format_step_report(
        {"pgrad": 0.5, "SRdot": 0.9, "cg_iterations": 3, "cg_converged": True}
    )
    assert "|grad| = 0.5" in line and "grad.step = 0.9" in line
    assert "cg_iterations = 3" in line
    assert "cg_converged" not in line  # nothing to report


def test_format_step_report_shows_flags_that_are_not():
    line = _format_step_report({"pgrad": 0.5, "cg_converged": False})
    assert "cg_converged = False" in line


def test_format_step_report_shows_timings_as_times():
    """delta_p does not parallelize the way the sampling does, so its cost is
    worth seeing; any updater can report one the same way."""
    line = _format_step_report({"pgrad": 0.5, "delta_p_seconds": 1.25})
    assert "delta_p = 1.250s" in line
    assert "delta_p_seconds" not in line


def test_format_step_report_skips_non_scalars():
    line = _format_step_report({"pgrad": 0.5, "vector": np.arange(3), "name": "sr"})
    assert "vector" not in line and "name" not in line


def test_block_sem_is_quiet_for_a_single_block():
    """The per-sample methods default to one block, where the standard error is
    undefined; it must be nan without a RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        err = _block_sem(np.ones((1, 4, 2)))
    assert err.shape == (4, 2)
    assert np.all(np.isnan(err))


def test_block_sem_matches_scipy_for_several_blocks():
    import scipy.stats

    rng = np.random.default_rng(0)
    blocks = rng.normal(size=(5, 3, 2))
    assert np.allclose(_block_sem(blocks), scipy.stats.sem(blocks, axis=0))
