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
"""The ensemble CG-SR update reproduces the ensemble minSR and SR updates.

The interesting part is the overlap penalty gradient, which does not lie in the
row space of the sampled derivatives. minSR reaches it through the Woodbury
branch of sr_solve; CG just adds it to the right hand side. They must agree.
"""

import numpy as np
import pytest

from pyqmc.method.ensemble_cgsr import CGSRWfbyWf
from pyqmc.method.ensemble_minsr import MinSRWfbyWf
from pyqmc.method.ensemble_optimization import (
    StochasticReconfigurationWfbyWf,
    make_updater,
)

RTOL = 1e-11


def make_data(rng, nsamples, nparams, nwf):
    """Per-sample derivatives and local energies for the top state, plus overlap
    sampling data for all nwf states."""
    dppsi = rng.normal(size=(nsamples, nparams))
    eloc = rng.normal(size=nsamples) - 1.0
    m = rng.normal(size=(nwf, nwf)) * 0.1
    weights = np.eye(nwf) + 0.5 * (m + m.T)
    wtdp = rng.normal(size=(nparams, nwf, nwf))
    nblocks = 2
    return dppsi, eloc, np.array([weights] * nblocks), np.array([wtdp] * nblocks)


def averaged(updater, dppsi, eloc, wtdp_blocks, overlap_blocks):
    sample1 = {
        "dppsi": dppsi,
        "total": eloc,
        "block_energy": np.array([np.mean(eloc)] * 2),
    }
    return updater.block_average(sample1, {"wtdp": wtdp_blocks}, overlap_blocks)[0]


@pytest.mark.parametrize("nsamples,nparams", [(20, 50), (60, 25)])
@pytest.mark.parametrize("nwf", [1, 3])
def test_ensemble_cgsr_equals_minsr(nsamples, nparams, nwf):
    rng = np.random.default_rng(seed=4321)
    eps, tau = 1e-2, 0.1
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, nsamples, nparams, nwf)
    penalty = np.ones((nwf, nwf)) * 0.5

    minsr = MinSRWfbyWf(None, None, eps=eps)
    dp_minsr = minsr.delta_p(
        [tau], averaged(minsr, dppsi, eloc, wtdp_blocks, overlap_blocks), penalty
    )[0][0]

    cg = CGSRWfbyWf(None, None, eps=eps, rtol=RTOL)
    dp_list, report = cg.delta_p(
        [tau], averaged(cg, dppsi, eloc, wtdp_blocks, overlap_blocks), penalty
    )

    assert report["cg_converged"]
    assert np.abs(dp_list[0] - dp_minsr).max() / np.abs(dp_minsr).max() < 1e-6


@pytest.mark.parametrize("nsamples,nparams", [(20, 50), (60, 25)])
@pytest.mark.parametrize("nwf", [1, 3])
def test_ensemble_cgsr_equals_sr(nsamples, nparams, nwf):
    """And therefore the explicit SR update, which builds the S matrix."""
    rng = np.random.default_rng(seed=4321)
    eps, tau = 1e-2, 0.1
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, nsamples, nparams, nwf)
    penalty = np.ones((nwf, nwf)) * 0.5

    sr = StochasticReconfigurationWfbyWf(None, None, eps=eps)
    sr_sample1 = {
        "total": np.array([np.mean(eloc)] * 2),
        "dppsi": np.array([np.mean(dppsi, axis=0)] * 2),
        "dpH": np.array([np.mean(eloc[:, np.newaxis] * dppsi, axis=0)] * 2),
        "dpidpj": np.array(
            [np.einsum("ij,ik->jk", dppsi, dppsi) / dppsi.shape[0]] * 2
        ),
    }
    sr_avg, _ = sr.block_average(sr_sample1, {"wtdp": wtdp_blocks}, overlap_blocks)
    dp_sr = sr.delta_p([tau], sr_avg, penalty)[0][0]

    cg = CGSRWfbyWf(None, None, eps=eps, rtol=RTOL)
    dp_cg = cg.delta_p(
        [tau], averaged(cg, dppsi, eloc, wtdp_blocks, overlap_blocks), penalty
    )[0][0]

    assert np.abs(dp_cg - dp_sr).max() / np.abs(dp_sr).max() < 1e-6


def test_overlap_penalty_changes_the_step():
    """So the equivalence above is exercising the penalty term, not a zero."""
    rng = np.random.default_rng(seed=99)
    nwf = 3
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, 30, 40, nwf)
    cg = CGSRWfbyWf(None, None, eps=1e-2, rtol=RTOL, warm_start=False)
    avg = averaged(cg, dppsi, eloc, wtdp_blocks, overlap_blocks)
    with_penalty = cg.delta_p([0.1], avg, np.ones((nwf, nwf)) * 0.5)[0][0]
    without = cg.delta_p([0.1], avg, np.zeros((nwf, nwf)))[0][0]
    assert not np.allclose(with_penalty, without)


def test_penalty_gradient_leaves_the_row_space():
    """The penalty gradient has a component no sample can resolve when there are
    fewer samples than parameters, and it comes back divided by eps. That is a
    property of the equations, so CG must reproduce it rather than smooth it."""
    rng = np.random.default_rng(seed=7)
    nwf, nsamples, nparams = 2, 20, 60
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, nsamples, nparams, nwf)
    penalty = np.ones((nwf, nwf)) * 0.5

    steps = {}
    for eps in [1e-2, 1e-3]:
        cg = CGSRWfbyWf(None, None, eps=eps, rtol=RTOL, warm_start=False)
        steps[eps] = cg.delta_p(
            [0.1], averaged(cg, dppsi, eloc, wtdp_blocks, overlap_blocks), penalty
        )[0][0]
        minsr = MinSRWfbyWf(None, None, eps=eps)
        dp_minsr = minsr.delta_p(
            [0.1], averaged(minsr, dppsi, eloc, wtdp_blocks, overlap_blocks), penalty
        )[0][0]
        assert np.abs(steps[eps] - dp_minsr).max() / np.abs(dp_minsr).max() < 1e-5
    # shrinking eps by ten amplifies the unresolved component, in both solvers
    assert np.linalg.norm(steps[1e-3]) > 5 * np.linalg.norm(steps[1e-2])


def test_warm_start_does_not_change_the_step():
    rng = np.random.default_rng(seed=13)
    nwf = 2
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, 80, 30, nwf)
    penalty = np.ones((nwf, nwf)) * 0.5

    cold = CGSRWfbyWf(None, None, eps=1e-2, rtol=RTOL, warm_start=False)
    warm = CGSRWfbyWf(None, None, eps=1e-2, rtol=RTOL, warm_start=True)
    avg_cold = averaged(cold, dppsi, eloc, wtdp_blocks, overlap_blocks)
    avg_warm = averaged(warm, dppsi, eloc, wtdp_blocks, overlap_blocks)
    for _ in range(3):  # the warm updater carries state between calls
        dp_cold = cold.delta_p([0.1], avg_cold, penalty)[0][0]
        dp_warm = warm.delta_p([0.1], avg_warm, penalty)[0][0]
    assert warm._previous_v is not None
    assert np.abs(dp_warm - dp_cold).max() / np.abs(dp_cold).max() < 1e-6


def test_method_selection():
    """Choosing CG-SR is a keyword on optimize_ensemble, not a different import."""
    assert isinstance(make_updater(None, None, method="cgsr"), CGSRWfbyWf)
    assert isinstance(make_updater(None, None, method="minsr"), MinSRWfbyWf)
    assert make_updater(None, None, method="cgsr").eps == 1e-2
    assert make_updater(None, None, method="cgsr", eps=1e-4).eps == 1e-4
    with pytest.raises(ValueError, match="cgsr"):
        make_updater(None, None, method="not_a_method")
