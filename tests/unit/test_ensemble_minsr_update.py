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
"""Check that the ensemble minSR update reproduces the ensemble SR update,
including the overlap penalty gradient, which does not live in the row space of
the sampled derivatives and so goes through the Woodbury path of sr_solve."""

import h5py
import numpy as np
import pytest

from pyqmc.method.ensemble_minsr import MinSRWfbyWf
from pyqmc.method.ensemble_optimization import StochasticReconfigurationWfbyWf


def make_data(rng, nsamples, nparams, nwf):
    """Per-sample derivatives and local energies for the top state, plus overlap
    sampling data for all nwf states."""
    dppsi = rng.normal(size=(nsamples, nparams))
    eloc = rng.normal(size=nsamples) - 1.0

    # a plausible overlap matrix: symmetric, positive diagonal near 1
    m = rng.normal(size=(nwf, nwf)) * 0.1
    weights = np.eye(nwf) + 0.5 * (m + m.T)
    wtdp = rng.normal(size=(nparams, nwf, nwf))
    # two identical blocks, so that block averages are exact and sem is defined
    nblocks = 2
    overlap_blocks = np.array([weights] * nblocks)
    wtdp_blocks = np.array([wtdp] * nblocks)
    return dppsi, eloc, overlap_blocks, wtdp_blocks


def sr_sample1(dppsi, eloc, nblocks=2):
    """The block-averaged quantities the SR accumulator would produce from the
    same samples."""
    return {
        "total": np.array([np.mean(eloc)] * nblocks),
        "dppsi": np.array([np.mean(dppsi, axis=0)] * nblocks),
        "dpH": np.array([np.mean(eloc[:, np.newaxis] * dppsi, axis=0)] * nblocks),
        "dpidpj": np.array(
            [np.einsum("ij,ik->jk", dppsi, dppsi) / dppsi.shape[0]] * nblocks
        ),
    }


@pytest.mark.parametrize("nsamples,nparams", [(20, 50), (60, 25)])
@pytest.mark.parametrize("nwf", [1, 3])
def test_ensemble_minsr_equals_sr(nsamples, nparams, nwf):
    rng = np.random.default_rng(seed=4321)
    eps, tau = 1e-2, 0.1
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, nsamples, nparams, nwf)
    penalty = np.ones((nwf, nwf)) * 0.5

    sr = StochasticReconfigurationWfbyWf(None, None, eps=eps)
    sr_avg, _ = sr.block_average(
        sr_sample1(dppsi, eloc), {"wtdp": wtdp_blocks}, overlap_blocks
    )
    dp_sr = sr.delta_p([tau], sr_avg, penalty)[0][0]

    minsr = MinSRWfbyWf(None, None, eps=eps)
    minsr_avg, minsr_err = minsr.block_average(
        {
            "dppsi": dppsi,
            "total": eloc,
            "block_energy": np.array([np.mean(eloc)] * 2),
        },
        {"wtdp": wtdp_blocks},
        overlap_blocks,
    )
    dp_list, report = minsr.delta_p([tau], minsr_avg, penalty)
    dp_minsr = dp_list[0]

    assert np.allclose(dp_sr, dp_minsr, atol=1e-8), np.abs(dp_sr - dp_minsr).max()
    assert np.isclose(minsr_avg["total"], np.mean(eloc))
    assert report["SRdot"] > 0
    assert minsr_err["total"] >= 0


def test_ensemble_minsr_overlap_term_matters():
    """The overlap penalty actually changes the step, so the equivalence above is
    testing the Woodbury path and not a zero contribution."""
    rng = np.random.default_rng(seed=99)
    nwf = 3
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(rng, 30, 40, nwf)
    sample1 = {
        "dppsi": dppsi,
        "total": eloc,
        "block_energy": np.array([np.mean(eloc)] * 2),
    }

    minsr = MinSRWfbyWf(None, None, eps=1e-2)
    avg, _ = minsr.block_average(sample1, {"wtdp": wtdp_blocks}, overlap_blocks)
    with_penalty = minsr.delta_p([0.1], avg, np.ones((nwf, nwf)) * 0.5)[0][0]
    no_penalty = minsr.delta_p([0.1], avg, np.zeros((nwf, nwf)))[0][0]
    assert not np.allclose(with_penalty, no_penalty)


def test_method_selection():
    """optimize_ensemble builds the updaters, so choosing an algorithm is a
    keyword rather than importing a different class from a different module."""
    from pyqmc.method.ensemble_optimization import build_updaters, make_updater

    class FakeEnacc:
        def keys(self):
            return set()

        def shapes(self):
            return {}

    class FakeTransform:
        nparams = 3

    enacc, transform = FakeEnacc(), FakeTransform()

    assert isinstance(
        make_updater(transform, enacc, "sr"), StochasticReconfigurationWfbyWf
    )
    assert isinstance(make_updater(transform, enacc, "minsr"), MinSRWfbyWf)
    with pytest.raises(ValueError, match="Unknown method"):
        make_updater(transform, enacc, "nonsense")

    # each method keeps its own default regularization, and eps overrides it
    assert make_updater(transform, enacc, "sr").eps == 1e-3
    assert make_updater(transform, enacc, "minsr").eps == 1e-2
    assert make_updater(transform, enacc, "minsr", eps=0.5).eps == 0.5
    # eps reaches the solve; it used to land in nodal_cutoff instead
    sr = make_updater(transform, enacc, "sr", eps=0.25, nodal_cutoff=0.01)
    assert sr.onewf().eps == 0.25 and sr.onewf().nodal_cutoff == 0.01

    # a flat list of transforms is one per state; nesting adds sub-iterations
    flat = build_updaters([transform, transform], enacc, "minsr")
    assert [len(state) for state in flat] == [1, 1]
    nested = build_updaters([[transform, transform], [transform]], enacc, "minsr")
    assert [len(state) for state in nested] == [2, 1]

    # already-built updaters pass through untouched
    built = MinSRWfbyWf(enacc, transform)
    assert build_updaters([built], enacc)[0][0] is built

    # one accumulator per state is allowed; a mismatched count is an error
    per_state = build_updaters([transform, transform], [enacc, FakeEnacc()], "sr")
    assert per_state[0][0].enacc is enacc
    with pytest.raises(ValueError, match="energy accumulators"):
        build_updaters([transform, transform], [enacc], "sr")
    with pytest.raises(ValueError, match="enacc is required"):
        build_updaters([transform], None, "sr")
