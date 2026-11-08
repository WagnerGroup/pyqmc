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

"""The ensemble updaters give the same step whichever single-state solver they
wrap, including the overlap penalty gradient, which does not lie in the row space
of the sampled derivatives."""

import numpy as np
import pytest

from pyqmc.method.ensemble_updaters import EnsembleUpdater, build_updaters, make_updater
from pyqmc.method.optimize import CGSRUpdate, MinSRUpdate, SRUpdate
from pyqmc.observables.stochastic_reconfiguration import sr_averages


def make_data(seed, nsamples, nparams, nwf):
    """Per-sample derivatives and local energies for the top state, plus
    overlap sampling data for all nwf states."""
    rng = np.random.default_rng(seed)
    dppsi = rng.normal(size=(nsamples, nparams))
    eloc = rng.normal(size=nsamples) - 1.0
    m = rng.normal(size=(nwf, nwf)) * 0.1
    weights = np.eye(nwf) + 0.5 * (m + m.T)
    wtdp = rng.normal(size=(nparams, nwf, nwf))
    return dppsi, eloc, np.array([weights] * 2), np.array([wtdp] * 2)


def energy_data(single, dppsi, eloc):
    """What single.sample would return for these samples."""
    if isinstance(single, SRUpdate):
        data = sr_averages({"total": eloc}, dppsi)
    else:
        data = {"dppsi": dppsi, "total": eloc}
    return {**data, "energy": np.mean(eloc), "energy_error": 0.0}


def step(single, data, penalty, tau=0.1):
    dppsi, eloc, overlap_blocks, wtdp_blocks = data
    up = EnsembleUpdater(single)
    avg, _ = up.block_average(energy_data(single, dppsi, eloc), {"wtdp": wtdp_blocks}, overlap_blocks)
    return up.delta_p([tau], avg, penalty)


SOLVERS = {
    "sr": lambda eps: SRUpdate(None, None, eps=eps),
    "minsr": lambda eps: MinSRUpdate(None, None, eps=eps),
    "cgsr": lambda eps: CGSRUpdate(None, None, eps=eps, rtol=1e-11, warm_start=False),
}


@pytest.mark.parametrize("nsamples,nparams", [(20, 50), (60, 25)])
@pytest.mark.parametrize("nwf", [1, 3])
def test_solvers_agree(nsamples, nparams, nwf):
    data = make_data(4321, nsamples, nparams, nwf)
    penalty = np.ones((nwf, nwf))
    steps = {k: step(make(1e-2), data, penalty)[0][0] for k, make in SOLVERS.items()}
    scale = np.abs(steps["sr"]).max()
    for k in ["minsr", "cgsr"]:
        assert np.abs(steps[k] - steps["sr"]).max() / scale < 1e-6, k


def test_single_state_matches_optimize():
    """With one state there is no penalty, so the ensemble step is the
    single-state step: the energy gradient carries the same factor of two."""
    dppsi, eloc, overlap_blocks, wtdp_blocks = make_data(1, 40, 10, 1)
    single = MinSRUpdate(None, None, eps=1e-2)
    v, _ = single.solve({"dppsi": dppsi, "total": eloc})
    dp = step(single, (dppsi, eloc, overlap_blocks, wtdp_blocks), np.ones((1, 1)), tau=0.1)[0][0]
    assert np.allclose(dp, -0.1 * v)


def test_overlap_penalty_changes_the_step():
    data = make_data(99, 30, 40, 3)
    for make in SOLVERS.values():
        with_penalty = step(make(1e-2), data, np.ones((3, 3)))[0][0]
        without = step(make(1e-2), data, np.zeros((3, 3)))[0][0]
        assert not np.allclose(with_penalty, without)


def test_penalty_gradient_leaves_the_row_space():
    """With fewer samples than parameters, part of the penalty gradient lies
    outside the sampled rows and comes back divided by eps. That is a property of
    the equations, so every solver must reproduce it."""
    data = make_data(7, 20, 60, 2)
    penalty = np.ones((2, 2))
    norms = {}
    for eps in [1e-2, 1e-3]:
        steps = {k: step(make(eps), data, penalty)[0][0] for k, make in SOLVERS.items()}
        for k in ["minsr", "cgsr"]:
            assert np.abs(steps[k] - steps["sr"]).max() / np.abs(steps["sr"]).max() < 1e-5
        norms[eps] = np.linalg.norm(steps["sr"])
    assert norms[1e-3] > 5 * norms[1e-2]


def test_cg_warm_start_does_not_change_the_step():
    data = make_data(13, 80, 30, 2)
    penalty = np.ones((2, 2))
    cold = EnsembleUpdater(CGSRUpdate(None, None, eps=1e-2, rtol=1e-11, warm_start=False))
    warm = EnsembleUpdater(CGSRUpdate(None, None, eps=1e-2, rtol=1e-11, warm_start=True))
    dppsi, eloc, overlap_blocks, wtdp_blocks = data
    for _ in range(3):  # the warm updater carries state between calls
        dps = []
        for up in [cold, warm]:
            avg, _ = up.block_average(energy_data(up.single, dppsi, eloc), {"wtdp": wtdp_blocks}, overlap_blocks)
            dps.append(up.delta_p([0.1], avg, penalty)[0][0])
    assert warm.single._previous_v is not None
    assert np.abs(dps[1] - dps[0]).max() / np.abs(dps[0]).max() < 1e-6


def test_method_selection():
    class FakeEnacc:
        pass

    class FakeTransform:
        nparams = 3

    enacc, transform = FakeEnacc(), FakeTransform()
    for method, cls, eps in [("sr", SRUpdate, 1e-3), ("minsr", MinSRUpdate, 1e-2), ("cgsr", CGSRUpdate, 1e-2)]:
        up = make_updater(transform, enacc, method)
        assert type(up.single) is cls and up.eps == eps
    assert make_updater(transform, enacc, "minsr", eps=0.5).eps == 0.5
    sr = make_updater(transform, enacc, "sr", eps=0.25, nodal_cutoff=0.01)
    assert sr.single._accumulator.eps == 0.25 and sr.single._accumulator.nodal_cutoff == 0.01
    with pytest.raises(ValueError, match="Unknown method"):
        make_updater(transform, enacc, "nonsense")

    # a flat list of transforms is one per state; nesting adds sub-iterations
    assert [len(s) for s in build_updaters([transform, transform], enacc, "minsr")] == [1, 1]
    assert [len(s) for s in build_updaters([[transform, transform], [transform]], enacc, "minsr")] == [2, 1]
    # already-built updaters pass through untouched
    built = make_updater(transform, enacc, "cgsr")
    assert build_updaters([built], enacc)[0][0] is built
    # one accumulator per state is allowed; a mismatched count is an error
    assert build_updaters([transform, transform], [enacc, FakeEnacc()], "sr")[0][0].enacc is enacc
    with pytest.raises(ValueError, match="energy accumulators"):
        build_updaters([transform, transform], [enacc], "sr")
    with pytest.raises(ValueError, match="enacc is required"):
        build_updaters([transform], None, "sr")
