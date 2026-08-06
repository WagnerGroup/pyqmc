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

from pyqmc.configurations.coord import OpenConfigs, PeriodicConfigs
from pyqmc.method.ensemble_minsr import (
    MinSRWfbyWf,
    load_configs_ensemble,
    save_configs_ensemble,
)
from pyqmc.method.ensemble_optimization_wfbywf import StochasticReconfigurationWfbyWf


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


def make_configs_ensemble(rng, nwf, nsub, nconfig=5, periodic=False):
    """A distinct walker population for every state, sub-iteration, and thread."""
    lvecs = np.eye(3) * 4.0

    def one():
        c = rng.normal(size=(nconfig, 2, 3))
        return PeriodicConfigs(c, lvecs) if periodic else OpenConfigs(c)

    return [[[one() for _ in range(2)] for _ in range(nsub)] for _ in range(nwf)]


@pytest.mark.parametrize("periodic", [False, True])
def test_configs_ensemble_roundtrip(tmp_path, periodic):
    """Every walker population survives a save/load cycle, which is what makes a
    restart resume with the populations it equilibrated."""
    rng = np.random.default_rng(seed=7)
    nwf, nsub = 3, 2
    stored = make_configs_ensemble(rng, nwf, nsub, periodic=periodic)
    hdf_file = str(tmp_path / "configs.hdf5")
    save_configs_ensemble(hdf_file, stored)

    loaded = make_configs_ensemble(rng, nwf, nsub, periodic=periodic)
    with h5py.File(hdf_file, "r") as hdf:
        load_configs_ensemble(hdf, loaded)

    # configs datasets are created without a dtype, so h5py stores them as
    # float32; this is how every pyqmc restart file already stores walkers
    for wfi in range(nwf):
        for sub in range(nsub):
            for thread in range(2):
                assert np.allclose(
                    stored[wfi][sub][thread].configs,
                    loaded[wfi][sub][thread].configs,
                    rtol=1e-6,
                ), (wfi, sub, thread)
                if periodic:
                    assert np.array_equal(
                        stored[wfi][sub][thread].wrap, loaded[wfi][sub][thread].wrap
                    )
    # populations really are distinct, so the test above is not comparing copies
    assert not np.array_equal(stored[0][0][0].configs, stored[0][0][1].configs)
    assert not np.array_equal(stored[0][0][0].configs, stored[1][0][0].configs)


def test_configs_ensemble_missing_group_falls_back(tmp_path, caplog):
    """A file written without the per-thread populations leaves them as passed
    in rather than failing, so old restart files still work."""
    rng = np.random.default_rng(seed=8)
    hdf_file = str(tmp_path / "old_style.hdf5")
    original = make_configs_ensemble(rng, 1, 1)
    # a file with only the top-level configs, as the SR version writes
    with h5py.File(hdf_file, "a") as hdf:
        original[0][0][0].initialize_hdf(hdf)
        original[0][0][0].to_hdf(hdf)

    loaded = make_configs_ensemble(rng, 1, 1)
    expected = [c.configs.copy() for c in loaded[0][0]]
    with h5py.File(hdf_file, "r") as hdf:
        load_configs_ensemble(hdf, loaded)
    for got, want in zip(loaded[0][0], expected):
        assert np.array_equal(got.configs, want)
    assert "no stored walkers" in caplog.text


def test_configs_ensemble_shape_mismatch(tmp_path):
    """Restarting with a different number of walkers is an error, not silent
    corruption."""
    rng = np.random.default_rng(seed=9)
    hdf_file = str(tmp_path / "configs.hdf5")
    save_configs_ensemble(hdf_file, make_configs_ensemble(rng, 1, 1, nconfig=5))
    wrong = make_configs_ensemble(rng, 1, 1, nconfig=7)
    with h5py.File(hdf_file, "r") as hdf:
        with pytest.raises(ValueError, match="same number of walkers"):
            load_configs_ensemble(hdf, wrong)
