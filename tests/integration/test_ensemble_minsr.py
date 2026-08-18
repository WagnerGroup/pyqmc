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

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import copy

import h5py
import numpy as np
import pytest

import pyqmc.api as pyq
from pyqmc.method.ensemble_optimization import (
    load_all_configs,
    make_updater,
    optimize_ensemble,
)
from pyqmc.observables.accumulators import LinearTransform

# small samplings so the test is quick; nblocks=1 for the energy sampling because
# the minSR kernel is square in the total number of samples
KWS = dict(
    vmc_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    overlap_kwargs={"nblocks": 2, "nsteps_per_block": 3},
    initial_vmc_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    initial_overlap_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    refresh_vmc_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 2},
    refresh_overlap_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 2},
)


def make_ensemble(mol, mf, mc, nstates):
    """A CASCI state for each root, with its determinant coefficients free."""
    mcs = [copy.copy(mc) for _ in range(nstates)]
    wfs = []
    transforms = []
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]
        wf, to_opt = pyq.generate_slater(
            mol, mf, mc=mcs[i], optimize_determinants=True, tol=1e-20
        )
        wfs.append(wf)
        transforms.append(LinearTransform(wf.parameters, to_opt))
    return wfs, transforms


@pytest.mark.slow
def test_ensemble_minsr(H2_casci, tmp_path):
    """Two states optimized together through the shared ensemble driver: energies
    are recorded for both, the states stay distinct, and the checkpoint holds
    every walker population."""
    mol, mf, mc = H2_casci
    hdf_file = str(tmp_path / "ensemble_minsr.hdf5")

    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    configs = pyq.initial_guess(mol, 200)

    wfs = optimize_ensemble(
        wfs, configs, transforms, hdf_file, enacc=pyq.EnergyAccumulator(mol),
        method="minsr", tau=0.1, max_iterations=3, npartitions=1, verbose=False, **KWS
    )

    with h5py.File(hdf_file, "r") as hdf:
        assert {"energy0", "energy1", "overlap0", "overlap1", "iteration"} <= set(hdf)
        assert hdf.attrs["tau"] == 0.1
        assert list(hdf["wavefunction"][()]) == [0, 1] * 3
        assert list(hdf["iteration"][()]) == [0, 0, 1, 1, 2, 2]
        # energy{i} is only written on the rows belonging to state i
        e0 = hdf["energy0"][()]
        e1 = hdf["energy1"][()]
        err0 = hdf["energy_error0"][()]
        overlap1 = hdf["overlap1"][()]
        assert len(e0) == 3 and len(e1) == 3
        assert np.all(np.isfinite(e0)) and np.all(np.isfinite(e1))
        assert np.all(err0 > 0)
        assert e1[-1] > e0[-1]  # the excited state stays above the ground state
        norm = np.sqrt(np.abs(overlap1[-1][0, 0] * overlap1[-1][1, 1]))
        assert np.abs(overlap1[-1][1, 0]) / norm < 0.5  # states stay distinct

    # every walker population is checkpointed, and they are genuinely different
    wfs2, transforms2 = make_ensemble(mol, mf, mc, nstates=2)
    fresh = pyq.initial_guess(mol, 200)
    with h5py.File(hdf_file, "r") as hdf:
        norm_configs, gradient_configs = load_all_configs(hdf, fresh, [[t] for t in transforms2])
    populations = [norm_configs.configs] + [
        gradient_configs[wfi][0][kind].configs
        for wfi in range(2)
        for kind in ("energy", "overlap")
    ]
    assert len(populations) == 5
    for i, a in enumerate(populations):
        for b in populations[i + 1 :]:
            assert not np.array_equal(a, b)

    # restarting continues from the recorded iteration
    optimize_ensemble(
        wfs2, fresh, transforms2, hdf_file, enacc=pyq.EnergyAccumulator(mol),
        method="minsr", tau=0.1, max_iterations=4, npartitions=1, verbose=False, **KWS
    )
    with h5py.File(hdf_file, "r") as hdf:
        assert list(hdf["iteration"][()]) == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.slow
def test_ensemble_minsr_with_client(H2_casci, tmp_path):
    """With a client the sampling happens in worker processes, so the driver
    runs the state threads concurrently instead of one at a time."""
    from concurrent.futures import ProcessPoolExecutor

    mol, mf, mc = H2_casci
    hdf_file = str(tmp_path / "ensemble_minsr_client.hdf5")

    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    ncore = 2
    with ProcessPoolExecutor(max_workers=ncore) as client:
        optimize_ensemble(
            wfs, pyq.initial_guess(mol, 200), transforms, hdf_file,
            enacc=pyq.EnergyAccumulator(mol), method="minsr", tau=0.1,
            max_iterations=2, client=client, npartitions=ncore, verbose=False, **KWS
        )
    with h5py.File(hdf_file, "r") as hdf:
        assert np.all(np.isfinite(hdf["energy0"][()]))
        assert np.all(np.isfinite(hdf["energy1"][()]))
        assert len(hdf["energy1"][()]) == 2


@pytest.mark.slow
def test_ensemble_minsr_matches_sr(H2_casci):
    """The ensemble minSR step reproduces the ensemble SR step on real sampled
    data, including the overlap penalty."""
    from pyqmc.method.ensemble_optimization import StochasticReconfigurationWfbyWf

    mol, mf, mc = H2_casci
    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    configs = pyq.initial_guess(mol, 300)
    wfi = 1  # the excited state, which carries the overlap penalty
    up = make_updater(transforms[wfi], pyq.EnergyAccumulator(mol), method="minsr")
    eps, tau = 1e-2, 0.1
    penalty = np.ones((2, 2)) * 0.5

    # one energy sample and one overlap sample, shared by both updates
    sample1, configs = up.sample_energy(wfs[wfi], configs, nsteps_per_block=3)
    weighted, unweighted, _ = up.sample_overlap(
        wfs, configs, nblocks=2, nsteps_per_block=3
    )

    up.eps = eps
    avg, _ = up.block_average(sample1, weighted, unweighted["overlap"])
    dp_minsr = up.delta_p([tau], avg, penalty)[0][0]

    # the same data, averaged the way the SR accumulator would have
    dppsi, eloc = sample1["dppsi"], sample1["total"]
    sr = StochasticReconfigurationWfbyWf(up.enacc, up.transform, eps=eps)
    sr_sample1 = {
        "total": np.array([np.mean(eloc)] * 2),
        "dppsi": np.array([np.mean(dppsi, axis=0)] * 2),
        "dpH": np.array([np.mean(eloc[:, np.newaxis] * dppsi, axis=0)] * 2),
        "dpidpj": np.array(
            [np.einsum("ij,ik->jk", dppsi, dppsi) / dppsi.shape[0]] * 2
        ),
    }
    sr_avg, _ = sr.block_average(sr_sample1, weighted, unweighted["overlap"])
    dp_sr = sr.delta_p([tau], sr_avg, penalty)[0][0]

    assert np.allclose(dp_sr, dp_minsr, atol=1e-8), np.abs(dp_sr - dp_minsr).max()


@pytest.mark.slow
def test_overlap_derivatives_match_accumulator(H2_casci):
    """The snapshot estimator of the overlap gradient computes exactly the
    quantity the every-step accumulator averages.

    minSR evaluates the weighted derivatives once per block instead of at every
    Metropolis step, which is the same unbiased average of the same
    per-configuration quantity -- just far fewer pgradient calls. On a fixed set
    of configurations the two must agree to machine precision.
    """
    from pyqmc.method.ensemble_minsr import (
        MinSRWfbyWf,
        overlap_derivatives_worker,
    )
    from pyqmc.method.ensemble_optimization import StochasticReconfigurationWfbyWf
    from pyqmc.method.sample_many import compute_weights

    mol, mf, mc = H2_casci
    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    configs = pyq.initial_guess(mol, 50)

    enacc = pyq.EnergyAccumulator(mol)
    up = MinSRWfbyWf(enacc, transforms[-1])
    sr = StochasticReconfigurationWfbyWf(enacc, transforms[-1])

    for wf in wfs:
        wf.recompute(configs)
    accumulated = sr.avg(configs, wfs, compute_weights(wfs))["wtdp"]

    total, nconfig = overlap_derivatives_worker(wfs, configs, up.transform)
    assert nconfig == configs.configs.shape[0]
    assert np.allclose(total / nconfig, accumulated, atol=1e-12)


@pytest.mark.slow
def test_sample_overlap_shapes(H2_casci):
    """MinSRWfbyWf.sample_overlap returns what block_average expects: one wtdp
    per block, and the overlap matrix still accumulated at every step."""
    from pyqmc.method.ensemble_minsr import MinSRWfbyWf

    mol, mf, mc = H2_casci
    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    configs = pyq.initial_guess(mol, 50)

    up = MinSRWfbyWf(pyq.EnergyAccumulator(mol), transforms[-1])
    nblocks = 3
    weighted, unweighted, configs = up.sample_overlap(
        wfs, configs, nblocks=nblocks, nsteps_per_block=2
    )
    nparams = transforms[-1].nparams
    assert weighted["wtdp"].shape == (nblocks, nparams, 2, 2)
    assert unweighted["overlap"].shape == (nblocks, 2, 2)
    assert unweighted["acceptance"].shape == (nblocks,)
    assert np.all(np.isfinite(weighted["wtdp"]))
