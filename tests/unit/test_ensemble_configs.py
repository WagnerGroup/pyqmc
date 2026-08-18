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
"""Ensemble checkpoints hold every walker population, and give them back exactly.

A restart that shared one population between the states, or that rounded the
walkers to single precision on the way to disk, would not be caught by anything
that only checks the file reloads.
"""

import h5py
import numpy as np
import pytest

from pyqmc.configurations.coord import OpenConfigs, PeriodicConfigs
from pyqmc.method.ensemble_optimization import hdf_save, load_all_configs


def make_configs(rng, periodic, nconfig=5):
    coords = rng.normal(size=(nconfig, 2, 3))
    if periodic:
        return PeriodicConfigs(coords, np.eye(3) * 4.0)
    return OpenConfigs(coords)


class FakeWF:
    def __init__(self, rng):
        self.parameters = {"det_coeff": rng.normal(size=3)}


@pytest.mark.parametrize("periodic", [False, True])
def test_all_configs_round_trip(tmp_path, periodic):
    rng = np.random.default_rng(seed=11)
    nwf, nsub = 2, 2
    updater = [[None] * nsub for _ in range(nwf)]
    wfs = [FakeWF(rng) for _ in range(nwf)]

    norm_configs = make_configs(rng, periodic)
    gradient_configs = [
        [
            {kind: make_configs(rng, periodic) for kind in ("energy", "overlap")}
            for _ in range(nsub)
        ]
        for _ in range(nwf)
    ]
    hdf_file = str(tmp_path / "ensemble.hdf5")
    hdf_save(
        hdf_file, {"iteration": 0}, {"tau": 0.1}, wfs, norm_configs, gradient_configs
    )

    fresh = make_configs(rng, periodic)
    with h5py.File(hdf_file, "r") as hdf:
        loaded_norm, loaded_gradient = load_all_configs(hdf, fresh, updater)
        assert hdf["all_configs/normalization/configs"].dtype == np.float64
        assert hdf.attrs["tau"] == 0.1

    # exact, not merely close: configurations are stored in double precision
    assert np.array_equal(loaded_norm.configs, norm_configs.configs)
    for wfi in range(nwf):
        for sub in range(nsub):
            for kind in ("energy", "overlap"):
                stored = gradient_configs[wfi][sub][kind]
                got = loaded_gradient[wfi][sub][kind]
                assert np.array_equal(got.configs, stored.configs), (wfi, sub, kind)
                if periodic:
                    assert np.array_equal(got.wrap, stored.wrap)

    # each population is distinct, so the comparison above is not trivial
    everything = [norm_configs] + [
        gradient_configs[w][s][k]
        for w in range(nwf)
        for s in range(nsub)
        for k in ("energy", "overlap")
    ]
    for i, a in enumerate(everything):
        for b in everything[i + 1 :]:
            assert not np.array_equal(a.configs, b.configs)


def test_legacy_checkpoint_returns_none(tmp_path):
    """A file without the all_configs group is a legacy checkpoint; the driver
    falls back to the single stored population rather than failing."""
    rng = np.random.default_rng(seed=12)
    configs = make_configs(rng, periodic=False)
    hdf_file = str(tmp_path / "legacy.hdf5")
    with h5py.File(hdf_file, "a") as hdf:
        configs.initialize_hdf(hdf)
        configs.to_hdf(hdf)
    with h5py.File(hdf_file, "r") as hdf:
        assert load_all_configs(hdf, configs, [[None]]) is None
