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
"""Configurations are stored in the precision they are computed in.

h5py defaults to float32 when create_dataset is given a shape but no dtype, so
walkers used to be rounded to single precision on the way to disk and came back
from a restart perturbed by ~1e-8 per coordinate.
"""

import h5py
import numpy as np
import pytest

from pyqmc.configurations.coord import OpenConfigs, PeriodicConfigs


def make_configs(rng, periodic, nconfig=6):
    coords = rng.normal(size=(nconfig, 3, 3))
    if periodic:
        return PeriodicConfigs(coords, np.eye(3) * 4.0)
    return OpenConfigs(coords)


@pytest.mark.parametrize("periodic", [False, True])
def test_configs_stored_in_double_precision(tmp_path, periodic):
    rng = np.random.default_rng(seed=1)
    configs = make_configs(rng, periodic)
    hdf_file = str(tmp_path / "configs.hdf5")

    with h5py.File(hdf_file, "a") as hdf:
        configs.initialize_hdf(hdf)
        configs.to_hdf(hdf)

    with h5py.File(hdf_file, "r") as hdf:
        assert hdf["configs"].dtype == configs.configs.dtype == np.float64
        if periodic:
            assert hdf["wrap"].dtype == configs.wrap.dtype

    # a restart gets back exactly what it wrote
    loaded = make_configs(rng, periodic)
    with h5py.File(hdf_file, "r") as hdf:
        loaded.load_hdf(hdf)
    assert np.array_equal(loaded.configs, configs.configs)
    if periodic:
        assert np.array_equal(loaded.wrap, configs.wrap)


@pytest.mark.parametrize("periodic", [False, True])
def test_load_old_single_precision_file(tmp_path, periodic, caplog):
    """Files written by earlier versions still load, with a warning that they
    will keep being written in single precision."""
    rng = np.random.default_rng(seed=2)
    configs = make_configs(rng, periodic)
    hdf_file = str(tmp_path / "old.hdf5")

    # what initialize_hdf used to produce: no dtype, so h5py chose float32
    with h5py.File(hdf_file, "a") as hdf:
        hdf.create_dataset(
            "configs",
            configs.configs.shape,
            chunks=True,
            maxshape=(None, *configs.configs.shape[1:]),
        )
        hdf["configs"][...] = configs.configs
        if periodic:
            hdf.create_dataset("wrap", configs.wrap.shape, chunks=True)
            hdf["wrap"][...] = configs.wrap
        assert hdf["configs"].dtype == np.float32

    loaded = make_configs(rng, periodic)
    with h5py.File(hdf_file, "r") as hdf:
        loaded.load_hdf(hdf)
    assert loaded.configs.dtype == np.float64
    assert np.allclose(loaded.configs, configs.configs, rtol=1e-6)
    assert not np.array_equal(loaded.configs, configs.configs)  # the old rounding
    assert "stored as float32" in caplog.text


def test_dmc_weights_stored_in_double_precision(tmp_path):
    from pyqmc.method.dmc import dmc_file

    rng = np.random.default_rng(seed=3)
    configs = make_configs(rng, periodic=False)
    weights = rng.random(configs.configs.shape[0])
    hdf_file = str(tmp_path / "dmc.hdf5")

    dmc_file(hdf_file, {"energy": 1.0}, {}, configs, weights)
    with h5py.File(hdf_file, "r") as hdf:
        assert hdf["weights"].dtype == weights.dtype == np.float64
        assert np.array_equal(hdf["weights"][()], weights)
