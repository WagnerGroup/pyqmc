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

import h5py
import numpy as np
import pytest
import pyqmc.api as pyq


def _write(fname, groups):
    with h5py.File(fname, "w") as f:
        for path, params in groups.items():
            for k, v in params.items():
                f[f"{path}/{k}"] = v


def _random_params(wf, rng):
    return {k: rng.random(np.shape(v)) for k, v in wf.parameters.items()}


@pytest.fixture
def wf(LiH_sto3g_rhf):
    mol, mf = LiH_sto3g_rhf
    wf, _ = pyq.generate_wf(mol, mf)
    return wf


def test_single_layout(wf, tmp_path):
    fname = tmp_path / "single.hdf5"
    params = _random_params(wf, np.random.default_rng(0))
    _write(fname, {"wf": params})
    pyq.read_wf(wf, fname)
    for k, v in params.items():
        assert np.array_equal(wf.parameters[k], v)
    with pytest.raises(ValueError):
        pyq.read_wf(wf, fname, state=1)


def test_ensemble_layout(wf, tmp_path):
    fname = tmp_path / "ensemble.hdf5"
    rng = np.random.default_rng(1)
    states = [_random_params(wf, rng) for _ in range(2)]
    _write(fname, {f"wf/{i}": p for i, p in enumerate(states)})
    for i, params in enumerate(states):
        pyq.read_wf(wf, fname, state=i)
        for k, v in params.items():
            assert np.array_equal(wf.parameters[k], v)
    with pytest.raises(ValueError):
        pyq.read_wf(wf, fname, state=2)


def test_skip_prefixes(wf, tmp_path):
    fname = tmp_path / "single.hdf5"
    before = {k: np.array(v) for k, v in wf.parameters.items()}
    params = _random_params(wf, np.random.default_rng(2))
    _write(fname, {"wf": params})
    pyq.read_wf(wf, fname, skip_prefixes=("wf1",))
    for k, v in params.items():
        expected = before[k] if k.startswith("wf1") else v
        assert np.array_equal(wf.parameters[k], expected)
