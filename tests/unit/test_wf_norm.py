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

"""A settable norm (pyqmc.wf.norm) divides the wave function by exp(log_norm) and
changes nothing else."""
import copy

import h5py
import numpy as np
import pytest

import pyqmc.api as pyq
from pyqmc.method.ensemble_optimization import renormalize
from pyqmc.observables.accumulators import LinearTransform
from pyqmc.wf.addwf import AddWF
from pyqmc.wf.multiplywf import MultiplyWF
from pyqmc.wf.norm import can_set_norm


@pytest.fixture(scope="module")
def h2(H2_casci):
    mol, mf, mc = H2_casci
    mc = copy.copy(mc)
    mc.ci = mc.ci[0]
    return mol, mf, mc


def slater_jastrow(h2):
    mol, mf, mc = h2
    wf, _ = pyq.generate_wf(mol, mf, mc=mc, slater_kws=dict(tol=1e-20))
    rng = np.random.default_rng(3)
    wf.parameters["wf2bcoeff"] = 0.1 * rng.normal(size=wf.parameters["wf2bcoeff"].shape)
    wf.parameters["wf2acoeff"] = 0.1 * rng.normal(size=wf.parameters["wf2acoeff"].shape)
    return wf


def snapshot(wf, configs):
    """Everything the wave function returns, after a recompute and a move."""
    out = {"recompute": wf.recompute(configs)}
    epos = configs.make_irreducible(0, configs.configs[:, 0] + 0.1)
    out["testvalue"] = wf.testvalue(0, epos)[0]
    grad, val, _ = wf.gradient_value(0, epos)
    out["gradient_value"] = (grad, val)
    out["gradient_laplacian"] = wf.gradient_laplacian(1, configs.electron(1))
    out["pgradient"] = wf.pgradient()
    wf.updateinternals(0, epos, configs)
    out["value"] = wf.value()
    return out


@pytest.mark.parametrize("which", ["slater", "jastrow", "product"])
def test_log_norm_divides_the_value(h2, which):
    product = slater_jastrow(h2)
    wf = {"slater": product.wf_factors[0], "jastrow": product.wf_factors[1], "product": product}[which]
    configs = pyq.initial_guess(h2[0], 7, rng=1)
    before = snapshot(wf, configs.copy())
    assert can_set_norm(wf) and wf.log_norm() == 0.0
    wf.set_log_norm(0.75)
    assert wf.log_norm() == 0.75
    after = snapshot(wf, configs.copy())
    for key in ["recompute", "value"]:
        assert np.allclose(after[key][0], before[key][0])
        assert np.allclose(after[key][1], before[key][1] - 0.75)
    assert np.allclose(after["testvalue"], before["testvalue"])
    for key in ["gradient_value", "gradient_laplacian"]:
        for a, b in zip(after[key], before[key]):
            assert np.allclose(a, b)
    assert set(after["pgradient"].keys()) == set(before["pgradient"].keys())
    assert not any(k.endswith("log_norm") for k in after["pgradient"])
    for k in before["pgradient"]:
        assert np.allclose(after["pgradient"][k], before["pgradient"][k])


def test_multiplywf_norm_factor(h2):
    wf = slater_jastrow(h2)
    assert wf.norm_factor == 0  # the first factor that can
    wf.wf_factors[1].set_log_norm(0.25)  # e.g. read from a file
    assert wf.log_norm() == 0.25
    wf.set_log_norm(1.0)  # the total, set through factor 0
    assert wf.wf_factors[0].log_norm() == 0.75 and wf.wf_factors[1].log_norm() == 0.25

    other = MultiplyWF(*wf.wf_factors, norm_factor=1)
    other.set_log_norm(2.0)
    assert other.wf_factors[0].log_norm() == 0.75 and other.wf_factors[1].log_norm() == 1.25

    mol, mf, mc = h2
    added = AddWF(np.ones(2), [pyq.generate_slater(mol, mf)[0] for _ in range(2)])
    assert not can_set_norm(added)
    with pytest.raises(ValueError, match="cannot set its norm"):
        MultiplyWF(added, wf.wf_factors[1], norm_factor=0)
    assert not can_set_norm(MultiplyWF(added))


def test_log_norm_is_not_optimized(h2):
    wf = slater_jastrow(h2)
    transform = LinearTransform(wf.parameters)  # every parameter, by default
    assert not any(k.endswith("log_norm") for k in transform.to_opt)
    configs = pyq.initial_guess(h2[0], 5, rng=2)
    wf.recompute(configs)
    assert transform.serialize_pgradient(wf).shape == (5, transform.nparams)


def test_renormalize_sets_log_norm(h2):
    wfs = [slater_jastrow(h2) for _ in range(3)]
    det_coeff = [wf.parameters["wf1det_coeff"].copy() for wf in wfs]
    renormalize(wfs, np.array([2.0, 8.0, 0.5]) + 0j, pivot=0)
    assert np.allclose([wf.log_norm() for wf in wfs], [0.0, 0.5 * np.log(4), 0.5 * np.log(0.25)])
    for wf, dc in zip(wfs, det_coeff):
        assert np.array_equal(wf.parameters["wf1det_coeff"], dc)

    # a wave function that cannot set its norm still has its det_coeff scaled
    mol, mf, mc = h2
    slaters = [pyq.generate_slater(mol, mf, mc=mc, tol=1e-20)[0] for _ in range(2)]
    for s in slaters:
        s.can_set_norm = lambda: False
    before = slaters[1].parameters["det_coeff"].copy()
    renormalize(slaters, np.array([1.0, 4.0]), pivot=0)
    assert np.allclose(slaters[1].parameters["det_coeff"], before / 2)


def test_log_norm_survives_a_checkpoint(h2, tmp_path):
    """Parameters saved to a checkpoint and read back keep the normalization."""
    wf = slater_jastrow(h2)
    wf.set_log_norm(-0.3)
    fname = tmp_path / "wf.hdf5"
    with h5py.File(fname, "w") as hdf:
        for k, v in wf.parameters.items():
            hdf[f"wf/{k}"] = v
    fresh = slater_jastrow(h2)
    pyq.read_wf(fresh, str(fname))
    assert np.isclose(fresh.log_norm(), -0.3)
