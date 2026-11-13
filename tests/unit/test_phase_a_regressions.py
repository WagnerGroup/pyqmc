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

"""Regressions for small bugs in ensemble optimization and the func3d evaluator."""
import numpy as np

from pyqmc.method.ensemble_optimization import renormalize
from pyqmc.observables.accumulators_multiwf import AdaptSingleAccumulator
from pyqmc.wf.func3d import CutoffFunc3dEvaluator


class _Params:
    def __init__(self, parameters):
        self.parameters = parameters


def test_renormalize_keeps_real_coefficients():
    """Complex-dtype norms (complex orbitals) must not make a real det_coeff complex."""
    wfs = [_Params({"det_coeff": np.array([1.0, 0.5])}) for _ in range(2)]
    norms = np.array([2.0 + 0j, 0.5 + 0j])
    renormalize(wfs, norms, pivot=0)
    assert not np.iscomplexobj(wfs[1].parameters["det_coeff"])
    assert np.allclose(wfs[1].parameters["det_coeff"], [2.0, 1.0])
    assert np.allclose(wfs[0].parameters["det_coeff"], [1.0, 0.5])


def test_renormalize_checks_each_wf():
    """The parameter name is taken from each wave function, not from the last one."""
    wfs = [
        _Params({"det_coeff": np.ones(1)}),
        _Params({"wf1det_coeff": np.ones(1)}),
        _Params({"det_coeff": np.ones(1)}),
    ]
    renormalize(wfs, np.array([4.0, 1.0, 1.0]), pivot=0)
    assert np.allclose(wfs[1].parameters["wf1det_coeff"], 2.0)
    assert np.allclose(wfs[2].parameters["det_coeff"], 2.0)


class _Acc:
    def keys(self):
        return {"x"}

    def shapes(self):
        return {"x": (2,)}


def test_adapt_single_accumulator_keys_shapes():
    acc = AdaptSingleAccumulator(_Acc())
    assert acc.keys() == {"x"}
    assert acc.shapes() == {"x": (2,)}


class _ZBasis:
    """A basis function of the displacement, not only of its length."""

    parameters = {"rcut": 3.0}

    def value(self, rvec, r):
        return rvec[..., 2]


def test_cutoff_evaluator_value_passes_displacement():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(5, 7, 3))
    r = np.linalg.norm(d, axis=-1)
    out = CutoffFunc3dEvaluator([_ZBasis()], 3.0).value(d, r)
    assert np.allclose(out[..., 0], np.where(r < 3.0, d[..., 2], 0.0))
