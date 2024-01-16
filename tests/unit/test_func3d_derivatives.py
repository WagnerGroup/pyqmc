# MIT License
# 
# Copyright (c) 2019 Lucas K Wagner
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

import pytest
import numpy as np
import pyqmc.func3d as func3d


@pytest.mark.parametrize(
    "func",
    [
        func3d.PadeFunction(0.2),
        func3d.PolyPadeFunction(2.0, 1.5),
        func3d.CutoffCuspFunction(2.0, 1.5),
        func3d.GaussianFunction(0.4),
    ],
)
def test_func3d(func, delta=1e-6, epsilon=1e-5):
    """
    Ensure that the 3-dimensional functions correctly compute their gradient and laplacian
    """
    delta = 1e-6
    epsilon = 1e-5

    grad = func3d.test_func3d_gradient(func, delta=delta)
    lap = func3d.test_func3d_laplacian(func, delta=delta)
    gl = func3d.test_func3d_gradient_laplacian(func)
    gv = func3d.test_func3d_gradient_value(func)
    pgrad = func3d.test_func3d_pgradient(func, delta=1e-9)
    # print(name, grad, lap, "both:", gl["grad"], gl["lap"])
    # print(name, pgrad)
    assert grad < epsilon
    assert lap < epsilon
    assert gl["grad"] < epsilon
    assert gl["lap"] < epsilon
    assert gv["grad"] < epsilon
    assert gv["val"] < epsilon
    for k, v in pgrad.items():
        assert v < epsilon, (func, k, v)


def test_cutoff_cusp():
    # Check CutoffCusp does not diverge at r/rcut = 1
    gamma = 2.0
    rc = 1.5
    basis = func3d.CutoffCuspFunction(gamma, rc)
    rvec = np.array([0, 0, rc])[np.newaxis, :]
    r = np.linalg.norm(rvec)[np.newaxis]

    v = basis.value(rvec, r)
    g = basis.gradient(rvec, r)
    l = basis.laplacian(rvec, r)
    g_both, l_both = basis.gradient_laplacian(rvec, r)

    assert abs(v).sum() == 0
    assert abs(g).sum() == 0
    assert abs(l).sum() == 0
    assert abs(g_both).sum() == 0
    assert abs(l_both).sum() == 0
