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
"""CG-SR solves the same equations as explicit SR and minSR.

The point of the matrix-free form is that S is never built, so the checks here
build it explicitly from the same samples and compare.
"""

import numpy as np
import pytest

from pyqmc.method.cgsr import (
    SROperator,
    cgsr_update,
    preconditioned_conjugate_gradient,
    sr_gradient,
)
from pyqmc.method.minsr import minsr_update, real_design_matrix


def make_data(rng, nsamples, nparams, complex_derivatives=False, scales=None):
    """Per-sample derivatives and local energies."""
    dppsi = rng.normal(size=(nsamples, nparams))
    if complex_derivatives:
        dppsi = dppsi + 1j * rng.normal(size=(nsamples, nparams))
    if scales is not None:
        dppsi = dppsi * scales
    eloc = rng.normal(size=nsamples) - 1.0
    return dppsi, eloc


def explicit_S(dppsi):
    """The SR matrix, built the way the matrix-free operator refuses to."""
    A, _ = real_design_matrix(dppsi, np.zeros(dppsi.shape[0]))
    return A.T @ A


@pytest.mark.parametrize("nsamples,nparams", [(40, 15), (15, 40), (30, 30)])
@pytest.mark.parametrize("complex_derivatives", [False, True])
def test_matvec_matches_explicit_matrix(nsamples, nparams, complex_derivatives):
    rng = np.random.default_rng(17)
    dppsi, _ = make_data(rng, nsamples, nparams, complex_derivatives)
    eps = 1e-2
    S = explicit_S(dppsi)
    op = SROperator(dppsi, eps=eps)
    for _ in range(3):
        v = rng.normal(size=nparams)
        assert np.allclose(op.matvec(v), S @ v + eps * v, atol=1e-12)


@pytest.mark.parametrize("complex_derivatives", [False, True])
def test_diagonal_matches_explicit_matrix(complex_derivatives):
    rng = np.random.default_rng(3)
    dppsi, _ = make_data(rng, 50, 20, complex_derivatives)
    assert np.allclose(SROperator(dppsi).diagonal(), np.diag(explicit_S(dppsi)), atol=1e-12)


def test_diagonal_chunking_is_exact():
    """The chunked accumulation must not depend on the chunk size."""
    import pyqmc.method.cgsr as cgsr

    rng = np.random.default_rng(5)
    dppsi, _ = make_data(rng, 300, 12, complex_derivatives=True)
    reference = SROperator(dppsi).diagonal()
    old = cgsr._DIAG_CHUNK
    try:
        cgsr._DIAG_CHUNK = 7
        assert np.allclose(SROperator(dppsi).diagonal(), reference, atol=1e-12)
    finally:
        cgsr._DIAG_CHUNK = old


@pytest.mark.parametrize("complex_derivatives", [False, True])
def test_gradient_matches_design_matrix(complex_derivatives):
    """sr_gradient is 2 A^T b from the minSR design matrix."""
    rng = np.random.default_rng(11)
    dppsi, eloc = make_data(rng, 60, 25, complex_derivatives)
    A, b = real_design_matrix(dppsi, eloc)
    assert np.allclose(sr_gradient(dppsi, eloc), 2 * (A.T @ b), atol=1e-12)


@pytest.mark.parametrize("nsamples,nparams", [(80, 30), (30, 80)])
@pytest.mark.parametrize("precondition", [False, True])
def test_solve_matches_dense_solve(nsamples, nparams, precondition):
    """CG reproduces the explicit (S + eps I)^-1 f."""
    rng = np.random.default_rng(23)
    dppsi, eloc = make_data(rng, nsamples, nparams)
    eps = 1e-2
    S = explicit_S(dppsi)
    f = sr_gradient(dppsi, eloc)
    expected = np.linalg.solve(S + eps * np.eye(nparams), f)

    op = SROperator(dppsi, eps=eps)
    minv = 1.0 / (op.diagonal() + eps) if precondition else None
    v, info = preconditioned_conjugate_gradient(op.matvec, f, minv=minv, rtol=1e-10)
    assert info["converged"]
    assert np.abs(v - expected).max() / np.abs(expected).max() < 1e-7


@pytest.mark.parametrize("nsamples,nparams", [(25, 60), (60, 25)])
@pytest.mark.parametrize("complex_derivatives", [False, True])
def test_cgsr_update_matches_minsr(nsamples, nparams, complex_derivatives):
    """minSR is the closed-form solve of the same system, so the two updates
    agree wherever both are affordable."""
    rng = np.random.default_rng(31)
    dppsi, eloc = make_data(rng, nsamples, nparams, complex_derivatives)
    eps, tstep = 1e-2, 0.1

    dp_minsr, _ = minsr_update(dppsi, eloc, tstep, eps=eps)
    dp_cg, report, _ = cgsr_update(dppsi, eloc, tstep, eps=eps, rtol=1e-10)

    assert report["cg_converged"]
    scale = np.abs(dp_minsr).max()
    assert np.abs(dp_cg - dp_minsr).max() / scale < 1e-6


def test_no_large_matrix_is_formed(monkeypatch):
    """The solve must not allocate anything of size nparams^2 or nsamples^2."""
    rng = np.random.default_rng(41)
    nsamples, nparams = 200, 150
    dppsi, eloc = make_data(rng, nsamples, nparams)

    big = max(nsamples, nparams) ** 2
    seen = []
    real_empty, real_zeros = np.empty, np.zeros

    def check(shape, *args, **kwargs):
        size = np.prod(shape) if np.ndim(shape) else shape
        if size >= big:
            seen.append(shape)

    def guarded_empty(shape, *args, **kwargs):
        check(shape)
        return real_empty(shape, *args, **kwargs)

    def guarded_zeros(shape, *args, **kwargs):
        check(shape)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", guarded_empty)
    monkeypatch.setattr(np, "zeros", guarded_zeros)
    cgsr_update(dppsi, eloc, 0.1, eps=1e-2, rtol=1e-8)
    assert seen == [], f"allocated large arrays: {seen}"


def test_preconditioner_helps_with_heterogeneous_scales():
    """In the regime CG is for -- many more samples than parameters -- Jacobi is
    what makes it work at all.

    Parameter scales in a real trial function differ by orders of magnitude
    across orbital/Jastrow/geminal blocks, and without preconditioning CG does
    not converge.
    """
    rng = np.random.default_rng(53)
    nparams = 120
    dppsi, eloc = make_data(rng, 2000, nparams, scales=np.logspace(-1, 1, nparams))
    eps = 1e-3

    op = SROperator(dppsi, eps=eps)
    f = sr_gradient(dppsi, eloc)
    _, plain = preconditioned_conjugate_gradient(op.matvec, f, minv=None, rtol=1e-8)
    _, jacobi = preconditioned_conjugate_gradient(
        op.matvec, f, minv=1.0 / (op.diagonal() + eps), rtol=1e-8
    )
    assert jacobi["converged"]
    assert not plain["converged"]
    assert jacobi["iterations"] < 30


def test_preconditioner_is_not_always_a_win():
    """With fewer samples than parameters S is rank deficient, so the operator is
    exactly eps*I on the null space -- a big eigenvalue cluster that plain CG
    eats in about nsamples iterations. Jacobi breaks the cluster and costs more.

    That is minSR's regime rather than this solver's, but it is why
    `precondition` is a knob and not a hard-coded choice.
    """
    rng = np.random.default_rng(59)
    dppsi, eloc = make_data(rng, 40, 200)
    eps = 1e-2
    op = SROperator(dppsi, eps=eps)
    f = sr_gradient(dppsi, eloc)

    _, plain = preconditioned_conjugate_gradient(op.matvec, f, minv=None, rtol=1e-8)
    _, jacobi = preconditioned_conjugate_gradient(
        op.matvec, f, minv=1.0 / (op.diagonal() + eps), rtol=1e-8
    )
    assert plain["converged"] and jacobi["converged"]
    assert plain["iterations"] <= dppsi.shape[0]
    assert plain["iterations"] < jacobi["iterations"]


def test_warm_start_reduces_iterations():
    """Consecutive SR systems are similar, so the previous solution is a good
    starting guess."""
    rng = np.random.default_rng(67)
    nsamples, nparams = 400, 80
    dppsi, eloc = make_data(rng, nsamples, nparams)
    eps, rtol = 1e-3, 1e-8

    op = SROperator(dppsi, eps=eps)
    f = sr_gradient(dppsi, eloc)
    minv = 1.0 / (op.diagonal() + eps)
    v, first = preconditioned_conjugate_gradient(op.matvec, f, minv=minv, rtol=rtol)

    # a nearby system: the same walkers nudged, as one optimization step later
    dppsi2 = dppsi + 0.01 * rng.normal(size=dppsi.shape)
    eloc2 = eloc + 0.01 * rng.normal(size=eloc.shape)
    op2 = SROperator(dppsi2, eps=eps)
    f2 = sr_gradient(dppsi2, eloc2)
    minv2 = 1.0 / (op2.diagonal() + eps)

    cold_v, cold = preconditioned_conjugate_gradient(op2.matvec, f2, minv=minv2, rtol=rtol)
    warm_v, warm = preconditioned_conjugate_gradient(
        op2.matvec, f2, minv=minv2, x0=v, rtol=rtol
    )
    assert warm["iterations"] < cold["iterations"]
    # and it converges to the same place
    assert np.abs(warm_v - cold_v).max() / np.abs(cold_v).max() < 1e-6


def test_warm_start_does_not_change_the_answer():
    """Even a deliberately bad starting guess converges to the same solution."""
    rng = np.random.default_rng(71)
    dppsi, eloc = make_data(rng, 120, 40)
    eps = 1e-2
    S = explicit_S(dppsi)
    f = sr_gradient(dppsi, eloc)
    expected = np.linalg.solve(S + eps * np.eye(40), f)

    op = SROperator(dppsi, eps=eps)
    bad = rng.normal(size=40) * 100
    v, info = preconditioned_conjugate_gradient(op.matvec, f, x0=bad, rtol=1e-10)
    assert info["converged"]
    assert np.abs(v - expected).max() / np.abs(expected).max() < 1e-7


def test_iteration_cap_reports_non_convergence():
    rng = np.random.default_rng(83)
    dppsi, eloc = make_data(rng, 200, 100, scales=np.logspace(-3, 3, 100))
    dp, report, _ = cgsr_update(
        dppsi, eloc, 0.1, eps=1e-8, rtol=1e-14, max_cg_iterations=2, precondition=False
    )
    assert report["cg_iterations"] == 2
    assert not report["cg_converged"]
    assert np.all(np.isfinite(dp))


def test_max_norm_clips_the_step():
    rng = np.random.default_rng(97)
    dppsi, eloc = make_data(rng, 100, 30)
    dp, report, _ = cgsr_update(dppsi, eloc, 10.0, eps=1e-3, max_norm=0.05)
    assert report["clipped"]
    assert np.linalg.norm(dp) == pytest.approx(0.05)


def test_empty_transform():
    rng = np.random.default_rng(101)
    dppsi = np.zeros((10, 0))
    eloc = rng.normal(size=10)
    dp, report, v = cgsr_update(dppsi, eloc, 0.1)
    assert dp.shape == (0,) and v.shape == (0,)
    assert report["cg_iterations"] == 0


def test_zero_gradient_returns_zero_step():
    """A constant local energy has no gradient, so there is nothing to solve."""
    rng = np.random.default_rng(103)
    dppsi = rng.normal(size=(50, 20))
    dp, report, v = cgsr_update(dppsi, np.full(50, -1.5), 0.1)
    assert np.all(dp == 0.0)
    assert report["cg_iterations"] == 0
