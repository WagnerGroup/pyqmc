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
r"""
Matrix-free (conjugate gradient) stochastic reconfiguration.

The natural gradient step solves

.. math:: (S + \epsilon I)\, \delta p = f

with :math:`S_{ij} = \langle O_i^* O_j\rangle - \langle O_i^*\rangle\langle O_j\rangle`
the Fisher/overlap matrix built from the log-derivatives
:math:`O_i = \partial \ln\Psi/\partial p_i`, and
:math:`f = 2{\rm Re}[\langle O^* E\rangle - \langle O^*\rangle\langle E\rangle]`
the energy gradient. These are the same equations
:mod:`pyqmc.method.linemin` and :mod:`pyqmc.method.minsr` solve; only the solver
is different.

**Mechanism.** :math:`S` is a Gram matrix of the sampled derivatives, so it is
only ever needed through matrix-vector products:

.. math:: S v = \frac{1}{N}\bar O^\dagger (\bar O v), \qquad
          \bar O_{si} = O_{si} - \langle O_i \rangle

Each product is two passes over the samples: :math:`\bar O v` projects
:math:`v` onto every walker's derivative vector, giving one scalar per walker,
and :math:`\bar O^\dagger(\cdot)` weights the derivatives by those scalars.
Both are :math:`O(N_{\rm conf} N_{\rm param})` -- the same cost and shape as
gathering the derivatives in the first place. :math:`S` is symmetric positive
semidefinite, so :math:`S+\epsilon I` is positive definite and conjugate
gradient applies.

**Memory.** The only large array is the derivative matrix
(:math:`N_{\rm conf}\times N_{\rm param}`), which is data you already collected,
plus a handful of length-:math:`N_{\rm param}` CG vectors. There is no
:math:`N_{\rm param}^2` matrix as in explicit SR and no :math:`N_{\rm conf}^2`
kernel as in minSR. Note in particular that the centering is applied through the
mean rather than by building :math:`\bar O`, so no second copy of the derivative
matrix is made; see :class:`SROperator`.

**Parallelism.** :math:`\bar O v` and :math:`\bar O^\dagger u` are both
reductions over samples, so they distribute exactly like the derivative gather.
The only serial work is CG's scalar recurrences. There is no dense solve, so
this avoids the serial bottleneck that explicit SR
(:math:`N_{\rm param}^3`) and minSR (:math:`N_{\rm conf}^3`) both hit.

**Relationship to minSR.** minSR is the closed-form sample-space solve of these
same equations: it forms the :math:`N_{\rm conf}\times N_{\rm conf}` Gram matrix
:math:`T = \bar O\bar O^\dagger` and inverts that, which is valid because for
:math:`N_{\rm conf}\le N_{\rm param}` the solution lies in the span of the
sampled derivatives (the push-through/Woodbury identity). CG never forms even
:math:`T`. So: minSR when the sample count is modest and a one-shot
:math:`N_{\rm conf}^2` solve is cheap, CG when it is large. Both avoid
:math:`N_{\rm param}^2`. :func:`cgsr_direction` and
:func:`pyqmc.method.minsr.minsr_direction` agree to solver tolerance wherever both
are affordable, which is what ``tests/unit/test_cgsr_update.py`` checks.

**Practicalities.** :math:`\epsilon` regularizes and conditions the system, and
plays the usual SR stabilizer role. The Jacobi preconditioner
:math:`{\rm diag}(S)_i = \langle|O_i|^2\rangle - |\langle O_i\rangle|^2` costs
one cheap pass and is what makes this work at all, because parameter scales in a
typical trial function -- orbitals against two-body Jastrow against geminal --
are wildly heterogeneous. On synthetic data with 2000 samples, 120 parameters
and scales spread over two decades, Jacobi converges in 13 iterations while
unpreconditioned CG does not converge at all. With uniform scales the two are
identical, so it costs nothing to leave on. The iteration count is set by the
conditioning, not by the number of parameters.

Warm-starting from the previous step's solution is available (`warm_start`) and
is exact -- it converges to the same answer from any starting point -- but
measure before relying on it. It helps when consecutive systems really are
close, and each optimization step here draws fresh samples, so the systems
differ by sampling noise of order :math:`1/\sqrt{N}` that swamps the parameter
drift. On H2/ccpvdz it was a wash: 153 total CG iterations against 152 cold at
rtol=1e-10, and 101 against 108 at rtol=1e-6, with 2000 walkers and tstep=0.02.

One caveat, which is why `precondition` is a knob rather than a hard-coded
choice: when there are fewer samples than parameters, :math:`S` is rank
deficient and the regularized operator is exactly :math:`\epsilon I` on its null
space. That large eigenvalue cluster is something plain CG exploits, converging
in about :math:`N_{\rm conf}` iterations, and Jacobi breaks it up -- 15
iterations against 43 for 25 samples and 200 parameters. That is minSR's regime
rather than this solver's, but if you run CG there, turn the preconditioner off.

Use it through :class:`pyqmc.method.optimize.CGSRUpdate`.
"""

import logging

import numpy as np

#: rows of the derivative matrix touched at once when accumulating diag(S)
_DIAG_CHUNK = 4096


class SROperator:
    r"""The regularized SR matrix :math:`S + \epsilon I` as a matrix-vector product.

    Never forms :math:`S`, and never forms the centered derivative matrix
    :math:`\bar O` either: centering enters through the mean, using

    .. math:: \bar O v = O v - (\langle O\rangle\cdot v), \qquad
              \bar O^\dagger u = O^\dagger u - \langle O\rangle^* \textstyle\sum_s u_s

    so the only array of size (nsamples, nparameters) is the one passed in.

    The parameters are real (complex wave function parameters are serialized into
    real and imaginary components by LinearTransform), so only the real part of
    the product is kept. Conjugation is applied to the length-nsamples vector
    rather than to the derivatives, using
    :math:`{\rm Re}[O^\dagger u] = {\rm Re}[O^T u^*]`, which keeps complex
    derivatives from costing an extra full-size copy.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter float eps: regularization added to the diagonal
    """

    def __init__(self, dppsi, eps=1e-2):
        self.dppsi = dppsi
        self.eps = eps
        self.nsamples, self.nparameters = dppsi.shape
        self.mean = np.mean(dppsi, axis=0)

    def matvec(self, v):
        r""":math:`(S + \epsilon I) v` for a real vector `v`.

        Two BLAS-2 products over the derivative matrix.
        """
        t = self.dppsi @ v - (self.mean @ v)
        tc = np.conj(t)
        w = self.dppsi.T @ tc - self.mean * np.sum(tc)
        return np.real(w) / self.nsamples + self.eps * v

    def diagonal(self):
        r""":math:`{\rm diag}(S) = \langle|O_i|^2\rangle - |\langle O_i\rangle|^2`,
        without the regularization.

        Accumulated in chunks of rows so that the pass does not allocate a second
        array the size of the derivative matrix.
        """
        d = np.zeros(self.nparameters)
        for start in range(0, self.nsamples, _DIAG_CHUNK):
            block = self.dppsi[start : start + _DIAG_CHUNK]
            d += np.einsum("sp,sp->p", block.real, block.real)
            if np.iscomplexobj(block):
                d += np.einsum("sp,sp->p", block.imag, block.imag)
        d /= self.nsamples
        d -= np.real(self.mean * np.conj(self.mean))
        # roundoff can push a numerically zero direction slightly negative
        return np.maximum(d, 0.0)


def sr_gradient(dppsi, eloc):
    r"""The energy gradient
    :math:`f_i = 2{\rm Re}[\langle O_i^* E\rangle - \langle O_i^*\rangle\langle E\rangle]`.

    One pass over the samples, with the same centering-through-the-mean trick as
    :class:`SROperator`. This is the same quantity as ``2 * A.T @ b`` from
    :func:`pyqmc.method.minsr.real_design_matrix`.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter eloc: (nsamples,) local energies
    :returns: (nparameters,) real gradient
    """
    nsamples = dppsi.shape[0]
    ebar = np.conj(eloc - np.mean(eloc))
    w = dppsi.T @ ebar - np.mean(dppsi, axis=0) * np.sum(ebar)
    return 2.0 * np.real(w) / nsamples


def preconditioned_conjugate_gradient(
    matvec, b, minv=None, x0=None, rtol=1e-6, maxiter=None
):
    r"""Solve a symmetric positive definite system with preconditioned CG.

    :parameter matvec: callable taking and returning a (n,) real array
    :parameter b: (n,) right hand side
    :parameter minv: (n,) diagonal of the inverse preconditioner, or None for
        no preconditioning
    :parameter x0: (n,) starting guess, or None to start from zero. Starting from
        zero skips the first matrix-vector product.
    :parameter float rtol: stop when ``||b - Ax|| <= rtol * ||b||``
    :parameter int maxiter: iteration cap; defaults to ``min(max(2n, 50), 1000)``.
        Note that n iterations is CG's exact-arithmetic bound, not a practical
        one: roundoff usually costs a few extra iterations to polish the residual,
        so capping at n would report spurious non-convergence on small systems.
        In the regime this solver is for, the cap is never reached.
    :returns: (x, info) where info has 'iterations', 'residual' (relative), and
        'converged'
    """
    n = b.shape[0]
    if maxiter is None:
        maxiter = min(max(2 * n, 50), 1000)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0 or n == 0:
        return np.zeros(n), {"iterations": 0, "residual": 0.0, "converged": True}

    if x0 is None:
        x = np.zeros(n)
        r = b.copy()
    else:
        x = np.array(x0, dtype=float, copy=True)
        r = b - matvec(x)

    tol = rtol * bnorm
    rnorm = np.linalg.norm(r)
    if rnorm <= tol:  # a warm start can already be good enough
        return x, {"iterations": 0, "residual": rnorm / bnorm, "converged": True}

    z = r if minv is None else minv * r
    p = z.copy()
    rz = np.dot(r, z)

    iterations = 0
    for iterations in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = np.dot(p, Ap)
        if pAp <= 0:  # only reachable if the operator is not positive definite
            logging.warning("CG encountered a non-positive curvature direction")
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        rnorm = np.linalg.norm(r)
        if rnorm <= tol:
            break
        z = r if minv is None else minv * r
        rz_new = np.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new

    return x, {
        "iterations": iterations,
        "residual": rnorm / bnorm,
        "converged": bool(rnorm <= tol),
    }


def cgsr_direction(
    dppsi,
    eloc,
    eps=1e-2,
    rtol=1e-6,
    max_cg_iterations=None,
    x0=None,
    precondition=True,
    g=None,
):
    r"""The SR direction :math:`v = (S+\epsilon I)^{-1} (f + g)`, solved by
    conjugate gradient without ever forming S. Agrees with
    :func:`pyqmc.method.minsr.minsr_direction` and
    :func:`pyqmc.observables.stochastic_reconfiguration.sr_direction` to within
    the CG tolerance.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter eloc: (nsamples,) local energies
    :parameter float eps: regularization of the SR matrix
    :parameter float rtol: relative residual at which CG stops
    :parameter int max_cg_iterations: iteration cap for CG
    :parameter x0: starting guess for the solve, normally the previous step's `v`
    :parameter boolean precondition: use the Jacobi preconditioner diag(S)+eps
    :parameter g: (nparameters,) gradient term to add to the energy gradient, or None
    :returns: (v, report)
    """
    if dppsi.shape[1] == 0:  # nothing to optimize in this transform
        return np.zeros(0), {
            "pgrad": 0.0,
            "SRdot": 0.0,
            "nsamples": dppsi.shape[0],
            "cg_iterations": 0,
            "cg_residual": 0.0,
            "cg_converged": True,
        }

    operator = SROperator(dppsi, eps=eps)
    pgrad = sr_gradient(dppsi, eloc)
    if g is not None:
        pgrad = pgrad + g
    minv = 1.0 / (operator.diagonal() + eps) if precondition else None

    v, info = preconditioned_conjugate_gradient(
        operator.matvec,
        pgrad,
        minv=minv,
        x0=x0,
        rtol=rtol,
        maxiter=max_cg_iterations,
    )
    vnorm = np.linalg.norm(v)
    gnorm = np.linalg.norm(pgrad)
    report = {
        "pgrad": gnorm,
        "SRdot": np.dot(pgrad, v) / (vnorm * gnorm) if vnorm * gnorm > 0 else 0.0,
        "nsamples": dppsi.shape[0],
        "cg_iterations": info["iterations"],
        "cg_residual": info["residual"],
        "cg_converged": info["converged"],
    }
    if not info["converged"]:
        logging.warning(
            f"CG stopped at relative residual {info['residual']:.2e} after "
            f"{info['iterations']} iterations without reaching rtol={rtol:.1e}"
        )
    return v, report
