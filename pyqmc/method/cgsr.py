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
:math:`N_{\rm param}^2`. :func:`cgsr_update` and
:func:`pyqmc.method.minsr.minsr_update` agree to solver tolerance wherever both
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
"""

import logging
import os

import h5py
import numpy as np

import pyqmc.gpu as gpu
import pyqmc.method.mc
from pyqmc.method.linemin import opt_hdf
from pyqmc.method.minsr import (
    local_energy_error,
    sample_minsr_data,
    set_wf_params,
)

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


def cgsr_update(
    dppsi,
    eloc,
    tstep,
    eps=1e-2,
    rtol=1e-6,
    max_cg_iterations=None,
    x0=None,
    precondition=True,
    max_norm=None,
    verbose=False,
):
    r"""Compute the SR parameter change without ever forming S.

    Solves :math:`(S+\epsilon I) v = f` by conjugate gradient and returns
    :math:`\delta p = -\tau v`, the same update
    :func:`pyqmc.method.minsr.minsr_update` and the regularized inverse in
    :class:`pyqmc.observables.stochastic_reconfiguration.StochasticReconfiguration`
    produce for the same eps, to within the CG tolerance.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter eloc: (nsamples,) local energies
    :parameter float tstep: step size along the update direction
    :parameter float eps: regularization of the SR matrix
    :parameter float rtol: relative residual at which CG stops
    :parameter int max_cg_iterations: iteration cap for CG
    :parameter x0: starting guess for the solve, normally the previous step's `v`
    :parameter boolean precondition: use the Jacobi preconditioner diag(S)+eps
    :parameter float max_norm: if not None, rescale the update so that its 2-norm
        is at most max_norm
    :parameter boolean verbose: print diagnostics
    :returns: (dp, report, v) -- the parameter change, a dictionary of
        diagnostics, and the solution vector to warm-start the next step from
    """
    if dppsi.shape[1] == 0:  # nothing to optimize in this transform
        zero = np.zeros(0)
        return (
            zero,
            {
                "pgrad": 0.0,
                "SRdot": 0.0,
                "step_norm": 0.0,
                "clipped": False,
                "cg_iterations": 0,
                "cg_residual": 0.0,
                "cg_converged": True,
            },
            zero,
        )

    operator = SROperator(dppsi, eps=eps)
    pgrad = sr_gradient(dppsi, eloc)
    minv = 1.0 / (operator.diagonal() + eps) if precondition else None

    v, info = preconditioned_conjugate_gradient(
        operator.matvec,
        pgrad,
        minv=minv,
        x0=x0,
        rtol=rtol,
        maxiter=max_cg_iterations,
    )
    dp = -tstep * v

    norm = np.linalg.norm(dp)
    vnorm = np.linalg.norm(v)
    gnorm = np.linalg.norm(pgrad)
    report = {
        "pgrad": gnorm,
        "SRdot": np.dot(pgrad, v) / (vnorm * gnorm) if vnorm * gnorm > 0 else 0.0,
        "step_norm": norm,
        "cg_iterations": info["iterations"],
        "cg_residual": info["residual"],
        "cg_converged": info["converged"],
    }
    if max_norm is not None and norm > max_norm:
        dp = dp * (max_norm / norm)
        report["step_norm"] = max_norm
    report["clipped"] = max_norm is not None and norm > max_norm

    if not info["converged"]:
        logging.warning(
            f"CG stopped at relative residual {info['residual']:.2e} after "
            f"{info['iterations']} iterations without reaching rtol={rtol:.1e}"
        )
    if verbose:
        print(
            "CG iterations",
            info["iterations"],
            "relative residual",
            f"{info['residual']:.2e}",
        )
        print("Number of samples", dppsi.shape[0], "number of parameters", dppsi.shape[1])
        print("Gradient norm:", report["pgrad"])
        print("Dot product between gradient and SR step:", report["SRdot"])
        print("Step norm:", report["step_norm"])
    return dp, report, v


def cgsr_optimization(
    wf,
    coords,
    transform,
    enacc,
    tstep=0.02,
    eps=1e-2,
    nodal_cutoff=1e-3,
    rtol=1e-6,
    max_cg_iterations=None,
    precondition=True,
    warm_start=True,
    max_iterations=30,
    warmup_options=None,
    vmcoptions=None,
    max_norm=None,
    verbose=False,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Optimize the energy with conjugate-gradient stochastic reconfiguration.

    The same optimizer as :func:`pyqmc.method.minsr.minsr_optimization` -- same
    sampling, same equations, same fixed `tstep` with no line minimization --
    solved iteratively instead of in closed form. Use this one when the number of
    samples is large, since minSR's kernel is square in the sample count and its
    solve is cubic in it, while CG's cost is the number of iterations times the
    cost of the derivative gather.

    Like :func:`pyqmc.method.minsr.minsr_optimization`, this takes the parameter
    transform and the energy accumulator directly rather than a
    StochasticReconfiguration object, since the S matrix that object builds is
    exactly what is being avoided::

        transform = LinearTransform(wf.parameters, to_opt)
        enacc = EnergyAccumulator(mol)
        wf, df = cgsr_optimization(wf, coords, transform, enacc)

    `transform` may also be a list of transforms, in which case each iteration
    runs one sub-iteration per transform, optimizing that subset of the
    parameters while holding the rest fixed. Each sub-iteration keeps its own
    warm start.

    :parameter wf: initial wave function
    :parameter coords: initial configurations
    :parameter transform: a LinearTransform object defining the parameters to
        optimize, or a list of them to optimize in alternating sub-iterations
    :parameter enacc: an EnergyAccumulator-like object
    :parameter float tstep: step size in parameter space
    :parameter float eps: regularization of the SR matrix
    :parameter float nodal_cutoff: regularization distance for the nodal
        divergence of the derivatives
    :parameter float rtol: relative residual at which CG stops
    :parameter int max_cg_iterations: iteration cap for CG
    :parameter boolean precondition: use the Jacobi preconditioner
    :parameter boolean warm_start: start each solve from the previous step's
        solution. Exact either way; see the module docstring for why the saving
        is small when every step resamples.
    :parameter int max_iterations: total number of optimization steps, including
        any read from hdf_file
    :parameter dict warmup_options: kwargs for the initial vmc warmup
    :parameter dict vmcoptions: kwargs for sampling; nblocks, nsteps_per_block,
        and tstep are passed to sample_minsr_data
    :parameter float max_norm: if not None, the maximum 2-norm of a parameter change
    :parameter boolean verbose: print output if True
    :parameter str hdf_file: hdf file to store output; the format matches
        line_minimization
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :return: optimized wave function, optimization data
    """
    transforms = list(transform) if isinstance(transform, (list, tuple)) else [transform]
    for t in transforms:
        if hasattr(t, "transform"):
            raise ValueError(
                "cgsr_optimization takes a transform and an energy accumulator, not a "
                "StochasticReconfiguration object; pass acc.transform and acc.enacc."
            )
    if vmcoptions is None:
        vmcoptions = {}
    if warmup_options is None:
        warmup_options = {"nblocks": 1, "nsteps_per_block": 100}
    if "tstep" not in warmup_options and "tstep" in vmcoptions:
        warmup_options["tstep"] = vmcoptions["tstep"]

    iteration_offset = 0
    sub_iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf:
                grp = hdf["wf"]
                for k in grp:
                    wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf:
                # resume at the sub-iteration after the last one recorded
                iteration_offset = np.max(hdf["iteration"][...])
            if "sub_iteration" in hdf:
                sub_iteration_offset = hdf["sub_iteration"][-1] + 1
            coords.load_hdf(hdf)
    else:  # not restarting -- VMC warm up period
        if verbose:
            print("starting warmup")
        _, coords = pyqmc.method.mc.vmc(
            wf,
            coords,
            accumulators={},
            client=client,
            npartitions=npartitions,
            verbose=verbose,
            **warmup_options,
        )
        if verbose:
            print("finished warmup", flush=True)

    if iteration_offset >= max_iterations:
        logging.warning(
            f"iteration_offset {iteration_offset} >= max_iterations {max_iterations}; no steps will be run."
        )

    attr = {
        "max_iterations": max_iterations,
        "tstep": tstep,
        "eps": eps,
        "rtol": rtol,
        "precondition": precondition,
        "warm_start": warm_start,
    }

    # one warm start per sub-iteration, since each has its own parameter set
    previous_v = [None] * len(transforms)

    df = []
    for it in range(iteration_offset, max_iterations):
        for sub_it in range(sub_iteration_offset, len(transforms)):
            if verbose:
                print(
                    "#############################\nStarting iteration",
                    it,
                    "sub iteration",
                    sub_it,
                )
            sub_transform = transforms[sub_it]
            x0 = sub_transform.serialize_parameters(wf.parameters)

            data, coords = sample_minsr_data(
                wf,
                coords,
                sub_transform,
                enacc,
                nodal_cutoff,
                client=client,
                npartitions=npartitions,
                **vmcoptions,
            )
            if np.isnan(data["total"]).any():
                raise ValueError(
                    "NaN in optimization. Try reducing the step size or increasing eps."
                )

            energy = np.mean(data["total"]).real
            nsamples = data["total"].shape[0]
            energy_error = local_energy_error(data["total"])
            if verbose:
                print("Current energy", energy, energy_error)

            dp, report, v = cgsr_update(
                data["dppsi"],
                data["total"],
                tstep,
                eps=eps,
                rtol=rtol,
                max_cg_iterations=max_cg_iterations,
                x0=previous_v[sub_it] if warm_start else None,
                precondition=precondition,
                max_norm=max_norm,
                verbose=verbose,
            )
            previous_v[sub_it] = v

            step_data = dict(report)
            step_data["energy"] = energy
            step_data["energy_error"] = energy_error
            step_data["iteration"] = it
            step_data["sub_iteration"] = sub_it
            step_data["nconfig"] = coords.configs.shape[0]
            step_data["nsamples"] = nsamples

            set_wf_params(wf, x0 + dp, sub_transform)
            opt_hdf(hdf_file, step_data, attr, coords, wf.parameters)
            df.append(step_data)

        sub_iteration_offset = 0

    return wf, df
