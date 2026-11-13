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
"""
Energy minimization using the minimum-step stochastic reconfiguration (minSR)
algorithm of Chen and Heyl, Nature Physics 20, 1476 (2024).

minSR computes exactly the same parameter update as regularized stochastic
reconfiguration (:class:`pyqmc.method.optimize.SRUpdate`), but never forms the
(nparameters, nparameters) overlap matrix S. Instead it works with the sample
space kernel T = A A^T, which is (nsamples, nsamples). The two are related by
the push-through identity

.. math:: A^T (A A^T + \\epsilon I)^{-1} = (A^T A + \\epsilon I)^{-1} A^T

so the update is identical (for the same eps) while the memory cost changes
from O(nparameters^2) to O(nsamples * nparameters) for the stored derivatives
plus O(nsamples^2) for the kernel. This is a large win when the number of
parameters is much larger than the number of samples used per step.

Use it through :class:`pyqmc.method.optimize.MinSRUpdate`.
"""

import numpy as np

import pyqmc.method.mc
import pyqmc.rng
from pyqmc.observables.stochastic_reconfiguration import local_derivatives


@pyqmc.rng.worker
def sample_derivatives_worker(wf, configs, transform, enacc, nodal_cutoff, *, rng=None):
    """Evaluate the local energy and the parameter derivatives of log(psi) for
    every configuration, without averaging over configurations.

    :parameter wf: wave function object
    :parameter configs: (nconfig, nelec, 3) configurations
    :parameter transform: a LinearTransform object
    :parameter enacc: an EnergyAccumulator-like object
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    :returns: dictionary with 'dppsi' (nconfig, nparameters) and 'total' (nconfig,)
    """
    wf.recompute(configs)
    den, dppsi = local_derivatives(wf, configs, enacc, transform, nodal_cutoff)
    return {"dppsi": dppsi, "total": den["total"]}


def sample_derivatives(
    wf,
    configs,
    transform,
    enacc,
    nodal_cutoff,
    client=None,
    npartitions=None,
    rng=None,
):
    """Same as :func:`sample_derivatives_worker`, distributed over a client if one is given."""
    rng = pyqmc.rng.get_rng(rng)
    if client is None:
        return sample_derivatives_worker(
            wf, configs, transform, enacc, nodal_cutoff, rng=rng
        )
    config = configs.split(npartitions)
    children = rng.spawn(len(config))
    runs = [
        client.submit(
            sample_derivatives_worker, wf, conf, transform, enacc, nodal_cutoff, rng=child
        )
        for conf, child in zip(config, children)
    ]
    allresults = [r.result() for r in runs]
    return {
        k: np.concatenate([res[k] for res in allresults], axis=0)
        for k in allresults[0]
    }


def real_design_matrix(dppsi, eloc):
    """Build the real least-squares system whose normal equations are the
    stochastic reconfiguration equations.

    With :math:`\\bar{O}_{sk} = (O_{sk} - \\langle O_k \\rangle)/\\sqrt{N}` and
    :math:`\\epsilon_s = (E_s - \\langle E \\rangle)/\\sqrt{N}`, the SR overlap
    matrix is :math:`S = {\\rm Re}(\\bar{O}^\\dagger \\bar{O})` and the energy
    gradient is :math:`f = 2 {\\rm Re}(\\bar{O}^\\dagger \\epsilon)`. Stacking
    real and imaginary parts into a real matrix A and vector b gives
    :math:`S = A^T A` and :math:`f = 2 A^T b`, so that all the linear algebra
    below can be done in real arithmetic.

    The parameters are always real (complex wave function parameters are
    serialized into real and imaginary components by LinearTransform), so the
    derivatives are conjugated here. Note that
    :class:`pyqmc.observables.stochastic_reconfiguration.StochasticReconfiguration`
    does not conjugate; the two agree for real derivatives.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter eloc: (nsamples,) local energies
    :returns: (A, b) real arrays of shape (nrows, nparameters) and (nrows,), where
        nrows is nsamples for real input and 2*nsamples for complex input
    """
    nsamples = dppsi.shape[0]
    norm = 1.0 / np.sqrt(nsamples)
    obar = (dppsi - np.mean(dppsi, axis=0, keepdims=True)) * norm
    ebar = (eloc - np.mean(eloc)) * norm

    if np.iscomplexobj(obar) or np.iscomplexobj(ebar):
        # Re(Obar^dagger eps) = Re(Obar)^T Re(eps) + Im(Obar)^T Im(eps)
        obar = np.asarray(obar, dtype=complex)
        ebar = np.asarray(ebar, dtype=complex)
        A = np.concatenate([obar.real, obar.imag], axis=0)
        b = np.concatenate([ebar.real, ebar.imag], axis=0)
    else:
        A = np.asarray(obar, dtype=float)
        b = np.asarray(ebar, dtype=float)
    return A, b


def sr_solve(A, b, g=None, eps=1e-2, inverse_strategy="regularized_inverse"):
    """Apply the regularized SR inverse using only the (nrows, nrows) kernel.

    Returns :math:`(A^T A + \\epsilon I)^{-1} (A^T b + g)`, where A and b come
    from :func:`real_design_matrix`, so that :math:`S = A^T A` is the SR overlap
    matrix and :math:`A^T b` is the energy gradient. S is never formed.

    The :math:`A^T b` part uses the push-through identity. The optional `g` is a
    gradient contribution that does not come from the sampled rows -- the
    overlap penalty in ensemble optimization is one -- and generally has a
    component outside the row space of A, so it needs the Woodbury identity

    .. math:: (A^T A + \\epsilon I)^{-1} = (I - A^T (A A^T + \\epsilon I)^{-1} A)/\\epsilon

    Note that the component of `g` orthogonal to every sampled row comes back
    divided by eps. That is what SR itself does with the null space of S, not an
    artifact of working in sample space, but it does mean a small eps amplifies
    any part of the penalty gradient the samples cannot resolve.

    :parameter A: (nrows, nparameters) real design matrix
    :parameter b: (nrows,) real right hand side
    :parameter g: (nparameters,) additional gradient, or None
    :parameter float eps: regularization
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    :returns: (nparameters,) solution vector
    """
    kernel = A @ A.T
    rhs = b[:, np.newaxis] if g is None else np.stack([b, A @ g], axis=1)

    if inverse_strategy == "regularized_inverse":
        y = np.linalg.solve(kernel + eps * np.eye(kernel.shape[0]), rhs)
    elif inverse_strategy == "pseudo_inverse":
        if g is not None:
            raise ValueError(
                "pseudo_inverse cannot be applied to a gradient outside the row space "
                "of the samples; use regularized_inverse."
            )
        y = np.linalg.pinv(kernel, rcond=eps) @ rhs
    else:
        raise ValueError(
            "Invalid inverse strategy. Valid options are pseudo_inverse and regularized_inverse."
        )

    v = A.T @ y[:, 0]
    if g is not None:
        v = v + (g - A.T @ y[:, 1]) / eps
    return v


def minsr_direction(dppsi, eloc, eps=1e-2, inverse_strategy="regularized_inverse", g=None):
    """The SR direction :math:`v = (S + \\epsilon I)^{-1} (f + g)`, computed in
    the (nsamples, nsamples) sample space without forming S. With
    :math:`f = 2 A^T b`, this is :math:`2 A^T (A A^T + \\epsilon I)^{-1} b`
    when g is None.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter eloc: (nsamples,) local energies
    :parameter float eps: regularization of the kernel
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    :parameter g: (nparameters,) gradient term to add to the energy gradient, or None
    :returns: (v, report)
    """
    if dppsi.shape[1] == 0:  # nothing to optimize in this transform
        return np.zeros(0), {"pgrad": 0.0, "SRdot": 0.0, "nsamples": dppsi.shape[0]}

    A, b = real_design_matrix(dppsi, eloc)
    # (S + eps)^-1 (2 A^T b + g) = 2 (S + eps)^-1 (A^T b + g/2)
    v = 2 * sr_solve(A, b, g=None if g is None else g / 2, eps=eps, inverse_strategy=inverse_strategy)
    pgrad = 2 * (A.T @ b)
    if g is not None:
        pgrad = pgrad + g
    report = {
        "pgrad": np.linalg.norm(pgrad),
        "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
        "nsamples": dppsi.shape[0],
    }
    return v, report


def local_energy_error(eloc):
    """Standard error of the mean local energy, from the per-configuration values.

    The methods that keep derivatives per configuration keep the local energies
    that way too, so the error bar comes from the spread across configurations
    rather than from the scatter of a handful of block means. That means a single
    block is enough -- which matters, because these methods default to one -- and
    with several blocks it uses every sample instead of `nblocks` numbers.

    Walkers are independent Markov chains, and blocks are separated by
    `nsteps_per_block` sweeps precisely to decorrelate them, so the samples are
    treated as independent. That is the same assumption the block-mean estimator
    makes; if it is in doubt, raise `nsteps_per_block`.

    :parameter eloc: (nsamples,) local energies, possibly complex
    :returns: the standard error, or nan if there is nothing to estimate from
    """
    eloc = np.real(np.asarray(eloc))
    if eloc.size < 2:
        return np.nan
    return np.std(eloc, ddof=1) / np.sqrt(eloc.size)


def sample_minsr_data(
    wf,
    coords,
    transform,
    enacc,
    nodal_cutoff,
    nblocks=1,
    nsteps_per_block=10,
    tstep=0.5,
    client=None,
    npartitions=None,
    rng=None,
):
    """Sample the derivatives and local energies used to build the minSR update.

    Each block runs `nsteps_per_block` VMC sweeps to decorrelate, then stores
    one row per walker. The total number of samples is
    nblocks * nwalkers, and the minSR kernel is (nsamples, nsamples) (twice that
    for complex derivatives), so increasing nblocks improves the statistics at a
    quadratic cost in memory and a cubic cost in solve time.

    :returns: (data, coords) where data has 'dppsi' (nsamples, nparameters),
        'total' (nsamples,), and 'block_energy' (nblocks,)
    """
    rng = pyqmc.rng.get_rng(rng)
    dppsi = []
    eloc = []
    block_energy = []
    for _ in range(nblocks):
        _, coords = pyqmc.method.mc.vmc(
            wf,
            coords,
            accumulators={},
            nblocks=1,
            nsteps_per_block=nsteps_per_block,
            tstep=tstep,
            client=client,
            npartitions=npartitions,
            rng=rng,
        )
        block = sample_derivatives(
            wf, coords, transform, enacc, nodal_cutoff, client, npartitions, rng=rng
        )
        dppsi.append(block["dppsi"])
        eloc.append(block["total"])
        block_energy.append(np.mean(block["total"]))

    data = {
        "dppsi": np.concatenate(dppsi, axis=0),
        "total": np.concatenate(eloc, axis=0),
        "block_energy": np.asarray(block_energy),
    }
    return data, coords
