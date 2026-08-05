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
reconfiguration (:mod:`pyqmc.method.linemin`), but never forms the
(nparameters, nparameters) overlap matrix S. Instead it works with the sample
space kernel T = A A^T, which is (nsamples, nsamples). The two are related by
the push-through identity

.. math:: A^T (A A^T + \\epsilon I)^{-1} = (A^T A + \\epsilon I)^{-1} A^T

so the update is identical (for the same eps) while the memory cost changes
from O(nparameters^2) to O(nsamples * nparameters) for the stored derivatives
plus O(nsamples^2) for the kernel. This is a large win when the number of
parameters is much larger than the number of samples used per step.

There is no line minimization here: the timestep `tstep` is a fixed
hyperparameter, as in the original minSR work.
"""

import logging
import os

import h5py
import numpy as np

import pyqmc.gpu as gpu
import pyqmc.method.mc
from pyqmc.method.linemin import opt_hdf
from pyqmc.observables.stochastic_reconfiguration import nodal_regularization


def set_wf_params(wf, params, transform):
    """Set the wave function parameters from a serialized parameter vector."""
    newparms = transform.deserialize(wf, params)
    for k in newparms:
        wf.parameters[k] = newparms[k]


def sample_derivatives_worker(wf, configs, transform, enacc, nodal_cutoff):
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
    den = enacc(configs, wf)
    dp = transform.serialize_gradients(wf.pgradient())
    _, f = nodal_regularization(den["grad2"], nodal_cutoff)
    return {"dppsi": dp * f[:, np.newaxis], "total": den["total"]}


def sample_derivatives(
    wf, configs, transform, enacc, nodal_cutoff, client=None, npartitions=None
):
    """Same as :func:`sample_derivatives_worker`, distributed over a client if one is given."""
    if client is None:
        return sample_derivatives_worker(wf, configs, transform, enacc, nodal_cutoff)
    config = configs.split(npartitions)
    runs = [
        client.submit(
            sample_derivatives_worker, wf, conf, transform, enacc, nodal_cutoff
        )
        for conf in config
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


def minsr_update(
    dppsi,
    eloc,
    tstep,
    eps=1e-3,
    inverse_strategy="regularized_inverse",
    max_norm=None,
    verbose=False,
):
    """Compute the minSR parameter change.

    The update is
    :math:`\\delta p = -2\\tau A^T (A A^T + \\epsilon I)^{-1} b`, which is equal
    to the regularized SR update :math:`-\\tau (S + \\epsilon I)^{-1} f` but is
    evaluated in the (nsamples, nsamples) sample space instead of the
    (nparameters, nparameters) parameter space.

    :parameter dppsi: (nsamples, nparameters) derivatives of log(psi)
    :parameter eloc: (nsamples,) local energies
    :parameter float tstep: step size along the update direction
    :parameter float eps: regularization of the kernel
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    :parameter float max_norm: if not None, rescale the update so that its 2-norm is at most max_norm
    :parameter boolean verbose: print diagnostics
    :returns: (dp, report) the parameter change and a dictionary of diagnostics
    """
    if dppsi.shape[1] == 0:  # nothing to optimize in this transform
        zero = np.zeros(0)
        return zero, {"pgrad": 0.0, "SRdot": 0.0, "step_norm": 0.0, "clipped": False}

    A, b = real_design_matrix(dppsi, eloc)
    kernel = A @ A.T

    if inverse_strategy == "regularized_inverse":
        y = np.linalg.solve(kernel + eps * np.eye(kernel.shape[0]), b)
    elif inverse_strategy == "pseudo_inverse":
        y = np.linalg.pinv(kernel, rcond=eps) @ b
    else:
        raise ValueError(
            "Invalid inverse strategy. Valid options are pseudo_inverse and regularized_inverse."
        )

    v = 2 * (A.T @ y)  # (S + eps)^-1 f, with f = 2 A^T b
    pgrad = 2 * (A.T @ b)  # the bare energy gradient, for diagnostics
    dp = -tstep * v

    norm = np.linalg.norm(dp)
    report = {
        "pgrad": np.linalg.norm(pgrad),
        "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
        "step_norm": norm,
    }
    if max_norm is not None and norm > max_norm:
        dp = dp * (max_norm / norm)
        report["step_norm"] = max_norm
    report["clipped"] = max_norm is not None and norm > max_norm

    if verbose:
        print("Kernel size", kernel.shape[0], "number of parameters", A.shape[1])
        print("Gradient norm:", report["pgrad"])
        print("Dot product between gradient and minSR step:", report["SRdot"])
        print("Step norm:", report["step_norm"])
    return dp, report


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
        )
        block = sample_derivatives(
            wf, coords, transform, enacc, nodal_cutoff, client, npartitions
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


def minsr_optimization(
    wf,
    coords,
    transform,
    enacc,
    tstep=0.02,
    eps=1e-2,
    nodal_cutoff=1e-3,
    inverse_strategy="regularized_inverse",
    max_iterations=30,
    warmup_options=None,
    vmcoptions=None,
    max_norm=None,
    verbose=False,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Optimize the energy with the minSR algorithm.

    This is an alternative to :func:`pyqmc.method.linemin.line_minimization`. It
    differs in two ways:

    * no line minimization; `tstep` is a fixed hyperparameter, and
    * the S matrix is never constructed, so the memory cost is set by the number
      of samples rather than by the square of the number of parameters.

    Unlike line_minimization, this takes the parameter transform and the energy
    accumulator directly rather than a StochasticReconfiguration object, since
    the S matrix that object builds is exactly what minSR avoids::

        transform = LinearTransform(wf.parameters, to_opt)
        enacc = EnergyAccumulator(mol)
        wf, df = minsr_optimization(wf, coords, transform, enacc)

    `transform` may also be a list of transforms, in which case each iteration
    runs one sub-iteration per transform, optimizing that subset of the
    parameters while holding the rest fixed. Each sub-iteration draws its own
    samples, so this trades wall time for a smaller stored derivative matrix.
    It is less useful here than in line_minimization, since minSR's memory is
    already set by the sample count rather than by nparameters squared.

    `tstep` plays the same role as `steprange` in line_minimization: for the same
    `eps`, the update direction and magnitude are identical to the SR step taken
    at that step size.

    `eps` matters more here than it does in line_minimization, which can reject
    an overshoot with its line search. On H2/ccECP with 1000 walkers, eps=1e-3
    (the default of :func:`pyqmc.observables.accumulators.gradient_generator`)
    diverged in 2 of 20 runs, while eps=1e-2 converged in 20 of 20 and reached a
    lower energy than eps=1e-1, which was stable but over-damped. Set `max_norm`
    if steps still occasionally overshoot.

    :parameter wf: initial wave function
    :parameter coords: initial configurations
    :parameter transform: a LinearTransform object defining the parameters to optimize,
        or a list of them to optimize in alternating sub-iterations
    :parameter enacc: an EnergyAccumulator-like object
    :parameter float tstep: step size in parameter space
    :parameter float eps: regularization of the kernel
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    :parameter int max_iterations: total number of optimization steps, including any read from hdf_file
    :parameter dict warmup_options: kwargs for the initial vmc warmup
    :parameter dict vmcoptions: kwargs for sampling; nblocks, nsteps_per_block, and tstep are passed to sample_minsr_data
    :parameter float max_norm: if not None, the maximum 2-norm of a parameter change
    :parameter boolean verbose: print output if True
    :parameter str hdf_file: hdf file to store output; the format matches line_minimization
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :return: optimized wave function, optimization data
    """
    transforms = list(transform) if isinstance(transform, (list, tuple)) else [transform]
    for t in transforms:
        if hasattr(t, "transform"):
            raise ValueError(
                "minsr_optimization takes a transform and an energy accumulator, not a "
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
        "inverse_strategy": inverse_strategy,
    }

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
            if len(data["block_energy"]) > 1:
                block_energy = data["block_energy"].real
                energy_error = np.std(block_energy) / np.sqrt(len(block_energy))
            else:
                energy_error = np.std(data["total"].real) / np.sqrt(nsamples)
            if verbose:
                print("Current energy", energy, energy_error)

            dp, report = minsr_update(
                data["dppsi"],
                data["total"],
                tstep,
                eps=eps,
                inverse_strategy=inverse_strategy,
                max_norm=max_norm,
                verbose=verbose,
            )

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
