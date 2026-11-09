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



import os
import time

import h5py
import numpy as np

import pyqmc.method.sample_many
import pyqmc.rng
from pyqmc.method import hdftools
from pyqmc.method.ensemble_sampling import (
    warmup_overlap,
    evaluate_gradients_threaded,
    sample_gradient_configs_threaded,
)
from pyqmc.method.ensemble_updaters import EnsembleUpdater, build_updaters, make_updater  # noqa: F401
from pyqmc.method.optimize import set_wf_params


def _configs_to_hdf(hdf, configs):
    if "configs" not in hdf:
        configs.initialize_hdf(hdf)
    configs.to_hdf(hdf)


def load_all_configs(hdf, configs, updater):
    """
    Load all optimizer configs, or return `None` if the checkpoint has none.

    Args:
        hdf: the HDF5 object to load the configurations from
        configs: configs that supply the configuration type and shape
        updater: nested list of EnsembleUpdater objects indexed by state then by sub-iteration

    Return:
        norm_configs: configs used for normalization sampling
        gradient_configs (list[list[dict]]): nested list of configs used for gradient sampling,
            indexed by state, sub-iteration, then kind ("energy" or "overlap")
    """
    if "all_configs" not in hdf:
        return None

    gradient_configs = []
    norm_configs = configs.copy()
    norm_configs.load_hdf(hdf["all_configs/normalization"])
    for wfi, transform_list in enumerate(updater):
        state_configs = []
        for sub_iteration in range(len(transform_list)):
            configs_by_kind = {}
            for kind in ("energy", "overlap"):
                cfg = configs.copy()
                cfg.load_hdf(
                    hdf[f"all_configs/gradient/{wfi}/{sub_iteration}/{kind}"]
                )
                configs_by_kind[kind] = cfg
            state_configs.append(configs_by_kind)
        gradient_configs.append(state_configs)
    return norm_configs, gradient_configs


def hdf_save(hdf_file, data, attr, wfs, norm_configs, gradient_configs):
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            for k, it in attr.items():
                if k not in hdf.attrs:
                    hdf.attrs[k] = it
            for wfi, wf in enumerate(wfs):
                if f"wf/{wfi}" not in hdf.keys():
                    hdf.create_group(f"wf/{wfi}")
                    for k, it in wf.parameters.items():
                        hdf[f"wf/{wfi}/" + k] = it.copy()

            hdftools.append_hdf(hdf, data)
            all_configs_group = hdf.require_group("all_configs")
            _configs_to_hdf(all_configs_group.require_group("normalization"), norm_configs)
            gradient_group = all_configs_group.require_group("gradient")
            for wfi, state_configs in enumerate(gradient_configs):
                for sub_iteration, configs_by_kind in enumerate(state_configs):
                    for kind, cfg in configs_by_kind.items():
                        group = gradient_group.require_group(f"{wfi}/{sub_iteration}/{kind}")
                        _configs_to_hdf(group, cfg)
            for wfi, wf in enumerate(wfs):
                for k, it in wf.parameters.items():
                    hdf[f"wf/{wfi}/" + k][:] = it.copy()


def renormalize(wfs, norms, pivot=0, N=1):
    """
    Renormalize the wave functions so that they have the same normalization as the pivot wave function.

    .. math::

    """
    for i, wf in enumerate(wfs):
        if i == pivot:
            continue
        renorm = np.sqrt(norms[pivot] / norms[i] * N)
        if "wf1det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["wf1det_coeff"] = wf.parameters["wf1det_coeff"] * renorm
        elif "det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["det_coeff"] = wf.parameters["det_coeff"] * renorm
        else:
            raise NotImplementedError("need wf1det_coeff or det_coeff in parameters")


def _format_step_report(report):
    """The diagnostics an updater returns from delta_p, on one line.

    pgrad and SRdot are common to every updater; anything else it reports --
    the overlap penalty cost, CG iteration counts, how long the solve took -- is
    appended as it comes, so that a new updater's diagnostics show up without
    touching the driver. Keys ending in `_seconds` are printed as times, and
    flags that are True are left out, having nothing to report.
    """
    named = {"pgrad": "|grad|", "SRdot": "grad.step"}
    parts = [
        f"{named[key]} = {float(np.real(report[key])):.4g}"
        for key in named
        if key in report
    ]
    for key, value in report.items():
        if key in named or np.ndim(value) != 0 or isinstance(value, str):
            continue
        if isinstance(value, (bool, np.bool_)):
            if not value:  # a flag is worth printing only when it is a problem
                parts.append(f"{key} = False")
            continue
        if key.endswith("_seconds"):  # any updater can report a timing this way
            parts.append(f"{key[: -len('_seconds')]} = {float(value):.3f}s")
            continue
        parts.append(f"{key} = {float(np.real(value)):.4g}")
    return "   ".join(parts)




def _restore(hdf_file, wfs, configs, updater):
    """Read parameters, and configs if present, from a checkpoint.

    :returns: (first iteration to run, (norm_configs, gradient_configs) or None)
    """
    if hdf_file is None or not os.path.isfile(hdf_file):
        return 0, None
    with h5py.File(hdf_file, "r") as hdf:
        if "wf" in hdf.keys():
            for wfi, wf in enumerate(wfs):
                grp = hdf[f"wf/{wfi}"]
                for k in grp.keys():
                    wf.parameters[k] = np.asarray(grp[k])
        first_iteration = np.max(hdf["iteration"][...]) + 1 if "iteration" in hdf else 0
        return first_iteration, load_all_configs(hdf, configs, updater)


def _initial_configs(wfs, configs, updater, all_configs, vmc_warmup_kwargs, overlap_warmup_kwargs, **kwargs):
    """Build every set of configs and equilibrate each under its own distribution.

    :parameter kwargs: client, npartitions, overlap_thread_weight, verbose, rng
    :returns: (norm_configs, gradient_configs)
    """
    if all_configs is None:
        norm_configs = configs.copy()
        gradient_configs = [
            [{"energy": configs.copy(), "overlap": configs.copy()} for _ in state]
            for state in updater
        ]
    else:
        norm_configs, gradient_configs = all_configs
    norm_configs = warmup_overlap(
        wfs, norm_configs, kwargs["client"], kwargs["npartitions"], overlap_warmup_kwargs, kwargs["rng"]
    )
    _, _, _, gradient_configs = sample_gradient_configs_threaded(
        wfs,
        gradient_configs,
        updater=None,
        vmc_kwargs=vmc_warmup_kwargs,
        overlap_kwargs=overlap_warmup_kwargs,
        **kwargs,
    )
    return norm_configs, gradient_configs


def _normalize(wfs, norm_configs, refresh_kwargs, norm_kwargs, client, npartitions, verbose, rng):
    """Sample the norms of all states and rescale each to match state 0.

    The norms only set a rescaling of the wave functions, so a couple of blocks
    is plenty; see norm_kwargs in optimize_ensemble.

    :returns: norm_configs
    """
    norm_configs = warmup_overlap(wfs, norm_configs, client, npartitions, refresh_kwargs, rng)
    _, data_unweighted, norm_configs = pyqmc.method.sample_many.sample_overlap(
        wfs, norm_configs, None, client=client, npartitions=npartitions, rng=rng, **norm_kwargs
    )
    norm = np.mean(data_unweighted["overlap"], axis=0)
    if verbose:
        diag = np.array2string(np.real(norm.diagonal()), precision=4, suppress_small=True)
        print(f"  normalization: {diag}", flush=True)
    renormalize(wfs, norm.diagonal(), pivot=0)
    return norm_configs


def _print_state(label, wfi, avg, error, report):
    line = (
        f"  {label}: E = {float(np.real(avg['energy'])):.6f}"
        f" +/- {float(np.real(error['energy'])):.6f}"
    )
    if wfi > 0:  # overlap with the states below this one
        lower = np.array2string(np.abs(avg["overlap"][wfi, :wfi]), precision=4, suppress_small=True)
        line += f"   overlap with lower states {lower}"
    print(line, flush=True)
    print(f"      {_format_step_report(report)}", flush=True)


def _update_state(wf, wfi, updater, energy_data, overlap_weighted, overlap_unweighted, tau, overlap_penalty):
    """Take one step on one state's parameters.

    :returns: (avg, error, report)
    """
    avg, error = updater.block_average(energy_data, overlap_weighted, overlap_unweighted["overlap"])
    start = time.perf_counter()
    dp, report = updater.delta_p([tau], avg, overlap_penalty, verbose=False)
    # the solve does not parallelize the way the sampling does, so it is worth
    # seeing next to the per-job sampling times
    report["delta_p_seconds"] = time.perf_counter() - start
    x = updater.transform.serialize_parameters(wf.parameters) + dp[0]
    set_wf_params(wf, x, updater.transform)
    return avg, error, report


def optimize_ensemble(
    wfs,
    configs,
    transforms,
    hdf_file,
    enacc=None,
    method="sr",
    eps=None,
    nodal_cutoff=1e-3,
    tau=0.01,
    max_iterations=100,
    overlap_penalty=None,
    npartitions=None,
    client=None,
    verbose=True,
    overlap_thread_weight=None,
    vmc_kwargs=None,
    overlap_kwargs=None,
    norm_kwargs=None,
    initial_vmc_warmup_kwargs=None,
    initial_overlap_warmup_kwargs=None,
    refresh_vmc_warmup_kwargs=None,
    refresh_overlap_warmup_kwargs=None,
    all_configs=None,
    rng=None,
):
    """
    Optimize a set of wave functions using ensemble VMC.

    Separate configurations are maintained for the all-wf overlap, the vmc, and the prefix-overlap distributions.
    Initial warmups are performed by default, and are skipped when all configurations can be restored from a checkpoint.
    Empty individual warmup dictionaries disable the corresponding warmups.
    Refresh warmups before each iteration's measurements are off by default: every set of configs is already
    propagated by its own measurement each iteration, and the measurement sampling is doing almost exactly what
    a refresh warmup would do. Enable them by passing the corresponding dictionaries if a case needs it.
    Starting configs precedence:
        1. If `hdf_file` exists, restart from its parameters and configs; initial warmups are skipped.
           A checkpoint without configs restores only the parameters.
        2. Otherwise, use `(norm_configs, gradient_configs)` from the supplied `all_configs` (optional)
        3. Otherwise, construct all configs from `configs`

    Args:
        wfs (list): list of wave functions to be optimized
        configs: initial configs before warmup if not loaded from a checkpoint
        transforms (list): one LinearTransform per state, or a nested list indexed
            by state then by sub-iteration. Already-built updaters are accepted
            here too, in which case enacc, method, eps, and nodal_cutoff are unused.
        hdf_file (str): path for the checkpoint file
        enacc: an EnergyAccumulator-like object shared by every state, or a list
            with one per state. Required unless transforms are already updaters.
        method (str): 'sr' for stochastic reconfiguration, which builds the
            (nparameters, nparameters) S matrix; 'minsr', which solves the same
            equations in sample space without building it; or 'cgsr', which
            solves them by conjugate gradient and builds neither that matrix nor
            the sample-space kernel.
            minSR is worth it when there are more parameters than samples. Note
            that the number of samples per state is
            nconfig * vmc_kwargs['nblocks'], and the kernel it solves is square
            in that, which is why it defaults to one block rather than ten.
            cgsr shares that default because it shares the sampling, but it does
            not share the reason: its cost and memory are both linear in the
            sample count, so it is the one to raise nblocks on if the statistics
            need it. See pyqmc.method.cgsr for where the crossover sits.
        eps (float): regularization of the solve; defaults to what the method
            wants, 1e-3 for sr and 1e-2 for minsr and cgsr
        nodal_cutoff (float): regularization distance for the nodal divergence of
            the parameter derivatives
        tau (float): step size along the SR direction, which uses the true
            gradient of the objective in pyqmc.method.ensemble_updaters
        max_iterations (int): maximum number of optimization iterations
        overlap_penalty (np.ndarray): (nwf, nwf) penalty lambda; state i is optimized on
            E_i + sum_{j<i} lambda_ij |S_ij|^2 with S the normalized overlap. Defaults to ones.
        npartitions (int): total number of partitions distributed among the threads
        client: an object with submit() functions that return futures
        overlap_thread_weight (list): a list of float that overrides the default thread weights (1 + wfi) / 2.0
        vmc_kwargs (dict): options for measurement `vmc`; None gives nblocks=1 for minsr
            and cgsr and 10 for sr, with nsteps_per_block=10
        overlap_kwargs (dict): options for measurement `sample_overlap`, with the same default
        norm_kwargs (dict): options for the normalization `sample_overlap`; defaults to
            overlap_kwargs with nblocks=2, since the norms only set a rescaling and
            only matter to within a factor of two or so
        initial_vmc_warmup_kwargs (dict): options for initial warmup `vmc`; an empty dictionary disables it
        initial_overlap_warmup_kwargs (dict): options for initial warmup `sample_overlap`; an empty dictionary disables it
        refresh_vmc_warmup_kwargs (dict): options for refresh warmup `vmc`; defaults to off, pass a dictionary to enable
        refresh_overlap_warmup_kwargs (dict): options for refresh warmup `sample_overlap`; defaults to off, pass a dictionary to enable
        all_configs (tuple): `(norm_configs, gradient_configs)`, a full set of configs to start the optimization
        rng: anything accepted by :func:`pyqmc.rng.get_rng`; None uses pyqmc's module generator

    Return:
        wfs (list): list of optimized wave functions
    """

    rng = pyqmc.rng.get_rng(rng)
    if initial_vmc_warmup_kwargs is None:
        initial_vmc_warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if initial_overlap_warmup_kwargs is None:
        initial_overlap_warmup_kwargs = dict(nblocks=1, nsteps_per_block=100)
    if refresh_vmc_warmup_kwargs is None:
        refresh_vmc_warmup_kwargs = {}
    if refresh_overlap_warmup_kwargs is None:
        refresh_overlap_warmup_kwargs = {}
    # minsr and cgsr both keep the derivatives per configuration, so they share
    # these defaults; sr averages them into dpidpj and wants more blocks
    default_kwargs = dict(nblocks=1 if method in ("minsr", "cgsr") else 10, nsteps_per_block=10)
    if vmc_kwargs is None:
        vmc_kwargs = default_kwargs
    if overlap_kwargs is None:
        overlap_kwargs = default_kwargs
    if norm_kwargs is None:
        norm_kwargs = dict(overlap_kwargs, nblocks=2)
    updater = build_updaters(transforms, enacc, method, eps, nodal_cutoff)
    if len(updater) != len(wfs):
        raise ValueError(f"got transforms for {len(updater)} states but {len(wfs)} wave functions")
    nwf = len(wfs)
    if overlap_penalty is None:
        overlap_penalty = np.ones((nwf, nwf))
    parallel = dict(client=client, npartitions=npartitions)

    first_iteration, restored = _restore(hdf_file, wfs, configs, updater)
    if restored is not None:
        norm_configs, gradient_configs = restored
    else:
        norm_configs, gradient_configs = _initial_configs(
            wfs,
            configs,
            updater,
            all_configs,
            initial_vmc_warmup_kwargs,
            initial_overlap_warmup_kwargs,
            overlap_thread_weight=overlap_thread_weight,
            verbose=verbose,
            rng=rng,
            **parallel,
        )

    for i in range(first_iteration, max_iterations):
        if verbose:
            print(f"\n=== Iteration {i} ===", flush=True)
        norm_configs = _normalize(
            wfs, norm_configs, refresh_overlap_warmup_kwargs, norm_kwargs, verbose=verbose, rng=rng, **parallel
        )
        energy_data, overlap_weighted, overlap_unweighted, gradient_configs = evaluate_gradients_threaded(
            wfs,
            gradient_configs,
            updater,
            vmc_kwargs=vmc_kwargs,
            overlap_kwargs=overlap_kwargs,
            refresh_vmc_warmup_kwargs=refresh_vmc_warmup_kwargs,
            refresh_overlap_warmup_kwargs=refresh_overlap_warmup_kwargs,
            overlap_thread_weight=overlap_thread_weight,
            rng=rng,
            **parallel,
        )

        for wfi, wf in enumerate(wfs):
            for sub_iteration, state_updater in enumerate(updater[wfi]):
                avg, error, report = _update_state(
                    wf,
                    wfi,
                    state_updater,
                    energy_data[wfi][sub_iteration],
                    overlap_weighted[wfi][sub_iteration],
                    overlap_unweighted[wfi][sub_iteration],
                    tau,
                    overlap_penalty,
                )
                if verbose:
                    label = f"wf {wfi}" + (f" sub {sub_iteration}" if len(updater[wfi]) > 1 else "")
                    _print_state(label, wfi, avg, error, report)
                save_data = {
                    f"energy{wfi}": avg["energy"],
                    f"energy_error{wfi}": error["energy"],
                    f"overlap{wfi}": avg["overlap"],
                    "iteration": i,
                    "wavefunction": wfi,
                    "sub_iteration": sub_iteration,
                }
                hdf_save(hdf_file, save_data, {"tau": tau}, wfs, norm_configs, gradient_configs)

    return wfs
