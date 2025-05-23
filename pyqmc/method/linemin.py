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

import numpy as np
import pyqmc.gpu as gpu
import pyqmc.method.sample_many as sm
import h5py
import os
import pyqmc.method.mc
import copy
import logging


def opt_hdf(hdf_file, data, attr, configs, parameters):
    import pyqmc.method.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
                hdf.create_group("wf")
                for k, it in parameters.items():
                    hdf.create_dataset("wf/" + k, data=gpu.asnumpy(it))
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)
            for k, it in parameters.items():
                hdf["wf/" + k][...] = gpu.asnumpy(it.copy())


def polyfit_relative(xfit, yfit, degree):
    p = np.polyfit(xfit, yfit, degree)
    ypred = np.polyval(p, xfit)
    resid = (ypred - yfit) ** 2
    relative_error = np.var(resid) / np.var(yfit)
    return p, relative_error


def stable_fit(xfit, yfit, tolerance=1e-2):
    """Fit a line and quadratic to xfit and yfit.
    1. If the linear fit is as good as the quadriatic, choose the lower endpoint.
    2. If the curvature is positive, estimate the minimum x value.
    3. If the lowest yfit is less than the new guess, use that xfit instead.

    We've stopped using this function because it sometimes thinks that the quadratic fit is bad and instead uses the linear fit, and it ends up not moving parameters or moving backwards (or moving forwards too much!). We've replaced it with find_minimum.

    :parameter list xfit: scalar step sizes along line
    :parameter list yfit: estimated energies at xfit points
    :parameter float tolerance: how good the quadratic fit needs to be
    :returns: estimated x-value of minimum
    :rtype: float
    """
    steprange = np.max(xfit)
    minstep = np.min(xfit)
    a = np.argmin(yfit)
    pq, relative_errq = polyfit_relative(xfit, yfit, 2)
    pl, relative_errl = polyfit_relative(xfit, yfit, 1)

    if relative_errl / relative_errq < 2:  # If a linear fit is about as good..
        if pl[0] < 0:
            est_min = steprange
        else:
            est_min = minstep
        out_y = np.polyval(pl, est_min)
    elif relative_errq < tolerance and pq[0] > 0:  # If quadratic fit is good
        est_min = -pq[1] / (2 * pq[0])
        if est_min > steprange:
            est_min = steprange
        if est_min < minstep:
            est_min = minstep
        out_y = np.polyval(pq, est_min)
    else:
        est_min = xfit[a]
        out_y = yfit[a]
    if (
        out_y > yfit[a]
    ):  # If min(yfit) has a lower energy than the guess, use it instead
        est_min = xfit[a]
    return est_min


def find_minimum(xfit, yfit):
    """
    Here we just take the minimum of the yfit. This is a simpler version of stable_fit.
    """
    a = np.argmin(yfit)
    est_min = xfit[a]
    return est_min


def line_minimization(
    wf,
    coords,
    pgrad_acc,
    steprange=0.2,
    max_iterations=30,
    warmup_options=None,
    correlated_reference_wfs=None,
    stderr_weight=3.0,
    vmcoptions=None,
    lmoptions=None,
    correlatedoptions=None,
    update_kws=None,
    verbose=False,
    npts=40,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Optimizes energy by determining gradients with stochastic reconfiguration
        and minimizing the energy along gradient directions using correlated sampling.

    :parameter wf: initial wave function
    :parameter coords: initial configurations
    :parameter pgrad_acc: A PGradAccumulator-like object
    :parameter float steprange: How far to search in the line minimization. If you find that est_min is always at the edge of steprange, you may need to increase steprange. If you find that est_min is always much smaller than steprange, decrease this.
    :parameter int max_iterations: (maximum) number of steps in the gradient descent. If the calculation is continued from the same hdf file, the iterations from previous runs are included in the total, i.e. when calling line_minimization multiple times with the same hdf_file, max_iterations is the total number of iterations that will be run.
    :parameter dict warmup_options: kwargs to use for vmc warmup
    :parameter list correlated_reference_wfs: which wave functions to use for correlated sampling (default [0,1])
    :parameter int stderr_weight: weight of the standard error in the line minimization. Increase this if your optimization is struggling to find downhill directions.
    :parameter dict vmcoptions: a dictionary of options for the vmc method
    :parameter dict lmoptions: a dictionary of options for the lm method
    :parameter update: A function that generates a parameter change
    :parameter update_kws: Any keywords
    :parameter boolean verbose: print output if True
    :parameter int npts: number of points to fit to in each line minimization
    :parameter str hdf_file: Hdf_file to store vmc output.
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :return: optimized wave function, optimization data
    """

    if vmcoptions is None:
        vmcoptions = {}
    vmcoptions.update({"verbose": verbose})
    if lmoptions is None:
        lmoptions = {}
    if correlatedoptions is None:
        correlatedoptions = dict(nsteps=3, nblocks=1)
    if update_kws is None:
        update_kws = {}
    if warmup_options is None:
        warmup_options = dict(nblocks=1, nsteps_per_block=100)
    if "tstep" not in warmup_options and "tstep" in vmcoptions:
        warmup_options["tstep"] = vmcoptions["tstep"]
    assert npts >= 3, f"linemin npts={npts}; need npts >= 3 for correlated sampling"
    if correlated_reference_wfs is None:
        correlated_reference_wfs = [0, 1]

    iteration_offset = 0
    sub_iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                grp = hdf["wf"]
                for k in grp.keys():
                    wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf.keys():
                iteration_offset = np.max(hdf["iteration"][...])
            if "sub_iteration" in hdf.keys():
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
            **warmup_options,
        )
        if verbose:
            print("finished warmup", flush=True)
    if iteration_offset >= max_iterations:
        logging.warning(
            f"iteration_offset {iteration_offset} >= max_iterations {max_iterations}; no steps will be run."
        )

    # Attributes for linemin
    attr = dict(max_iterations=max_iterations,
                npts=npts,
                steprange=steprange,
                correlated_reference_wfs=correlated_reference_wfs)
    try:
        sub_iterations = len(pgrad_acc)
    except TypeError:
        if verbose:
            print("Was passed a single PGradAccumulator; using 1 sub_iteration. This is deprecated behavior.")
        sub_iterations = 1
        pgrad_acc = [pgrad_acc]

    df = []
    # Gradient descent cycles
    for it in range(iteration_offset, max_iterations):
        for sub_it in range(sub_iteration_offset, sub_iterations):
            if verbose: 
                print("#############################\nStarting iteration", it, "sub iteration", sub_it)
            pgrad = pgrad_acc[sub_it]
            x0 = pgrad.transform.serialize_parameters(wf.parameters)

            df_vmc, coords = pyqmc.method.mc.vmc(
                wf,
                coords,
                accumulators={"pgrad": pgrad},
                client=client,
                npartitions=npartitions,
                **vmcoptions,
            )

            data = {}
            for k in pgrad.keys():
                data[k] = np.mean(df_vmc["pgrad" + k], axis=0)
            data["total_err"] = np.std(df_vmc["pgradtotal"], axis=0) / np.sqrt(
                df_vmc["pgradtotal"].shape[0]
            )
            if np.isnan(df_vmc["pgradtotal"]).any():
                raise ValueError("NaN in optimization. Try reducing the step size or increasing stabilization.")
            if verbose:
                print("Current energy", data["total"], data["total_err"])

            step_data = {}
            step_data["energy"] = data["total"].real
            step_data["energy_error"] = data["total_err"].real
            #step_data["x"] = x0
            step_data["iteration"] = it
            step_data['sub_iteration'] = sub_it
            step_data["nconfig"] = coords.configs.shape[0]

            # Correlated sampling line minimization.
            steps = np.linspace(-steprange / (npts - 2), steprange, npts)
            dps, update_report = pgrad.delta_p(steps, data, verbose=verbose)
            step_data.update(update_report)
            params = [x0 + dp for dp in dps]

            correlated_data = correlated_compute(
                wf,
                coords,
                params,
                pgrad,
                client=client,
                npartitions=npartitions,
                ref_wfs=correlated_reference_wfs
            )

            w = correlated_data["weight"].copy()
            w = w / np.mean(w, axis=1, keepdims=True)
            en = np.real(np.mean(correlated_data["total"] * w, axis=1))
            en_std = np.std(correlated_data["total"], axis=1)
            yfit = en + stderr_weight*en_std
            est_min = find_minimum(steps, yfit)

            x0 = pgrad.delta_p([est_min], data, verbose=False)[0][0] + x0

            step_data["tau"] = steps
            step_data["yfit"] = yfit
            step_data["est_min"] = est_min
            step_data["correlated_energy"] = en
            step_data["correlated_energy_std"] = en_std

            if verbose:
                print("energies from correlated sampling", en)
                print("Chose to move", est_min, "from", steps[0], "to", steps[-1])
                print("weight variance", np.var(w, axis=1))
                print("avg weights", np.mean(correlated_data['weight'], axis=1))
                print("energy standard deviation", en_std)

            set_wf_params(wf, x0, pgrad)
            opt_hdf(
                hdf_file, step_data, attr, coords, wf.parameters
            )
            df.append(step_data)

        sub_iteration_offset = 0
    return wf, df


def correlated_compute(
    wf, configs, params, pgrad_acc, client=None, npartitions=None, ref_wfs=None, **kws
):
    """
    Evaluates energy on the same set of configs for correlated sampling of different wave function parameters

    :parameter wf: wave function object
    :parameter configs: (nconf, nelec, 3) array
    :parameter params: (nsteps, nparams) array
        list of arrays of parameters (serialized) at each step
    :parameter pgrad_acc: PGradAccumulator
    :returns: a single dict with indices [parameter, values]

    """
    if ref_wfs is None:
        ref_wfs = [0, 1]

    wfs = [copy.deepcopy(wf) for i in ref_wfs]
    for i in ref_wfs:
        set_wf_params(wfs[i], params[i], pgrad_acc)
    # sample combined distribution
    _, _, configs = sm.sample_overlap(
        wfs, configs, None, client=client, npartitions=npartitions, **kws
    )

    if client is None:
        return correlated_compute_worker(wf, configs, params, pgrad_acc, ref_wfs)
    config = configs.split(npartitions)
    runs = [
        client.submit(correlated_compute_worker, wf, conf, params, pgrad_acc, ref_wfs)
        for conf in config
    ]
    allresults = [r.result() for r in runs]
    block_avg = {}
    for k in allresults[0].keys():
        block_avg[k] = np.hstack([res[k] for res in allresults])
    return block_avg


def correlated_compute_worker(wf, configs, params, pgrad_acc, ref_wfs):
    """
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters

    :parameter wf: wave function object
    :parameter configs: (nconf, nelec, 3) array
    :parameter params: (nsteps, nparams) array
        list of arrays of parameters (serialized) at each step
    :parameter pgrad_acc: PGradAccumulator
    :parameter ref_wfs: Reference wave functions. assume that rho = sum_i exp(2*(psi_i))

    :returns: a single dict with indices [parameter, values]

    """

    data = []
    current_state = np.random.get_state()
    psi = np.zeros((len(params), len(configs.configs)))
    for i, p in enumerate(params):
        np.random.set_state(current_state)
        set_wf_params(wf, p, pgrad_acc)
        psi[i] = wf.recompute(configs)[1]  # recompute gives logdet
        df = pgrad_acc.enacc(configs, wf)
        data.append(df)

    data_ret = sm.invert_list_of_dicts(data)

    ref = np.amax(psi, axis=0)
    psirel = np.exp(2 * (psi - ref))
    rho = np.mean([psirel[i] for i in ref_wfs], axis=0)
    data_ret["weight"] = psirel / rho
    return data_ret


def set_wf_params(wf, params, pgrad_acc):
    newparms = pgrad_acc.transform.deserialize(wf, params)
    for k in newparms:
        wf.parameters[k] = newparms[k]
