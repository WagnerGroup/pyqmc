import numpy as np
import pyqmc.gpu as gpu
import pyqmc.sample_many as sm
import scipy
import h5py
import os
import pyqmc.mc
import copy
import logging


def sr_update(pgrad, Sij, step, eps=0.1):
    invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step  # / np.linalg.norm(v)


def sd_update(pgrad, Sij, step, eps=0.1):
    return -pgrad * step  # / np.linalg.norm(pgrad)


def sr12_update(pgrad, Sij, step, eps=0.1):
    invSij = scipy.linalg.sqrtm(np.linalg.inv(Sij + eps * np.eye(Sij.shape[0])))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step  # / np.linalg.norm(v)


def opt_hdf(hdf_file, data, attr, configs, parameters):
    import pyqmc.hdftools as hdftools

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


def line_minimization(
    wf,
    coords,
    pgrad_acc,
    steprange=0.2,
    max_iterations=30,
    warmup_options=None,
    vmcoptions=None,
    lmoptions=None,
    correlatedoptions=None,
    update=sr_update,
    update_kws=None,
    verbose=False,
    npts=5,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Optimizes energy by determining gradients with stochastic reconfiguration
        and minimizing the energy along gradient directions using correlated sampling.

    :parameter wf: initial wave function
    :parameter coords: initial configurations
    :parameter pgrad_acc: A PGradAccumulator-like object
    :parameter float steprange: How far to search in the line minimization
    :parameter int max_iterations: (maximum) number of steps in the gradient descent. If the calculation is continued from the same hdf file, the iterations from previous runs are included in the total, i.e. when calling line_minimization multiple times with the same hdf_file, max_iterations is the total number of iterations that will be run. 
    :parameter int warmup_options: kwargs to use for vmc warmup
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

    iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                grp = hdf["wf"]
                for k in grp.keys():
                    wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf.keys():
                iteration_offset = np.max(hdf["iteration"][...]) + 1
            coords.load_hdf(hdf)
    else:  # not restarting -- VMC warm up period
        if verbose:
            print("starting warmup")
        _, coords = pyqmc.mc.vmc(
            wf,
            coords,
            accumulators={},
            client=client,
            npartitions=npartitions,
            **warmup_options,
        )
        if verbose:
            print("finished warmup", flush=True)

    # Attributes for linemin
    attr = dict(max_iterations=max_iterations, npts=npts, steprange=steprange)

    def gradient_energy_function(x, coords):
        set_wf_params(wf, x, pgrad_acc)
        df, coords = pyqmc.mc.vmc(
            wf,
            coords,
            accumulators={"pgrad": pgrad_acc},
            client=client,
            npartitions=npartitions,
            **vmcoptions,
        )
        en = np.real(np.mean(df["pgradtotal"], axis=0))
        en_err = np.std(df["pgradtotal"], axis=0) / np.sqrt(df["pgradtotal"].shape[0])
        sigma = np.std(df["pgradtotal"], axis=0) * np.sqrt(np.mean(df["nconfig"]))
        dpH = np.mean(df["pgraddpH"], axis=0)
        dp = np.mean(df["pgraddppsi"], axis=0)
        dpdp = np.mean(df["pgraddpidpj"], axis=0)
        grad = 2 * np.real(dpH - en * dp)
        Sij = np.real(dpdp - np.einsum("i,j->ij", dp, dp))

        if np.any(np.isnan(grad)):
            for nm, quant in {"dpH": dpH, "dp": dp, "en": en}.items():
                print(nm, quant)
            raise ValueError("NaN detected in derivatives")

        return coords, grad, Sij, en, en_err, sigma

    x0 = pgrad_acc.transform.serialize_parameters(wf.parameters)

    df = []
    if iteration_offset >= max_iterations:
        logging.warning(f"iteration_offset {iteration_offset} >= max_iterations {max_iterations}; no steps will be run.")
    # Gradient descent cycles
    for it in range(iteration_offset, max_iterations):
        # Calculate gradient accurately
        coords, pgrad, Sij, en, en_err, sigma = gradient_energy_function(x0, coords)
        step_data = {}
        step_data["energy"] = en
        step_data["energy_error"] = en_err
        step_data["x"] = x0
        step_data["pgradient"] = pgrad
        step_data["iteration"] = it
        step_data["nconfig"] = coords.configs.shape[0]

        if verbose:
            print("descent en", en, en_err, " estimated sigma ", sigma)
            print("descent |grad|", np.linalg.norm(pgrad), flush=True)

        xfit = []
        yfit = []

        # Calculate samples to fit.
        # include near zero in the fit, and go backwards as well
        # We don't use the above computed value because we are
        # doing correlated sampling.
        steps = np.linspace(-steprange / (npts - 2), steprange, npts)
        params = [x0 + update(pgrad, Sij, step, **update_kws) for step in steps]
        stepsdata = correlated_compute(
            wf,
            coords,
            params,
            pgrad_acc,
            client=client,
            npartitions=npartitions,
        )

        w = stepsdata["weight"]
        w = w / np.mean(w, axis=1, keepdims=True)
        en = np.real(np.mean(stepsdata["total"] * w, axis=1))
        yfit.extend(en)
        xfit.extend(steps)
        est_min = stable_fit(xfit, yfit)
        x0 += update(pgrad, Sij, est_min, **update_kws)
        step_data["tau"] = xfit
        step_data["yfit"] = yfit
        step_data["est_min"] = est_min

        opt_hdf(
            hdf_file, step_data, attr, coords, pgrad_acc.transform.deserialize(wf, x0)
        )
        df.append(step_data)

    set_wf_params(wf, x0, pgrad_acc)

    return wf, df


def correlated_compute(
    wf, configs, params, pgrad_acc, client=None, npartitions=None, **kws
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

    data = []
    wfs = [copy.deepcopy(wf) for i in [0, -1]]
    for p, wf_ in zip(params, wfs):
        set_wf_params(wf_, p, pgrad_acc)
    # sample combined distribution
    _, _, configs = sm.sample_overlap(
        wfs, configs, None, client=client, npartitions=npartitions, **kws
    )

    if client is None:
        return correlated_compute_worker(wf, configs, params, pgrad_acc)
    config = configs.split(npartitions)
    runs = [
        client.submit(correlated_compute_worker, wf, conf, params, pgrad_acc)
        for conf in config
    ]
    allresults = [r.result() for r in runs]
    block_avg = {}
    for k in allresults[0].keys():
        block_avg[k] = np.hstack([res[k] for res in allresults])
    return block_avg


def correlated_compute_worker(wf, configs, params, pgrad_acc):
    """
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters

    :parameter wf: wave function object
    :parameter configs: (nconf, nelec, 3) array
    :parameter params: (nsteps, nparams) array
        list of arrays of parameters (serialized) at each step
    :parameter pgrad_acc: PGradAccumulator
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
    rho = np.mean([psirel[i] for i in [0, -1]], axis=0)
    data_ret["weight"] = psirel / rho
    return data_ret


def set_wf_params(wf, params, pgrad_acc):
    newparms = pgrad_acc.transform.deserialize(wf, params)
    for k in newparms:
        wf.parameters[k] = newparms[k]
