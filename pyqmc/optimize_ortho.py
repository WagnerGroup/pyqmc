import numpy as np
import h5py
import pyqmc.linemin
import pyqmc.hdftools as hdftools
import pyqmc.mc as mc
import os


def ortho_hdf(hdf_file, data, attr, configs, parameters):

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                hdf.create_dataset("configs", configs.configs.shape)
                for k, it in parameters.items():
                    hdf.create_dataset("wf/" + k, data=it)

            hdftools.append_hdf(hdf, data)
            hdf["configs"][:, :, :] = configs.configs
            for k, it in parameters.items():
                hdf["wf/" + k][...] = it.copy()


def collect_overlap_data(wfs, configs, pgrad):
    r"""Collect the averages assuming that
    configs are distributed according to

    .. math:: \rho \propto \sum_i |\Psi_i|^2

    The keys 'overlap' and 'overlap_gradient' are

    `overlap` :

    .. math:: \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    `overlap_gradient`:

    .. math:: \partial_m \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\partial_{fm} \Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    """
    phase, log_vals = [
        np.nan_to_num(np.array(x)) for x in zip(*[wf.value() for wf in wfs])
    ]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    save_dat = {}
    # print('log_vals', log_vals)
    denominator = np.sum(np.exp(2 * (log_vals - ref)), axis=0)
    normalized_values = phase * np.exp(log_vals - ref)
    save_dat["overlap"] = np.einsum(  # shape (wf, wf)
        "ik,jk->ij", normalized_values.conj(), normalized_values / denominator
    ) / len(ref)

    dppsi = pgrad.transform.serialize_gradients(wfs[-1].pgradient())
    save_dat["overlap_gradient"] = np.einsum(
        "km,k,jk->jm",  # shape (wf, param)
        dppsi,
        normalized_values[-1].conj(),
        normalized_values / denominator,
    ) / len(ref)
    # print("ratio", save_dat["overlap"].diagonal())

    # Weight for quantities that are evaluated as
    # int( f(X) psi_f^2 dX )
    # since we sampled sum psi_i^2
    weight = np.exp(-2 * (log_vals[:, np.newaxis] - log_vals))
    weight = 1.0 / np.sum(weight, axis=1)

    dat = pgrad.avg(configs, wfs[-1], weight[-1])
    for k in dat.keys():
        save_dat[k] = dat[k]
    save_dat["weight_final"] = np.mean(weight[-1])
    return save_dat


def construct_rho_gradient(grads, log_values):
    total_grad = np.zeros_like(grads[0])
    for g, v in zip(grads, log_values):
        denominator = np.sum(np.exp(2 * np.real(log_values - v)))
        total_grad += g / denominator
    return total_grad


def sample_overlap_worker(wfs, configs, pgrad, nsteps, tstep=0.5):
    r"""Run nstep Metropolis steps to sample a distribution proportional to
    :math:`\sum_i |\Psi_i|^2`, where :math:`\Psi_i` = wfs[i]
    """
    nconf, nelec, _ = configs.configs.shape
    for wf in wfs:
        wf.recompute(configs)
    block_avg = {}
    block_avg["acceptance"] = np.zeros(nsteps)
    for n in range(nsteps):
        for e in range(nelec):  # a sweep
            # Propose move
            grads = [np.real(wf.gradient(e, configs.electron(e)).T) for wf in wfs]
            grad = mc.limdrift(np.mean(grads, axis=0))
            gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)

            # Compute reverse move
            grads, vals, savedvals = list(
                zip(*[wf.gradient_value(e, newcoorde) for wf in wfs])
            )
            grads = [np.real(g.T) for g in grads]
            new_grad = mc.limdrift(np.mean(grads, axis=0))
            forward = np.sum(gauss**2, axis=1)
            backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

            # Acceptance
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
            wf_ratios = np.abs(vals) ** 2
            log_values = np.real(np.array([wf.value()[1] for wf in wfs]))
            weights = np.exp(2 * (log_values - log_values[0]))

            ratio = t_prob * np.sum(wf_ratios * weights, axis=0) / weights.sum(axis=0)
            accept = ratio > np.random.rand(nconf)
            block_avg["acceptance"][n] += accept.mean() / nelec

            # Update wave function
            configs.move(e, newcoorde, accept)
            for wf, saved in zip(wfs, savedvals):
                wf.updateinternals(
                    e, newcoorde, configs, mask=accept, saved_values=saved
                )

        # Collect rolling average
        save_dat = collect_overlap_data(wfs, configs, pgrad)
        for k, it in save_dat.items():
            if k not in block_avg:
                block_avg[k] = np.zeros((*it.shape,), dtype=it.dtype)
            if k in ["overlap", "overlap_gradient", "weight_final"]:
                block_avg[k] += save_dat[k] / nsteps
            else:
                block_avg[k] += save_dat[k] * save_dat["weight_final"] / nsteps

    for k, it in block_avg.items():
        if k not in ["overlap", "overlap_gradient", "weight_final", "acceptance"]:
            it /= block_avg["weight_final"]
    return block_avg, configs


def sample_overlap(wfs, configs, pgrad, nblocks=10, nsteps=10, tstep=0.5):
    r"""
    Sample

    .. math:: \rho(R) = \sum_i |\Psi_i(R)|^2

    `pgrad` is expected to be a gradient generator. returns data as follows:

    `overlap` :

    .. math:: \left\langle \frac{\Psi_i^* \Psi_j}{\rho} \right\rangle

    `overlap_gradient`:

    .. math:: \left\langle \frac{\partial_{im} \Psi_i^* \Psi_j}{\rho} \right\rangle

    Note that overlap_gradient is only saved for i = f, where f is the final wave function.

    In addition, any key returned by `pgrad` will be saved for the final wave function.
    """

    return_data = {}
    for block in range(nblocks):
        block_avg, configs = sample_overlap_worker(wfs, configs, pgrad, nsteps, tstep)
        # Blocks stored
        for k, it in block_avg.items():
            if k not in return_data:
                return_data[k] = np.zeros((nblocks, *it.shape), dtype=it.dtype)
            return_data[k][block, ...] = it.copy()
    return return_data, configs


def dist_sample_overlap(
    wfs, configs, pgrad, nblocks=10, nsteps=10, client=None, npartitions=None, **kwargs
):
    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    return_data = {}
    for block in range(nblocks):
        coord = configs.split(npartitions)
        runs = []
        for nodeconfigs in coord:
            runs.append(
                client.submit(
                    sample_overlap_worker,
                    wfs,
                    nodeconfigs,
                    pgrad,
                    nsteps=nsteps,
                    **kwargs
                )
            )

        allresults = list(zip(*[r.result() for r in runs]))
        configs.join(allresults[1])
        confweight = np.array([len(c.configs) for c in coord], dtype=float)
        avgweights = np.array([res["weight_final"] for res in allresults[0]])
        avgweights *= confweight
        avgweights /= np.sum(avgweights)
        confweight /= np.sum(confweight)
        block_avg = {}
        for k in allresults[0][0].keys():
            if k not in ["weight", "overlap", "overlap_gradient"]:
                block_avg[k] = np.sum(
                    [res[k] * w for res, w in zip(allresults[0], avgweights)], axis=0
                )
            else:
                block_avg[k] = np.sum(
                    [res[k] * w for res, w in zip(allresults[0], confweight)], axis=0
                )

        # Blocks stored
        for k, it in block_avg.items():
            if k not in return_data:
                return_data[k] = np.zeros((nblocks, *it.shape), dtype=it.dtype)
            return_data[k][block, ...] = it.copy()
    return return_data, configs


def correlated_sample(wfs, configs, parameters, pgrad):
    r"""
    Given a configs sampled from the distribution

    .. math:: \rho = \sum_i \Psi_i^2

    Compute properties for replacing the last wave function with each of parameters

    For energy

    .. math:: \langle E \rangle = \left\langle \frac{H\Psi}{\Psi} \frac{|\Psi|^2}{\rho}  \right\rangle

    The ratio can be computed as

    .. math:: \frac{|\Psi|^2}{\rho} = \frac{1}{\sum_i e^{2(\alpha_i - \alpha}) }

    Where we write

    .. math:: \Psi = e^{i\theta}{e^\alpha}

    We also compute

    .. math:: \langle S_i \rangle = \left\langle \frac{\Psi_i^* \Psi}{\rho} \right\rangle

    And

    .. math:: \langle N_i \rangle = \left\langle \frac{|\Psi_i|^2}{\rho} \right\rangle

    """
    nparms = len(parameters)
    p0 = pgrad.transform.serialize_parameters(wfs[-1].parameters)
    wfvalues = [wf.recompute(configs) for wf in wfs]
    phase0, log_values0 = [np.nan_to_num(np.array(x)) for x in zip(*wfvalues)]
    log_values0 = np.real(log_values0)
    ref = np.max(log_values0)
    normalized_values = phase0 * np.exp(log_values0 - ref)
    denominator = np.sum(np.exp(2 * (log_values0 - ref)), axis=0)
    rhoprime_ = np.sum(np.exp(2 * (log_values0[:-1] - ref)), axis=0)

    wt0 = 1.0 / np.sum(np.exp(-2 * (log_values0[:, np.newaxis] - log_values0)), axis=1)
    weight = np.mean(wt0, axis=1)
    dtype = complex if wfs[-1].iscomplex else float

    data = {
        "total": np.zeros(nparms),
        "weight": np.zeros(nparms),
        "overlap": np.zeros((nparms, len(wfs)), dtype=dtype),
        "rhoprime": np.zeros(nparms),
    }
    data["base_weight"] = weight
    for p, parameter in enumerate(parameters):
        wf = wfs[-1]
        for k, it in pgrad.transform.deserialize(wf, parameter).items():
            wf.parameters[k] = it
        wf.recompute(configs)
        val = wf.value()
        dat = pgrad.enacc(configs, wf)

        wt = wt0[-1] * np.exp(2 * (val[1] - log_values0[-1]))
        normalized_val = val[0] * np.exp(val[1] - ref)
        overlap = normalized_val * normalized_values.conj() / denominator
        # This is the new rho with the test wave function
        rhoprime = np.mean((rhoprime_ + np.exp(2 * (val[1] - ref))) / denominator)

        data["total"][p] = np.real(np.average(dat["total"], weights=wt))
        data["rhoprime"][p] = rhoprime
        data["weight"][p] = np.mean(wt) / rhoprime
        data["overlap"][p] = np.mean(overlap, axis=1) / np.sqrt(np.mean(wt) * weight)

    for k, it in pgrad.transform.deserialize(wf, p0).items():
        wfs[-1].parameters[k] = it
    return data


def dist_correlated_sample(wfs, configs, *args, client, npartitions=None, **kwargs):

    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    coord = configs.split(npartitions)
    allruns = []
    for nodeconfigs in coord:
        allruns.append(
            client.submit(correlated_sample, wfs, nodeconfigs, *args, **kwargs)
        )

    allresults = [r.result() for r in allruns]
    df = {}
    for k in allresults[0].keys():
        df[k] = np.array([x[k] for x in allresults])
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    confweight /= confweight.mean()
    rhowt = np.einsum("i...,i->i...", df["rhoprime"], confweight)
    wt = df["weight"] * rhowt
    df["total"] = np.average(df["total"], weights=wt, axis=0)
    df["overlap"] = np.average(df["overlap"], weights=confweight, axis=0)
    df["weight"] = np.average(df["weight"], weights=rhowt, axis=0)

    # df["weight"] = np.mean(df["weight"], axis=0)
    df["rhoprime"] = np.mean(rhowt, axis=0)
    return df


def renormalize(wfs, N):
    """
    Normalizes the last wave function, given a current value of the normalization. Assumes that we want N to be 0.5

    .. math::

        b^2/(a^2 + b^2) = N

        b^2 = N a^2 /(1-N)

        f^2 b^2 = 0.5 a^2/0.5 = a^2

        f^2 = a^2/b^2 = (1-N)/N
    """
    renorm = np.sqrt((1 - N) / N)

    if "wf1det_coeff" in wfs[-1].parameters.keys():
        wfs[-1].parameters["wf1det_coeff"] *= renorm
    else:
        raise NotImplementedError("need wf1det_coeff in parameters")


def evaluate(return_data, warmup):
    """
    For wave functions wfs and coordinate set coords, evaluate the overlap and energy of the last wave function.

    Returns a dictionary with relevant information.
    """
    avg_data = {}
    for k, it in return_data.items():
        avg_data[k] = np.average(it[warmup:], axis=0)
    N = np.abs(avg_data["overlap"].diagonal())
    # Derivatives are only for the optimized wave function, so they miss
    # an index
    N_derivative = 2 * np.real(avg_data["overlap_gradient"][-1])
    Nij = np.sqrt(np.outer(N, N))
    S = avg_data["overlap"] / Nij
    S_derivative = avg_data["overlap_gradient"] / Nij[-1, :, np.newaxis] - np.einsum(
        "j,m->jm", 0.5 * avg_data["overlap"][-1, :] / Nij[-1, :], N_derivative / N[-1]
    )
    energy_derivative = 2.0 * np.real(
        avg_data["dpH"] - avg_data["total"] * avg_data["dppsi"]
    )
    dp = avg_data["dppsi"]
    condition = np.real(avg_data["dpidpj"] - np.einsum("i,j->ij", dp, dp))

    return {
        "N": N,
        "S": S,
        "S_derivative": S_derivative,
        "energy_derivative": energy_derivative,
        "N_derivative": N_derivative,
        "condition": condition,
        "total": np.real(avg_data["total"]),
    }


def optimize_orthogonal(
    wfs,
    coords,
    pgrad,
    Starget=None,
    forcing=None,
    tstep=0.1,
    max_iterations=30,
    warmup=1,
    warmup_options=None,
    Ntarget=0.5,
    max_step=10.0,
    hdf_file=None,
    linemin=True,
    step_offset=0,
    Ntol=0.05,
    weight_boundaries=0.3,
    sample_options=None,
    correlated_options=None,
    client=None,
    npartitions=None,
    verbose=True,
):
    r"""
    Minimize

    .. math:: f(p_f) = E_f + \sum_i \lambda_{i=0}^{f-1} |S_{fi} - S_{fi}^*|^2

    Where

    .. math:: N_i = \langle \Psi_i | \Psi_i \rangle

    .. math:: S_{fi} = \frac{\langle \Psi_f | \Psi_i \rangle}{\sqrt{N_f N_i}}

    The \*'d and lambda values are respectively targets and forcings. f is the final wave function in the wave function array.
    We only optimize the parameters of the final wave function, so all 'p' values here represent a parameter in the final wave function.

    **Important arguments**

        :wfs: a list of wave function objects. The last one is optimized; the rest are kept fixed and used as orthogonalization references

        :coords: A Coord set

        :pgrad: A Pgradient object

        :tstep: Maximum timestep for line minimization, or timestep when line minimization is off

        :max_iterations: Number of optimization steps to take

        :warmup_options: a dictionary of options for warm up vmc

        :Starget: An array-like of length len(wfs)-1, which indicates the target overlap for each reference wave function.

        :forcing: An array-like of length len(wfs)-1, which gives the penalty (lambda) for each reference wave function

        :hdf_file: A string that gives the filename to save the optimization data in.

    **Arguments for experts**

        Other arguments should not be changed unless you know what you're doing.

    **Details about the implementation**

    The derivatives are:

    .. math:: \partial_p N_f = 2 Re \langle \partial_p \Psi_f | \Psi_f \rangle

    .. math::  \langle \partial_p \Psi_f | \Psi_i \rangle = \int{ \frac{ \Psi_i\partial_p \Psi_f^*}{\rho} \frac{\rho}{\int \rho} }

    .. math:: \partial_p S_{fi} = \frac{\langle \partial_p \Psi_f | \Psi_i \rangle}{\sqrt{N_f N_i}} - \frac{\langle \Psi_f | \Psi_i \rangle}{2\sqrt{N_f N_i}} \frac{\partial_p N_f}{N_f}

    In this implementation, we set

    .. math:: \rho = \sum_i |\Psi_i|^2

    Note that in the definition of N there is an arbitrary normalization of rho. The targets are set relative to the normalization of the reference wave functions.

    Some implementation notes regarding the normalization:

    It's important for the normalization of the wave functions to be similar; otherwise the weights of one dominate and only one of the wave functions gets sampled.
    Some notes:

     * One could modify the relative weights in the definition of rho, but it's more convenient to output wave functions that are normalized with respect to each other.
     * Adding a penalty to the normalization turns out to be fairly unstable.
     * Moves to reduce the overlap and energy tend to change the normalization a lot (keep in mind that both the determinant and Jastrow parts can have gauge degrees of freedom). This can lead into a tailspin effect pretty quickly.

    In this implementation, we handle this using three techniques:

     * Most importantly, the cost function derivative is orthogonalized to the derivative of the normalization.
     * In the line minimization, if the normalization deviates too far from 0.5 relative to the reference wave function, we do not consider the move. The correlated sampling is unreliable in that situation anyway.
     * The wave function is renormalized if its normalization deviates too far from 0.5 relative to the first wave function.
    """

    # Restart
    if hdf_file is not None and os.path.isfile(hdf_file):
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                grp = hdf["wf"]
                for k in grp.keys():
                    wfs[-1].parameters[k] = np.array(grp[k])
            if "iteration" in hdf.keys():
                step_offset = np.max(hdf["iteration"][...]) + 1

    parameters = pgrad.transform.serialize_parameters(wfs[-1].parameters)

    if Starget is None:
        Starget = np.zeros(len(wfs) - 1)
    if forcing is None:
        forcing = np.ones(len(wfs) - 1)
    Starget = np.asarray(Starget)
    forcing = np.asarray(forcing)
    if len(forcing) != len(wfs) - 1:
        raise AttributeError(
            "forcing should be an array of length equal to the wfs minus 1: "
            + str(len(forcing))
        )
    attr = dict(
        tstep=tstep,
        max_iterations=max_iterations,
        forcing=forcing,
        warmup=warmup,
        Starget=Starget,
        Ntarget=Ntarget,
        max_step=max_step,
    )
    conditioner = pyqmc.linemin.sd_update
    if npartitions is None:
        npartitions = sum(client.nthreads().values())


    if sample_options is None:
        sample_options = {}
    if correlated_options is None:
        correlated_options = {}

    if client is None:
        sampler = sample_overlap
        correlated_sampler = correlated_sample
    else:
        sampler = dist_sample_overlap
        correlated_sampler = dist_correlated_sample
        sample_options["client"] = client
        sample_options["npartitions"] = npartitions
        correlated_options["client"] = client
        correlated_options["npartitions"] = npartitions

    # warm up to equilibrate the configurations before running optimization
    if warmup_options is None:
        warmup_options = dict(nblocks=1, nsteps=10)
    if "tstep" not in warmup_options and "tstep" in sample_options:
        warmup_options["tstep"] = sample_options["tstep"]

    data, coords = mc.vmc(
        wfs[-1],
        coords,
        accumulators={},
        client=client,
        npartitions=npartitions,
        **warmup_options
    )

    # One set of configurations for every wave function
    allcoords = [coords.copy() for _ in wfs[:-1]]
    dtype = np.complex if wfs[-1].iscomplex else np.float

    for step in range(max_iterations):
        # we iterate until the normalization is reasonable
        # One could potentially save a little time here by not computing the gradients
        # every time, but typically we don't have to renormalize if the moves are good

        # Memory efficient implementation
        nwf = len(wfs)
        normalization = np.zeros(nwf - 1)
        total_energy = 0
        # energy_derivative = np.zeros(len(parameters))
        N_derivative = np.zeros(len(parameters))
        condition = np.zeros((len(parameters), len(parameters)))
        overlaps = np.zeros(nwf - 1, dtype=dtype)
        overlap_derivatives = np.zeros((nwf - 1, len(parameters)), dtype=dtype)

        while True:
            return_data, _ = sampler(
                [wfs[0], wfs[-1]], allcoords[0], pgrad, **sample_options
            )
            tmp_deriv = evaluate(return_data, warmup)
            N = tmp_deriv["N"][-1]

            if verbose:
                print("Normalization", N, flush=True)
            if abs(N - Ntarget) < Ntol:
                normalization[0] = tmp_deriv["N"][-1]
                total_energy += tmp_deriv["total"] / (nwf - 1)
                energy_derivative = tmp_deriv["energy_derivative"] / (nwf - 1)
                N_derivative += tmp_deriv["N_derivative"] / (nwf - 1)
                condition += tmp_deriv["condition"] / (nwf - 1)
                overlaps[0] = tmp_deriv["S"][-1, 0]
                overlap_derivatives[0] = tmp_deriv["S_derivative"][0, :]
                break
            else:
                renormalize([wfs[0], wfs[-1]], N)
                parameters = pgrad.transform.serialize_parameters(wfs[-1].parameters)

        for i, wf in enumerate(wfs[1:-1]):
            return_data, _ = sampler(
                [wf, wfs[-1]], allcoords[i + 1], pgrad, **sample_options
            )
            deriv_data = evaluate(return_data, warmup)
            normalization[i + 1] = deriv_data["N"][-1]
            total_energy += deriv_data["total"] / (nwf - 1)
            energy_derivative += deriv_data["energy_derivative"] / (nwf - 1)
            N_derivative += deriv_data["N_derivative"] / (nwf - 1)
            condition += deriv_data["condition"] / (nwf - 1)
            overlaps[i + 1] = deriv_data["S"][-1, 0]
            overlap_derivatives[i + 1] = deriv_data["S_derivative"][0, :]
        if verbose:
            print("normalization", normalization)

        delta = overlaps - Starget
        delta_phase = delta / np.abs(delta)
        overlap_derivative = np.einsum(
            "j,jk->k",
            2.0 * forcing * np.abs(delta),
            np.real(overlap_derivatives / delta_phase[:, np.newaxis]),
        )

        total_derivative = energy_derivative + overlap_derivative

        if verbose:
            print("############################# iteration ", step)
            format_str = "{:<15}" * 2 + "{:<20.3}" * 2
            print(format_str.format("Quantity", "wf", "val", "|g|"))
            print(
                format_str.format(
                    "energy",
                    len(wfs) - 1,
                    total_energy,
                    np.linalg.norm(energy_derivative),
                )
            )
            print(
                format_str.format("norm", len(wfs) - 1, N, np.linalg.norm(N_derivative))
            )
            for i in range(len(wfs) - 1):
                print(
                    format_str.format(
                        "overlap",
                        i,
                        overlaps[i],
                        np.linalg.norm(overlap_derivatives[i]),
                    ),
                    flush=True,
                )

        # Use SR to condition the derivatives
        total_derivative, N_derivative = np.einsum(
            "ij,dj->di",
            np.linalg.inv(condition + 0.1 * np.eye(condition.shape[0])),
            [total_derivative, N_derivative],
        )

        # Try to move in the projection that doesn't change the norm
        # Here we project out the norm derivative
        if np.linalg.norm(N_derivative) > 1e-8:
            total_derivative -= (
                np.dot(total_derivative, N_derivative)
                * N_derivative
                / (np.linalg.norm(N_derivative)) ** 2
            )

        deriv_norm = np.linalg.norm(total_derivative)
        if deriv_norm > max_step:
            total_derivative = total_derivative * max_step / deriv_norm

        test_tsteps = np.linspace(-0.1 * tstep, tstep, 11)
        test_parameters = [
            parameters + conditioner(total_derivative, condition, x)
            for x in test_tsteps
        ]
        data = []
        for icoord, wf in zip(allcoords, wfs):
            data.append(
                correlated_sampler(
                    [wf, wfs[-1]], icoord, test_parameters, pgrad, **correlated_options
                )
            )
        line_data = {}
        for k in data[0].keys():
            line_data[k] = np.asarray([x[k] for x in data])

        yfit = []
        xfit = []
        overlap_cost = (
            forcing[:, np.newaxis]
            * np.abs(line_data["overlap"][:, :, 0] - Starget[:, np.newaxis]) ** 2
        )
        cost = np.mean(line_data["total"], axis=0) + np.sum(overlap_cost, axis=0)
        mask = (np.abs(line_data["weight"] - 1.0) > weight_boundaries) & (
            np.abs(line_data["weight"]) > weight_boundaries
        )
        mask = np.all(mask, axis=0)
        if verbose:
            print("tsteps", test_tsteps)
            print("cost", cost)
            print("overlap cost", overlap_cost)
            print("mask", mask)
        xfit = test_tsteps[mask]
        yfit = cost[mask]

        if verbose:
            print("|total_derivative|", np.linalg.norm(total_derivative))
        if len(xfit) > 2:
            min_tstep = pyqmc.linemin.stable_fit(xfit, yfit)
            if verbose:
                print("chose to move", min_tstep, flush=True)
            parameters = parameters + conditioner(
                total_derivative, condition, min_tstep
            )
        else:
            print("WARNING: did not find valid moves. Reducing the timestep")
            tstep *= 0.5

        for k, it in pgrad.transform.deserialize(wfs[-1], parameters).items():
            wfs[-1].parameters[k] = it

        save_data = {
            "energy": total_energy,
            "overlap": overlaps,
            "gradient": total_derivative,
            "N": N,
            "parameters": parameters,
            "iteration": step + step_offset,
            "normalization": normalization,
            "overlap_derivatives": overlap_derivatives,
            "energy_derivative": energy_derivative,
            "line_tsteps": test_tsteps,
            "line_cost": cost,
            "line_norm": line_data["weight"],
        }

        ortho_hdf(hdf_file, save_data, attr, coords, wfs[-1].parameters)

    return wfs
