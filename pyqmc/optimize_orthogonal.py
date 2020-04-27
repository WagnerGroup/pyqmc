import numpy as np
import pandas as pd
import scipy
import h5py
import pyqmc
import pyqmc.hdftools as hdftools


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


from pyqmc.mc import limdrift


def sample_overlap(wfs, configs, pgrad, nblocks=100, nsteps_per_block=1, nsteps=None, tstep=0.5):
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
    nconf, nelec, ndim = configs.configs.shape
    
    if nsteps is not None:
        nblocks = nsteps
        nsteps_per_block = 1

    for wf in wfs:
        wf.recompute(configs)

    return_data = {}
    for block in range(nblocks):
        block_avg = {}
        for step in range(nsteps_per_block):
            print("block", block, " step", step)
            for e in range(nelec):
                # Propose move
                grads = [np.real(wf.gradient(e, configs.electron(e)).T) for wf in wfs]

                grad = limdrift(np.mean(grads, axis=0))
                gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
                newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
                newcoorde = configs.make_irreducible(e, newcoorde)

                # Compute reverse move
                grads = [np.real(wf.gradient(e, newcoorde).T) for wf in wfs]
                new_grad = limdrift(np.mean(grads, axis=0))
                forward = np.sum(gauss ** 2, axis=1)
                backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

                # Acceptance
                t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
                wf_ratios = np.array([wf.testvalue(e, newcoorde) ** 2 for wf in wfs])
                log_values = np.array([wf.value()[1] for wf in wfs])
                ref = log_values[0]
                weights = np.exp(2 * (log_values - ref))

                ratio = (
                    t_prob * np.sum(wf_ratios * weights, axis=0) / np.sum(weights, axis=0)
                )
                accept = ratio > np.random.rand(nconf)

                # Update wave function
                configs.move(e, newcoorde, accept)
                for wf in wfs:
                    wf.updateinternals(e, newcoorde, mask=accept)
                # print("accept", np.mean(accept))

            log_values = np.array([wf.value() for wf in wfs])
            # print(log_values.shape)
            ref = np.max(log_values[:, 1, :], axis=0)
            save_dat = {}
            denominator = np.sum(np.exp(2 * (log_values[:, 1, :] - ref)), axis=0)
            normalized_values = log_values[:, 0, :] * np.exp(log_values[:, 1, :] - ref)
            save_dat["overlap"] = np.mean(
                np.einsum("ik,jk->ijk", normalized_values, normalized_values) / denominator,
                axis=-1,
            )
            weight = np.array(
                [
                    np.exp(-2 * (log_values[i, 1, :] - log_values[:, 1, :]))
                    for i in range(len(wfs))
                ]
            )
            weight = 1.0 / np.sum(weight, axis=1)
            
            #Fast evaluation of dppsi_reg
            dppsi = pgrad.transform.serialize_gradients(wfs[-1].pgradient())
            node_cut, f = pgrad._node_regr(configs, wfs[-1])
            dppsi_regularized = dppsi * f[:, np.newaxis]

            save_dat["overlap_gradient"] = np.mean(
                np.einsum(
                    "km,k,jk->jmk", dppsi_regularized, normalized_values[-1], normalized_values
                )
                / denominator,
                axis=-1,
            )

            #Weighted average on rest
            dat = pgrad.avg(configs, wf, weight[-1])
            for k in dat.keys():
                save_dat[k] = dat[k]
            save_dat["weight"] = np.mean(weight, axis=1)

            #Rolling average within block
            for k, it in save_dat.items():
                if k not in block_avg:
                    block_avg[k] = np.zeros((*it.shape, ))
                block_avg[k] += save_dat[k] / nsteps_per_block

        #Blocks stored
        for k, it in block_avg.items():
            if k not in return_data:
                return_data[k] = np.zeros((nblocks, *it.shape))
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

    .. math:: \frac{|\Psi|^2}{\rho} = \frac{1}{\sum_i e^{\alpha_i - \alpha} 

    Where we write 

    .. math:: \Psi = e^{i\theta}{e^\alpha} 

    We also compute 

    .. math:: \langle S_i \rangle = \left\langle \frac{\Psi_i^* \Psi}{\rho} \right\rangle

    And 

    .. math:: \langle N_i \rangle = \left\langle \frac{|\Psi_i|^2}{\rho} \right\rangle

    """
    p0 = pgrad.transform.serialize_parameters(wfs[-1].parameters)
    for wf in wfs:
        wf.recompute(configs)
    log_values0 = np.array([wf.value() for wf in wfs])
    nparms = len(parameters)
    ref = np.max(log_values0[:, 1, :])
    normalized_values = log_values0[:, 0, :] * np.exp(log_values0[:, 1, :] - ref)
    denominator = np.sum(np.exp(2 * (log_values0[:, 1, :] - ref)), axis=0)

    weight = np.array(
        [
            np.exp(-2 * (log_values0[i, 1, :] - log_values0[:, 1, :]))
            for i in range(len(wfs))
        ]
    )
    weight = np.mean(1.0 / np.sum(weight, axis=1), axis=1)

    data = {
        "total": np.zeros(nparms),
        "weight": np.zeros(nparms),
        "overlap": np.zeros((nparms, len(wfs))),
        "rhoprime": np.zeros(nparms),
    }
    data["base_weight"] = weight
    for p, parameter in enumerate(parameters):
        wf = wfs[-1]
        for k, it in pgrad.transform.deserialize(parameter).items():
            wf.parameters[k] = it
        wf.recompute(configs)
        val = wf.value()
        dat = pgrad.enacc(configs, wf)

        wt = 1.0 / np.sum(
            np.exp(2 * log_values0[:, 1, :] - 2 * val[1][np.newaxis, :]), axis=0
        )
        normalized_val = val[0] * np.exp(val[1] - ref)
        # This is the new rho with the test wave function
        rhoprime = (
            np.sum(np.exp(2 * log_values0[0:-1, 1, :] - 2 * ref), axis=0)
            + np.exp(2 * val[1] - 2 * ref)
        ) / denominator

        overlap = np.einsum("k,jk->jk", normalized_val, normalized_values) / denominator

        data["total"][p] = np.average(dat["total"], weights=wt)
        data["rhoprime"][p] = np.mean(rhoprime)
        data["weight"][p] = np.mean(wt) / np.mean(rhoprime)
        data["overlap"][p] = np.mean(overlap, axis=1) / np.sqrt(np.mean(wt) * weight)

        # print('wt', wt)
    # print('average energy',data['total'])
    for k, it in pgrad.transform.deserialize(p0).items():
        wfs[-1].parameters[k] = it
    return data


def renormalize(wfs, N):
    """
    Normalize the last wave function, given a current value of the normalization. Assumes that we want N to be 0.5

    .. math:: 
    
        b^2/(a^2 + b^2) = N

        b^2 = N a^2 /(1-N)

        f^2 b^2 = 0.5 a^2/0.5 = a^2 

        f^2 = a^2/b^2 = (1-N)/N
    """
    desired_n = 1.0 / len(wfs)
    current_n = N
    wfs[-1].parameters["wf1det_coeff"] *= np.sqrt((1 - N) / N)


def evaluate(wfs, coords, pgrad, sampler, sample_options, warmup):
    return_data, coords = sampler(wfs, coords, pgrad, **sample_options)
    """ 
    For wave functions wfs and coordinate set coords, evaluate the overlap and energy of the last wave function. 

    Returns a dictionary with relevant information.
    """
    avg_data = {}
    for k, it in return_data.items():
        avg_data[k] = np.average(it[warmup:, ...], axis=0)
    N = avg_data["overlap"].diagonal()
    # Derivatives are only for the optimized wave function, so they miss
    # an index
    N_derivative = 2 * np.real(avg_data["overlap_gradient"][-1])
    Nij = np.outer(N, N)
    S = avg_data["overlap"] / np.sqrt(Nij)
    S_derivative = avg_data["overlap_gradient"] / Nij[-1, :, np.newaxis] - np.einsum(
        "j,m->jm", avg_data["overlap"][-1, :] / Nij[-1, :], N_derivative / N[-1]
    )
    energy_derivative = 2.0 * (avg_data["dpH"] - avg_data["total"] * avg_data["dppsi"])
    dp = avg_data["dppsi"]
    condition = np.real(avg_data["dpidpj"] - np.einsum("i,j->ij", dp, dp))

    return {
        "N": N,
        "S": S,
        "S_derivative": S_derivative,
        "energy_derivative": energy_derivative,
        "N_derivative": N_derivative,
        "condition": condition,
        "total": avg_data["total"],
    }


def optimize_orthogonal(
    wfs,
    coords,
    pgrad,
    Starget=0.0,
    forcing=10.0,
    tstep=0.1,
    nsteps=30,
    warmup=5,
    Ntarget=0.5,
    max_step=10.0,
    hdf_file=None,
    update_method="linemin",
    linemin=True,
    beta1=0.9,
    beta2=0.999,
    adam_epsilon=1e-8,
    step_offset=0,
    Ntol=0.05,
    weight_boundaries=0.3,
    sampler=sample_overlap,
    sample_options=None,
    correlated_sampler=correlated_sample,
    correlated_options=None,
):
    r"""
    Minimize 

    .. math:: f(p_f) = E_f + \sum_i \lambda_{i=0}^{f-1} (S_{fi} - S_{fi}^*)^2 

    Where 

    .. math:: N_i = \langle \Psi_i | \Psi_j \rangle

    .. math:: S_{fi} = \frac{\langle \Psi_f | \Psi_j \rangle}{\sqrt{N_f N_i}}

    The *'d and lambda values are respectively targets and forcings. f is the final wave function in the wave function array.
    We only optimize the parameters of the final wave function, so all 'p' values here represent a parameter in the final wave function. 

    **Important arguments**

        :wfs: a list of wave function objects. The last one is optimized; the rest are kept fixed and used as orthogonalization references

        :coords: A Coord set

        :pgrad: A Pgradient object

        :tstep: Maximum timestep for line minimization, or timestep when line minimization is off

        :nsteps: Number of optimization steps to take

        :Starget: An array-like of length len(wfs)-1, which indicates the target overlap for each reference wave function.

        :forcing: An array-like of length len(wfs)-1, which gives the penalty (lambda) for each reference wave function

        :hdf_file: A string that gives the filename to save the optimization data in.

    **Arguments for experts**

        Other arguments should not be changed unless you know what you're doing.

    **Details about the implementation**

    The derivatives are:

    .. math:: \partial_p N_f = 2 Re \langle \partial_p \Psi_f | \Psi_f \rangle

    .. math::  \langle \partial_p \Psi_f | \Psi_f \rangle = \int{ \frac{ \Psi_f\partial_p \Psi_f^*}{\rho} \frac{\rho}{\int \rho} } 

    .. math:: \partial_p S_{fi} = \frac{\langle \partial_p \Psi_f | \Psi_j \rangle}{\sqrt{N_f N_i}} - \frac{\langle \Psi_f | \Psi_j \rangle}{\sqrt{N_f N_i}} \frac{\partial_p N_f}{N_f} 

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

    parameters = pgrad.transform.serialize_parameters(wfs[-1].parameters)
    adam_m = np.zeros(parameters.shape)
    adam_v = np.zeros(parameters.shape)
    Starget = np.asarray(Starget)
    forcing = np.asarray(forcing)
    attr = dict(
        tstep=tstep,
        nsteps=nsteps,
        forcing=forcing,
        warmup=warmup,
        Starget=Starget,
        Ntarget=Ntarget,
        max_step=max_step,
        update_method=update_method,
        beta1=beta1,
        beta2=beta2,
        adam_epsilon=adam_epsilon,
    )
    conditioner = pyqmc.linemin.sd_update
    if sample_options is None:
        sample_options = {}
    if correlated_options is None:
        correlated_options = {}

    # One set of configurations for every wave function
    allcoords = [coords.copy() for _ in wfs[:-1]]
    for step in range(nsteps):
        # we iterate until the normalization is reasonable
        # One could potentially save a little time here by not computing the gradients
        # every time, but typically we don't have to renormalize if the moves are good
        deriv_data = []
        while True:
            tmp_deriv = evaluate(
                [wfs[0], wfs[-1]], allcoords[0], pgrad, sampler, sample_options, warmup
            )
            N = tmp_deriv["N"][-1]

            print("Normalization", N)
            if abs(N - Ntarget) < Ntol:
                deriv_data.append(tmp_deriv)
                break
            else:
                renormalize([wfs[0], wfs[-1]], N)
                parameters = pgrad.transform.serialize_parameters(wfs[-1].parameters)

        for i, wf in enumerate(wfs[1:-1]):
            deriv_data.append(
                evaluate(
                    [wf, wfs[-1]],
                    allcoords[i + 1],
                    pgrad,
                    sampler,
                    sample_options,
                    warmup,
                )
            )
        collected_data = {}
        for k in deriv_data[0].keys():
            collected_data[k] = np.array([x[k] for x in deriv_data])
        print("normalization", collected_data["N"][:, -1])
        normalization = collected_data["N"][:, -1]
        total_energy = np.mean(collected_data["total"], axis=0)
        energy_derivative = np.mean(collected_data["energy_derivative"], axis=0)
        N_derivative = np.mean(collected_data["N_derivative"], axis=0)
        condition = np.mean(collected_data["condition"], axis=0)
        overlaps = collected_data["S"][:, -1, 0]
        overlap_derivatives = collected_data["S_derivative"][:, 0, :]

        overlap_derivative = np.sum(
            2.0 * (forcing * (overlaps - Starget))[:, np.newaxis] * overlap_derivatives,
            axis=0,
        )

        total_derivative = energy_derivative + overlap_derivative

        print("############################# step ", step)
        format_str = "{:<15}" * 2 + "{:<15.10}" * 2
        print(format_str.format("Quantity", "wf", "value", "|grad|"))
        print(
            format_str.format(
                "energy", len(wfs) - 1, total_energy, np.linalg.norm(energy_derivative)
            )
        )
        print(format_str.format("norm", len(wfs) - 1, N, np.linalg.norm(N_derivative)))
        for i in range(len(wfs) - 1):
            print(
                format_str.format(
                    "overlap", i, overlaps[i], np.linalg.norm(overlap_derivatives[i])
                )
            )

        # Use SR to condition the derivatives
        invSij = np.linalg.inv(condition + 0.1 * np.eye(condition.shape[0]))
        total_derivative = np.einsum("ij,j->i", invSij, total_derivative)
        N_derivative = np.einsum("ij,j->i", invSij, N_derivative)

        # Try to move in the projection that doesn't change the norm
        # Here we project out the
        if np.linalg.norm(N_derivative) > 1e-8:
            total_derivative -= (
                np.dot(total_derivative, N_derivative)
                * N_derivative
                / (np.linalg.norm(N_derivative)) ** 2
            )

        deriv_norm = np.linalg.norm(total_derivative)
        if deriv_norm > max_step:
            total_derivative = total_derivative * max_step / deriv_norm

        # ADAM uses a momentum term, which can sometimes accelerate the
        # optimization and allow for a much smaller sample size
        if update_method == "adam":
            adam_m = beta1 * adam_m + (1 - beta1) * total_derivative
            adam_v = beta2 * adam_v + (1 - beta2) * total_derivative ** 2
            adam_mhat = adam_m / (1 - beta1 ** (step + 1))
            adam_vhat = adam_v / (1 - beta2 ** (step + 1))
            total_derivative = adam_mhat / (np.sqrt(adam_vhat) + adam_epsilon)

        # print("derivative after modifications", total_derivative.round(2))
        if linemin:
            test_tsteps = np.linspace(-tstep, tstep, 21)
            test_parameters = [
                parameters + conditioner(total_derivative, condition, x)
                for x in test_tsteps
            ]
            data = []
            for icoord, wf in zip(allcoords, wfs):
                data.append(
                    correlated_sampler(
                        [wf, wfs[-1]],
                        icoord,
                        test_parameters,
                        pgrad,
                        **correlated_options
                    )
                )
            line_data = {}
            for k in data[0].keys():
                line_data[k] = np.asarray([x[k] for x in data])

            yfit = []
            xfit = []
            overlap_cost = (
                forcing[:, np.newaxis]
                * (line_data["overlap"][:, :, 0] - Starget[:, np.newaxis]) ** 2
            )
            cost = np.mean(line_data["total"], axis=0) + np.sum(overlap_cost, axis=0)
            mask = (np.abs(line_data["weight"] - 1.0) > weight_boundaries) & (
                np.abs(line_data["weight"]) > weight_boundaries
            )
            mask = np.all(mask, axis=0)
            xfit = test_tsteps[mask]
            yfit = cost[mask]

            row_format = "{:<5}" + "{:<12.6} " * 5 + "{:<10}"
            print(
                row_format.format(
                    "wf", "tstep", "energy", "S", "weight", "cost", "used"
                )
            )
            for pt in range(len(mask)):
                for wf in range(len(data)):
                    print(
                        row_format.format(
                            wf,
                            test_tsteps[pt],
                            line_data["total"][wf, pt],
                            line_data["overlap"][wf, pt, 0],
                            line_data["weight"][wf, pt],
                            cost[pt],
                            mask[pt],
                        )
                    )

            if len(xfit) > 0:
                min_tstep = pyqmc.linemin.stable_fit2(xfit, yfit)
                print("chose to move", min_tstep)
                parameters += conditioner(total_derivative, condition, min_tstep)
        else:
            parameters += conditioner(total_derivative, condition, tstep)

        for k, it in pgrad.transform.deserialize(parameters).items():
            wfs[-1].parameters[k] = it

        normalization = collected_data["N"][:, -1]
        total_energy = np.mean(collected_data["total"], axis=0)
        energy_derivative = np.mean(collected_data["energy_derivative"], axis=0)
        N_derivative = np.mean(collected_data["N_derivative"], axis=0)
        condition = np.mean(collected_data["condition"], axis=0)
        overlaps = collected_data["S"][:, -1, 0]
        overlap_derivatives = collected_data["S_derivative"][:, 0, :]

        save_data = {
            "energies": total_energy,
            "overlap": overlaps,
            "gradient": total_derivative,
            "N": N,
            "parameters": parameters,
            "step": step + step_offset,
            "normalization": normalization,
            "overlap_derivatives": overlap_derivatives,
            "energy_derivative": energy_derivative,
        }
        if linemin:
            save_data["line_tsteps"] = test_tsteps
            save_data["line_cost"] = cost
            save_data["line_norm"] = line_data["weight"]

        ortho_hdf(
            hdf_file, save_data, attr, coords, pgrad.transform.deserialize(parameters)
        )
        for wf in wfs:
            print(wf.parameters["wf1det_coeff"])
    return wfs
