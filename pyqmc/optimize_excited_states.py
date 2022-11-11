import numpy as np
from scipy.stats.stats import WeightedTauResult
import pyqmc.mc as mc
import scipy.stats
import pyqmc.linemin as linemin

"""
TODO:

Optimizer test
"""


def collect_overlap_data(wfs, configs, energy, transforms):
    r"""Collect the averages over configs assuming that
    configs are distributed according to

    .. math:: \rho \propto \sum_i |\Psi_i|^2

    The keys 'overlap' and 'overlap_gradient' are

    `overlap` :

    .. math:: \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    `overlap_gradient`:

    .. math:: \partial_m \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\partial_{fm} \Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    The function returns two dictionaries:

    weighted_dat: each element is a list (one item per wf) of quantities that are accumulated as O psi_i psi_j /rho
    unweighted_dat: Each element is a numpy array that are accumulated just as O (no weight).
                    This in particular includes 'weight' which is just psi_i psi_j/rho

    """
    phase, log_vals = [
        np.nan_to_num(np.array(x)) for x in zip(*[wf.value() for wf in wfs])
    ]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    denominator = np.mean(np.nan_to_num(np.exp(2 * (log_vals - ref))), axis=0)
    normalized_values = phase * np.nan_to_num(np.exp(log_vals - ref))

    energies = invert_list_of_dicts([energy(configs, wf) for wf in wfs])
    dppsi = [
        transform.serialize_gradients(wf.pgradient())
        for transform, wf in zip(transforms, wfs)
    ]

    weighted_dat = {}
    unweighted_dat = {}

    weight = np.einsum(
        "ic,jc->ijc", normalized_values.conj(), normalized_values / denominator
    )
    # normalized_values are [config,wf]
    # we average over configs here and produce [wf,wf]
    # c refers to the configuration
    unweighted_dat["overlap"] = np.mean(weight, axis=-1)

    # Weighted average
    for k, en in energies.items():
        weighted_dat[k] = (
            np.einsum("jc,ijc->ij", np.asarray(en), weight) / weight.shape[-1]
        )

    nconfig = weight.shape[-1]
    for wfi, dp in enumerate(dppsi):
        weighted_dat[("dp", wfi)] = np.zeros(
            (dp.shape[1], weight.shape[0], weight.shape[1]), dtype=dp.dtype
        )
        weighted_dat[("dpH", wfi)] = np.zeros(
            (dp.shape[1], weight.shape[0], weight.shape[1]), dtype=dp.dtype
        )
        weighted_dat[("dp2", wfi)] = np.zeros(
            (dp.shape[1], weight.shape[0], weight.shape[1]), dtype=dp.dtype
        )

        weighted_dat[("dp", wfi)][:, wfi, :] = (
            np.einsum("cp,jc->pj", dp, weight[wfi, :, :], optimize=True) / nconfig
        )

        weighted_dat[("dp2", wfi)][:, wfi, :] = (
            np.einsum("cp,jc->pj", dp**2, weight[wfi, :, :], optimize=True) / nconfig
        )

        weighted_dat[("dpH", wfi)][:, wfi, :] = (
            np.einsum("jc,cp,jc->pj", energies["total"], dp, weight[wfi, :, :])
            / nconfig
        )

        ## make it symmetric
        for k in ["dp", "dp2", "dpH"]:
            weighted_dat[(k, wfi)] += weighted_dat[(k, wfi)].transpose((0, 2, 1))
            weighted_dat[(k, wfi)][:, wfi, wfi] /= 2

    return weighted_dat, unweighted_dat


def invert_list_of_dicts(A):
    """
    if we have a list [ {'A':1,'B':2}, {'A':3, 'B':5}], invert the structure to
    {'A':[1,3], 'B':[2,5]}.
    If not all keys are present in all lists, error.
    """
    final_dict = {}
    for k in A[0].keys():
        final_dict[k] = []
    for a in A:
        for k, v in a.items():
            final_dict[k].append(v)
    return final_dict


def sample_overlap_worker(
    wfs, configs, energy, transforms, nsteps=10, nblocks=10, tstep=0.5
):
    r"""Run nstep Metropolis steps to sample a distribution proportional to
    :math:`\sum_i |\Psi_i|^2`, where :math:`\Psi_i` = wfs[i]
    """
    nconf, nelec, _ = configs.configs.shape
    for wf in wfs:
        wf.recompute(configs)
    weighted = []
    unweighted = []
    for block in range(nblocks):
        weighted_block = {}
        unweighted_block = {}

        for n in range(nsteps):
            for e in range(nelec):  # a sweep
                # Propose move
                grads = [np.real(wf.gradient(e, configs.electron(e)).T) for wf in wfs]
                grad = mc.limdrift(np.mean(grads, axis=0))
                gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
                newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
                newcoorde = configs.make_irreducible(e, newcoorde)

                # Compute reverse move
                grads, vals, saved_values = list(
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

                ratio = (
                    t_prob * np.sum(wf_ratios * weights, axis=0) / weights.sum(axis=0)
                )
                accept = ratio > np.random.rand(nconf)
                # block_avg["acceptance"][n] += accept.mean() / nelec

                # Update wave function
                configs.move(e, newcoorde, accept)
                for wf, saved in zip(wfs, saved_values):
                    wf.updateinternals(
                        e, newcoorde, configs, mask=accept, saved_values=saved
                    )

            # Collect rolling average
            weighted_dat, unweighted_dat = collect_overlap_data(
                wfs, configs, energy, transforms
            )
            for k, it in unweighted_dat.items():
                if k not in unweighted_block:
                    unweighted_block[k] = np.zeros((*it.shape,), dtype=it.dtype)
                unweighted_block[k] += unweighted_dat[k] / nsteps

            for k, it in weighted_dat.items():
                if k not in weighted_block:
                    weighted_block[k] = [
                        np.zeros((*x.shape,), dtype=x.dtype) for x in it
                    ]
                for b, v in zip(weighted_block[k], it):
                    b += v / nsteps
        weighted.append(weighted_block)
        unweighted.append(unweighted_block)

    # here we modify the data so that it's a dictionary of lists of arrays for weighted
    # and a dictionary of arrays for unweighted
    # Access weighted as weighted[quantity][block, ...]
    # Access unweighted as unweighted[quantity][block,...]
    weighted = invert_list_of_dicts(weighted)
    unweighted = invert_list_of_dicts(unweighted)

    for k in weighted.keys():
        weighted[k] = np.asarray(weighted[k])
    for k in unweighted.keys():
        unweighted[k] = np.asarray(unweighted[k])
    return weighted, unweighted, configs


def sample_overlap(
    wfs,
    configs,
    energy,
    transforms,
    nsteps=10,
    nblocks=10,
    tstep=0.5,
    client=None,
    npartitions=0,
):
    """ """
    if client is None:
        return sample_overlap_worker(
            wfs, configs, energy, transforms, nsteps, nblocks, tstep
        )
    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    coord = configs.split(npartitions)
    runs = []
    for nodeconfigs in coord:
        runs.append(
            client.submit(
                sample_overlap_worker,
                wfs,
                nodeconfigs,
                energy,
                transforms,
                nsteps,
                nblocks,
                tstep,
            )
        )
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[2])
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    weighted = {}
    for k, it in invert_list_of_dicts(allresults[0]).items():
        weighted[k] = np.average(
            it, weights=confweight, axis=0
        )  # [np.average(x, weights=confweight, axis=0) for x in inverted_array]
    unweighted = {}
    for k, it in invert_list_of_dicts(allresults[1]).items():
        unweighted[k] = np.average(np.asarray(it), weights=confweight, axis=0)

    return weighted, unweighted, configs


def average(weighted, unweighted):
    """
    (more or less) correctly average the output from sample_overlap
    Returns the average and error as dictionaries.

    TODO: use a more accurate formula for weighted uncertainties
    """
    avg = {}
    error = {}
    for k, it in unweighted.items():
        avg[k] = np.mean(it, axis=0)
        error[k] = scipy.stats.sem(it, axis=0)

    N = np.abs(avg["overlap"].diagonal())
    Nij = np.sqrt(np.outer(N, N))

    for k, it in weighted.items():
        avg[k] = np.mean(it, axis=0) / Nij
        error[k] = scipy.stats.sem(it, axis=0) / Nij
    return avg, error


def collect_terms(avg, error):
    """
    Generate the terms we need to do the optimization.
    """
    ret = {}

    nwf = avg["total"].shape[0]
    N = np.abs(avg["overlap"].diagonal())
    Nij = np.sqrt(np.outer(N, N))

    ret["norm"] = N
    ret["overlap"] = avg["overlap"] / Nij
    fac = np.ones((nwf, nwf)) + np.identity(nwf)
    for wfi in range(nwf):
        ret[("dp_energy", wfi)] = fac * np.real(
            avg[("dpH", wfi)] - avg["total"] * avg[("dp", wfi)]
        )
        ret[("dp_norm", wfi)] = 2.0 * np.real(avg[("dp", wfi)][:, wfi, wfi])
        norm_part = np.zeros(
            (ret[("dp_energy", wfi)].shape[0], nwf, nwf), dtype=avg["overlap"].dtype
        )
        norm_part[:, wfi, :] = (
            np.einsum("i,p->pi", avg["overlap"][wfi, :], ret[("dp_norm", wfi)]) / N[wfi]
        )
        norm_part += norm_part.transpose((0, 2, 1))
        ret[("dp_overlap", wfi)] = fac * (avg[("dp", wfi)] - 0.5 * norm_part) / Nij
        ret[("condition", wfi)] = np.real(
            avg[("dp2", wfi)][:, wfi, wfi] - avg[("dp", wfi)][:, wfi, wfi] ** 2
        )
    ret["energy"] = avg["total"]
    return ret


def objective_function_derivative(
    terms, overlap_penalty, norm_penalty, offdiagonal_energy_penalty
):
    """
    terms are output from generate_terms
    lam is the penalty
    norm_relative_penalty is the prefactor on the norm
    offdiagonal_energy_penalty is the penalty on the off-diagonal matrix elements.
    """
    nwf = terms["energy"].shape[0]

    return [
        terms[("dp_energy", i)][:, i, i]
        + overlap_penalty
        * 2
        * np.sum(np.triu(terms[("dp_overlap", i)] * terms["overlap"], 1), axis=(1, 2))
        + norm_penalty * 2 * (terms["norm"][i] - 1) * terms[("dp_norm", i)]
        + offdiagonal_energy_penalty
        * 2.0
        * np.sum(np.triu(terms["energy"] * terms[("dp_energy", i)], 1), axis=(1, 2))
        for i in range(nwf)
    ]


import pyqmc.hdftools as hdftools
import h5py


def hdf_save(hdf_file, data, attr, wfs):

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "energy" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                for wfi, wf in enumerate(wfs):
                    for k, it in wf.parameters.items():
                        hdf.create_dataset(f"wf/{wfi}/" + k, data=it)

            hdftools.append_hdf(hdf, data)
            for wfi, wf in enumerate(wfs):
                for k, it in wf.parameters.items():
                    hdf[f"wf/{wfi}/" + k][...] = it.copy()


def correlated_sampling(
    wfs, configs, energy, transforms, parameters, client=None, npartitions=0
):
    """
    Run in parallel if client is specified.
    """
    if client is None:
        return correlated_sampling_worker(wfs, configs, energy, transforms, parameters)
    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    coord = configs.split(npartitions)
    runs = []
    for nodeconfigs in coord:
        runs.append(
            client.submit(
                correlated_sampling_worker,
                wfs,
                nodeconfigs,
                energy,
                transforms,
                parameters,
            )
        )
    allresults = [r.result() for r in runs]
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    weighted = {}
    for k, it in invert_list_of_dicts(allresults).items():
        weighted[k] = np.average(it, weights=confweight, axis=0)

    return weighted


def correlated_sampling_worker(wfs, configs, energy, transforms, parameters):
    """
    Input:
       wfs
       configs

    returns:
      data along the path:
         overlap*weight_correlated
         energy*weight_correlated*weight_energy
         weights for correlated sampling: rhoprime /rho
         weights for energy expectation values: psi_i^2/rho
    """

    p0 = [
        transform.serialize_parameters(wf.parameters)
        for wf, transform in zip(wfs, transforms)
    ]
    phase, log_vals = [
        np.nan_to_num(np.array(x)) for x in zip(*[wf.recompute(configs) for wf in wfs])
    ]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    rho = np.mean(np.nan_to_num(np.exp(2 * (log_vals - ref))), axis=0)
    nconfig = configs.configs.shape[0]

    energy_final = []
    overlap_final = []
    for p, parameter in enumerate(parameters):
        for wf, transform, wf_parm in zip(wfs, transforms, parameter):
            for k, it in transform.deserialize(wf, wf_parm).items():
                wf.parameters[k] = it

        phase, log_vals = [
            np.nan_to_num(np.array(x))
            for x in zip(*[wf.recompute(configs) for wf in wfs])
        ]
        normalized_values = phase * np.nan_to_num(np.exp(log_vals - ref))
        energies = np.asarray([energy(configs, wf)["total"] for wf in wfs])

        overlap = np.einsum(
            "ik,jk->ijk", normalized_values.conj(), normalized_values / rho
        )
        energies = np.einsum("jk, ijk->ij", energies, overlap) / nconfig

        energy_final.append(energies)
        overlap_final.append(np.mean(overlap, axis=-1))

    for wf, transform, wf_parm in zip(wfs, transforms, p0):
        for k, it in transform.deserialize(wf, wf_parm).items():
            wf.parameters[k] = it
    return {
        "energy": np.asarray(energy_final),
        "overlap": np.asarray(overlap_final),
    }


def find_move_from_line(
    x,
    data,
    overlap_penalty,
    norm_penalty,
    offdiagonal_energy_penalty,
    max_norm_deviation=0.2,
):
    """
    Given the data from correlated sampling, find the best move.

    Return:
    cost function
    xmin estimation
    """
    N = np.abs(data["overlap"].diagonal(axis1=1, axis2=2))
    Nij = np.asarray([np.sqrt(np.outer(a, a)) for a in N])

    energy = data["energy"] / Nij
    overlap = data["overlap"]
    # print("energy cost", np.sum(energy.diagonal(axis1=1,axis2=2),axis=1))
    # print("overlap cost",np.sum(np.triu(overlap**2,1),axis=(1,2)) )
    # print("offdiagonal_energy", energy)
    # print("norm",np.einsum('ijj->i', (overlap-1)**2 ))
    cost = (
        np.sum(energy.diagonal(axis1=1, axis2=2), axis=1)
        + overlap_penalty * np.sum(np.triu(overlap**2, 1), axis=(1, 2))
        + offdiagonal_energy_penalty * np.sum(np.triu(energy**2, 1), axis=(1, 2))
        + norm_penalty * np.einsum("ijj->i", (overlap - 1) ** 2)
    )

    # good_norms = np.prod(np.einsum('ijj->ij',np.abs(overlap-1) < max_norm_deviation),axis=1)
    # print("good norms", good_norms, 'cost', cost[good_norms])
    xmin = linemin.stable_fit(x, cost)
    return xmin, cost


def direct_move(grad, N=40, max_tstep=0.1):
    x = np.linspace(0, max_tstep, N)
    return [[-delta * g for g in grad] for delta in x], x


def optimize(
    wfs,
    configs,
    energy,
    transforms,
    hdf_file,
    overlap_penalty=0.5,
    nsteps=40,
    max_tstep=0.1,
    condition_epsilon=0.1,
    norm_penalty=0.01,
    offdiagonal_energy_penalty=0.1,
    vmc_options=None,
    client=None,
    npartitions=0,
    n_line_min=10,
):
    """

    norm_penalty:
       decrease this if the optimization seems to get 'stuck' (xmin is often zero), and the normalization is fixed to very close to 1 for all wave functions.
    """
    parameters = [
        transform.serialize_parameters(wf.parameters)
        for transform, wf in zip(transforms, wfs)
    ]
    if vmc_options is None:
        vmc_options = {"nblocks": 10, "nsteps": 40}
    data = {}
    for k in ["energy", "parameters", "norm", "overlap", "energy_error"]:
        data[k] = []
    for step in range(nsteps):
        data_weighted, data_unweighted, configs = sample_overlap(
            wfs,
            configs,
            energy,
            transforms,
            client=client,
            npartitions=npartitions,
            **vmc_options,
        )
        avg, error = average(data_weighted, data_unweighted)
        print("energy", avg["total"], error["total"])
        terms = collect_terms(avg, error)
        print("norm", terms["norm"])
        print("overlap", terms["overlap"][0, 1])
        derivative = objective_function_derivative(
            terms, overlap_penalty, norm_penalty, offdiagonal_energy_penalty
        )
        derivative_conditioned = [
            d / (terms[("condition", i)] + condition_epsilon)
            for i, d in enumerate(derivative)
        ]
        print("|gradient|", [np.linalg.norm(d) for d in derivative_conditioned])

        line_parameters, x = direct_move(
            derivative_conditioned, N=n_line_min, max_tstep=max_tstep
        )
        for line_p in line_parameters:
            for p, p0 in zip(line_p, parameters):
                p += p0

        correlated_data = correlated_sampling(
            wfs,
            configs,
            energy,
            transforms,
            line_parameters,
            client=client,
            npartitions=npartitions,
        )
        xmin, cost = find_move_from_line(
            x,
            correlated_data,
            overlap_penalty,
            norm_penalty,
            offdiagonal_energy_penalty,
        )
        print("line search", x, cost)
        print("choosing to move", xmin)
        parameters = [p - xmin * d for p, d in zip(parameters, derivative_conditioned)]
        for wf, transform, parm in zip(wfs, transforms, parameters):
            for k, it in transform.deserialize(wf, parm).items():
                wf.parameters[k] = it

        data = {
            "energy": avg["total"],
            "energy_error": error["total"],
            "norm": terms["norm"],
            "overlap": terms["overlap"],
            "max_tstep": max_tstep,
            "line_x": x,
            "line_cost": cost,
            "line_xmin": xmin,
        }
        for i, parm in enumerate(parameters):
            data[f"parameters_{i}"] = parm
            data[f"gradient_{i}"] = derivative_conditioned[i]

        hdf_save(
            hdf_file,
            data,
            {
                "condition_epsilon": condition_epsilon,
                "nconfig": configs.configs.shape[0],
                "overlap_penalty": overlap_penalty,
                "norm_penalty": norm_penalty,
                "offdiagonal_energy_penalty": offdiagonal_energy_penalty,
            },
            wfs,
        )


class AdamMove:
    def __init__(self, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, g, m_old, v_old, t):
        m_new = self.beta1 * m_old + (1 - self.beta1) * g
        v_new = self.beta2 * v_old + (1 - self.beta2) * g**2
        mhat = m_new / (1 - self.beta1**t)
        vhat = v_new / (1 - self.beta2**t)
        theta_move = -self.alpha * mhat / (np.sqrt(vhat) + self.epsilon)
        return theta_move, m_new, v_new


def optimize_adam(
    wfs,
    configs,
    energy,
    transforms,
    hdf_file,
    penalty=0.5,
    nsteps=400,
    alpha=0.01,
    beta1=0.9,
):
    adam = AdamMove(alpha=alpha, beta1=beta1)
    parameters = [
        transform.serialize_parameters(wf.parameters)
        for transform, wf in zip(transforms, wfs)
    ]
    m_adam = [np.zeros_like(x) for x in parameters]
    v_adam = [np.zeros_like(x) for x in parameters]
    data = {}
    for k in ["energy", "parameters", "norm", "overlap", "energy_error"]:
        data[k] = []
    for step in range(nsteps):
        data_weighted, data_unweighted, configs = sample_overlap_worker(
            wfs, configs, energy, transforms, nsteps=10, nblocks=40
        )
        avg, error = average(data_weighted, data_unweighted)
        print("energy", avg["total"], error["total"])
        terms = collect_terms(avg, error)
        print("norm", terms["norm"])
        print("overlap", terms["overlap"][0, 1])
        derivative = objective_function_derivative(terms, penalty)
        derivative_conditioned = [
            d / np.sqrt(condition.diagonal())
            for d, condition in zip(derivative, terms["condition"])
        ]
        print("|gradient|", [np.linalg.norm(d) for d in derivative_conditioned])

        adam_moves = [
            adam.update(g, m, v, step + 1)
            for g, m, v in zip(derivative_conditioned, m_adam, v_adam)
        ]
        m_adam = [move[1] for move in adam_moves]
        v_adam = [move[2] for move in adam_moves]
        parameters = [parm + move[0] for parm, move in zip(parameters, adam_moves)]
        print("parameters", [param.real.round(3) for param in parameters])
        for wf, transform, parm in zip(wfs, transforms, parameters):
            for k, it in transform.deserialize(wf, parm).items():
                wf.parameters[k] = it

        data = {
            "energy": avg["total"],
            "energy_error": error["total"],
            "norm": terms["norm"],
            "overlap": terms["overlap"][0, 1],
        }
        for i, parm in enumerate(parameters):
            data[f"parameters_{i}"] = parm

        hdf_save(
            hdf_file,
            data,
            {
                "alpha": alpha,
                "beta1": beta1,
                "nconfig": configs.configs.shape[0],
                "penalty": penalty,
            },
        )
