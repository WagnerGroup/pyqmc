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
import pyqmc.method.mc as mc
import scipy.stats
import pyqmc.method.linemin as linemin
import pyqmc.gpu as gpu
import os
import h5py


def run_vmc_many(wfs, configs, energy, nreps=1, **kwargs):
    # data gets saved at the end of each rep
    for i in range(nreps):
        sample_overlap(wfs, configs, energy, **kwargs)


def hdf_save(hdf_file, weighted, unweighted, attr, configs):
    import pyqmc.method.hdftools as hdftools

    if hdf_file is not None:
        fulldata = dict(weighted=weighted, unweighted=unweighted)
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, {}, attr)
                configs.initialize_hdf(hdf)
                for label, data in fulldata.items():
                    hdf.create_group(label)
            for label, data in fulldata.items():
                extend_hdf(hdf[label], data)
            configs.to_hdf(hdf)


def extend_hdf(f, data):
    for k, it in data.items():
        if k not in f.keys():
            f.create_dataset(
                k, (0, *it.shape[1:]), maxshape=(None, *it.shape[1:]), dtype=it.dtype
            )
        n = it.shape[0]
        f[k].resize((f[k].shape[0] + n), axis=0)
        f[k][-n:] = it


def compute_weights(wfs):
    """
    computes psi_i psi_j / rho for all i,j and for each configuration.
    Returns:
      weights[wfi, wfj, config]
    """
    phase, log_vals = [
        np.nan_to_num(np.array(x)) for x in zip(*[wf.value() for wf in wfs])
    ]
    ref = np.max(log_vals, axis=0)  # for numerical stability
    rho = np.mean(np.nan_to_num(np.exp(2 * (log_vals - ref))), axis=0)
    psi = phase * np.nan_to_num(np.exp(log_vals - ref))
    weights = np.einsum("ic,jc->ijc", psi.conj(), psi / rho)
    return weights


def invert_list_of_dicts(A, asarray=True):
    """
    if we have a list [ {'A':1,'B':2}, {'A':3, 'B':5}], invert the structure to
    {'A':[1,3], 'B':[2,5]}.
    If not all keys are present in all lists, error.
    """
    if asarray:
        return {k: np.asarray([a[k] for a in A]) for k in A[0].keys()}
    else:
        return {k: [a[k] for a in A] for k in A[0].keys()}


def sample_overlap_worker(wfs, configs, tstep, nsteps, nblocks, energy):
    r"""Run nstep Metropolis steps to sample a distribution proportional to
    :math:`\sum_i |\Psi_i|^2`, where :math:`\Psi_i` = wfs[i]
    """
    nconf, nelec, _ = configs.configs.shape
    weighted = []
    unweighted = []
    for block in range(nblocks):
        w, u, _ = sample_overlap_block(wfs, configs, tstep, nsteps, energy)
        weighted.append(w)
        unweighted.append(u)

    # here we modify the data so that weighted and unweighted are dictionaries of arrays
    # Access as weighted[quantity][block, ...]
    weighted = invert_list_of_dicts(weighted)
    unweighted = invert_list_of_dicts(unweighted)
    return weighted, unweighted, configs


def sample_overlap_block(wfs, configs, tstep, nsteps, energy):
    for wf in wfs:
        wf.recompute(configs)
    weighted_block = {}
    unweighted_block = {"acceptance": 0.0}
    nconf, nelec = configs.configs.shape[:2]

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

            ratio = t_prob * np.sum(wf_ratios * weights, axis=0) / weights.sum(axis=0)
            accept = ratio > np.random.rand(nconf)
            # block_avg["acceptance"][n] += accept.mean() / nelec

            # Update wave function
            configs.move(e, newcoorde, accept)
            for wf, saved in zip(wfs, saved_values):
                wf.updateinternals(
                    e, newcoorde, configs, mask=accept, saved_values=saved
                )

        weights = compute_weights(wfs)
        unweighted_dat = {}
        unweighted_dat["overlap"] = np.mean(weights, axis=-1)
        rolling_average(unweighted_block, unweighted_dat, nsteps)
        # Collect rolling average
        if energy is not None:
            weighted_dat = energy.avg(configs, wfs, weights)
            rolling_average(weighted_block, weighted_dat, nsteps)

    return weighted_block, unweighted_block, configs


def rolling_average(block, data, nsteps):
    for k, it in data.items():
        if k not in block:
            block[k] = np.zeros((*it.shape,), dtype=it.dtype)
        block[k] += it / nsteps


def sample_overlap(
    wfs,
    configs,
    energy,
    nsteps=10,
    nblocks=10,
    tstep=0.5,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """ """
    if client is None:  # otherwise running in parallel
        w, u, _ = sample_overlap_worker(wfs, configs, tstep, nsteps, nblocks, energy)
        hdf_save(hdf_file, w, u, dict(tstep=tstep), configs)
        return w, u, configs

    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    coord = configs.split(npartitions)
    runs = [
        client.submit(sample_overlap_worker, wfs, conf, tstep, nsteps, nblocks, energy)
        for conf in coord
    ]
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[2])
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    weighted = {}
    for k, it in invert_list_of_dicts(allresults[0]).items():
        weighted[k] = np.average(it, weights=confweight, axis=0)
    unweighted = {}
    for k, it in invert_list_of_dicts(allresults[1]).items():
        unweighted[k] = np.average(it, weights=confweight, axis=0)

    hdf_save(hdf_file, weighted, unweighted, dict(tstep=tstep), configs)
    return weighted, unweighted, configs


def normalize(weighted, unweighted):
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
