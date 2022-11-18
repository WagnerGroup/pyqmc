import os
import numpy as np
import pyqmc.mc as mc
import sys
import h5py
import logging


def limdrift(g, tau, acyrus=0.25):
    """
    Use Cyrus Umrigar's algorithm to limit the drift near nodes.

    :parameter g: a [nconf,ndim] vector
    :parameter tau: time step
    :parameter acyrus: the maximum magnitude
    :returns: The vector with the cut off applied and multiplied by tau.
    """
    v2 = np.sum(g**2, axis=1)
    mask = v2 > 1e-8
    taueff = np.ones(v2.shape) * tau
    taueff[mask] = (np.sqrt(1 + 2 * tau * acyrus * v2[mask]) - 1) / (acyrus * v2[mask])
    return g * taueff[:, np.newaxis]


def get_V2(configs, wf, acc_out):
    if "grad2" in acc_out.keys():
        return acc_out["grad2"]

    nconfig, nelec = configs.configs.shape[0:2]
    v2 = np.zeros(nconfig)
    for e in range(nelec):
        v2 += np.sum(np.abs(wf.gradient(e, configs.electron(e))).T ** 2, axis=1)
    return v2


def propose_drift_diffusion(wf, configs, tstep, e):
    nconfig = configs.configs.shape[0]
    grad = limdrift(np.real(wf.gradient(e, configs.electron(e)).T), tstep)
    gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconfig, 3))
    eposnew = configs.configs[:, e, :] + gauss + grad
    newepos = configs.make_irreducible(e, eposnew)

    # Compute reverse move
    g, wfratio, saved = wf.gradient_value(e, newepos)
    new_grad = limdrift(np.real(g.T), tstep)
    forward = np.sum(gauss**2, axis=1)
    backward = np.sum((gauss + grad + new_grad) ** 2, axis=1)
    t_prob = np.exp(1 / (2 * tstep) * (forward - backward))

    # Acceptance -- fixed-node: reject if wf changes sign
    ratio = np.abs(wfratio) ** 2 * t_prob
    if not wf.iscomplex:
        ratio *= np.sign(wfratio)
    accept = ratio > np.random.rand(nconfig)
    r2 = np.sum((gauss + grad) ** 2, axis=1)

    return newepos, accept, r2, saved


def propose_tmoves(wf, configs, energy_accumulator, tstep, e):
    """
    No side effect calculation of t-moves

    Returns:
       new proposed positions
       probability of acceptance
       sum of weights
    """
    moves = energy_accumulator.nonlocal_tmoves(configs, wf, e, tstep)
    t_amplitudes = moves["ratio"] * moves["weight"]

    forward_probability = np.zeros_like(t_amplitudes)
    forward_probability[t_amplitudes > 0] = t_amplitudes[t_amplitudes > 0]
    norm = 1.0 + np.sum(forward_probability, axis=1)  # EQN 34

    def select_walker(array):
        r = np.random.rand()
        return np.searchsorted(array, r)

    cdf = np.cumsum(forward_probability / norm[:, np.newaxis], axis=1)
    selected_moves = np.apply_along_axis(select_walker, 1, cdf)
    move_selected = selected_moves < t_amplitudes.shape[1]

    newpos = np.zeros((norm.shape[0], 3))
    reverse_ratio = np.zeros((norm.shape[0]))
    backward_amplitudes = t_amplitudes.copy()
    for walker, move in enumerate(selected_moves):
        if move_selected[walker]:
            newpos[walker, :] = moves["configs"].configs[walker, move, :]
            reverse_ratio[walker] = 1.0 / moves["ratio"][walker, move]
            backward_amplitudes[walker, :] *= reverse_ratio[walker]
            # This is the move back to the original position
            backward_amplitudes[walker, move] = (
                reverse_ratio[walker] * moves["weight"][walker, move]
            )
        else:
            newpos[walker, :] = configs.configs[walker, e, :]
            reverse_ratio[walker] = 0.0

    newpos = configs.make_irreducible(e, newpos)

    backward_amplitudes[backward_amplitudes < 0] = 0.0
    back_norm = 1.0 + np.sum(backward_amplitudes, axis=1)
    acceptance = norm / back_norm
    acceptance[move_selected == False] = 0.0

    return newpos, move_selected, acceptance, np.sum(t_amplitudes)


def dmc_propagate(
    wf,
    configs,
    weights,
    tstep,
    branchcut_start,
    e_trial,
    e_est,
    nsteps=5,
    accumulators=None,
    ekey=("energy", "total"),
):
    """
    Propagate DMC without branching

    :parameter wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators
    :parameter configs: Configs object, (nconfig, nelec, 3) - initial coordinates to start calculation.
    :parameter weights: (nconfig,) - initial weights to start calculation
    :parameter tstep: Time step for move proposals. Introduces time step error.
    :parameter nsteps: number of DMC steps to take
    :parameter accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.
    :parameter ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]
    :returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation

    """
    assert accumulators is not None, "Need an energy accumulator for DMC"
    nconfig, nelec = configs.configs.shape[0:2]
    wf.recompute(configs)

    energy_acc = accumulators[ekey[0]](configs, wf)
    eloc = energy_acc[ekey[1]].real
    v2 = get_V2(configs, wf, energy_acc)
    df = []

    for _ in range(nsteps):
        r2_accepted = np.zeros(nconfig)
        r2_proposed = np.zeros(nconfig)
        prob_acceptance = np.zeros(nconfig)
        tmove_acceptance = np.zeros(nconfig)

        if accumulators[ekey[0]].has_nonlocal_moves():
            for e in range(nelec):  # T-moves
                newepos, mask, probability, ecp_totweight = propose_tmoves(
                    wf, configs, accumulators[ekey[0]], tstep, e
                )
                accept = mask & (probability > np.random.rand(nconfig))
                configs.move(e, newepos, accept)
                wf.updateinternals(e, newepos, configs, mask=accept)
                tmove_acceptance += accept / nelec

        for e in range(nelec):  # drift-diffusion
            newepos, accept, r2, saved = propose_drift_diffusion(wf, configs, tstep, e)
            configs.move(e, newepos, accept)
            wf.updateinternals(e, newepos, configs, mask=accept, saved_values=saved)
            r2_proposed += r2
            r2_accepted[accept] += r2[accept]
            prob_acceptance += accept / nelec

        # weights
        elocold = eloc.copy()
        v2old = v2.copy()
        energydat = accumulators[ekey[0]](configs, wf)
        eloc = energydat[ekey[1]].real

        tdamp = r2_accepted / r2_proposed
        v2 = get_V2(configs, wf, energydat)

        Snew = compute_S(e_trial, e_est, branchcut_start, v2, tstep, eloc, nelec)
        Sold = compute_S(e_trial, e_est, branchcut_start, v2old, tstep, elocold, nelec)
        wmult = np.exp(tstep * tdamp * (0.5 * Snew + 0.5 * Sold))
        weights *= wmult
        wavg = np.mean(weights)

        avg = {}
        for k, accumulator in accumulators.items():
            dat = accumulator(configs, wf) if k != ekey[0] else energydat
            for m, res in dat.items():
                avg[k + m] = np.einsum("...i,i...->...", weights, res) / (
                    nconfig * wavg
                )
        avg["weight"] = wavg
        avg["acceptance"] = np.mean(prob_acceptance)
        avg["tmove_acceptance"] = np.mean(tmove_acceptance)
        df.append(avg)
    weight = np.asarray([d["weight"] for d in df])
    avg_weight = weight / np.mean(weight)
    df_ret = {
        k: np.mean([d[k] * w for d, w in zip(df, avg_weight)], axis=0)
        for k in df[0].keys()
    }

    df_ret["weight"] = np.mean(weight)

    return df_ret, configs, weights


def compute_S(e_trial, e_est, branchcut, v2, tau, eloc, nelec):
    e_cut = e_est - eloc
    mask = np.abs(e_cut) > branchcut
    e_cut[mask] = branchcut * np.sign(e_cut[mask])
    denominator = 1 + (v2 * tau / nelec) ** 2

    return e_trial - e_est + e_cut / denominator


def dmc_propagate_parallel(wf, configs, weights, client, npartitions, *args, **kwargs):
    r"""Parallelizes calls to dmc_propagate by splitting configs

    If npartitions does not evenly divide nconfigs, we need to reweight the results based on the number of configs per parallel task.

    The final result should be equivalent to the non-parallelized case.
    The average weight :math:`w` and the weighted average of observables :math:`\langle O \rangle` are returned.
    Index :math:`i` refers to walker index.

    .. math::
        w = \sum_i w_i / n_{\rm config}
        \qquad\quad \langle O \rangle = \sum_i o_{i}  w_i / \sum_i w_i

    Split over parallel tasks, we need to reweight by number of walkers.
    The average weight :math:`w_p` and weighted average of observables :math:`\langle O\rangle_p` are returned from each task.

    .. math::
        w_p = \sum_j^{{\rm task}\, p} w_j / n_{{\rm config}, p}
        \qquad\quad \langle O \rangle_p = \frac{\sum_j^{{\rm task}\, p} o_{j}  w_j }{ \sum_j^{{\rm task}\, p} w_j }


    The total weight and total average (defined above) are computed from the task weights :math:`w_p` and task averages :math:`\langle O\rangle_p` as

    .. math::
        w = \sum_p w_p n_{{\rm config}, p} /  n_{\rm config},
        \qquad\quad \langle O \rangle = \frac{ \sum_p \langle O\rangle_p  \sum_j^{{\rm task}\, p} w_j }{ \sum_i w_i}.

    We can rewrite the weights using the equations above

    .. math::
        \langle O \rangle &= \frac{ \sum_p \langle O\rangle_p w_p  n_{{\rm config}, p}  }{ w n_{\rm config} }

        &= \sum_p \langle O\rangle_p \frac{w_p n_{{\rm config}, p}}{\sum_p w_p n_{{\rm config}, p}}


    By reweighting the task weights as :math:`\overline{w}_p = w_p n_{{\rm config}, p}`, we can omit the reweighting factor :math:`\frac{n_{{\rm config}, p}}{n_{\rm config}}` (that we use to collect parallel vmc).
    Instead, we use only the reweighting factor :math:`\overline{w}_p / \sum_p \overline{w}_p`

    .. math:: \langle O \rangle = \sum_p \langle O\rangle_p \frac{\overline{w}_p }{\sum_p \overline{w}_p }
    """

    config = configs.split(npartitions)
    weight = np.array_split(weights, npartitions)
    runs = [
        client.submit(dmc_propagate, wf, conf, wt, *args, **kwargs)
        for conf, wt in zip(config, weight)
    ]
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[1])
    weights = np.concatenate(allresults[2])
    confweight = np.array([len(c.configs) for c in config], dtype=float)
    weight = np.array([w["weight"] for w in allresults[0]]) * confweight
    weight_avg = weight / np.sum(weight)
    block_avg = {
        k: np.sum(
            [res[k] * ww for res, ww in zip(allresults[0], weight_avg)],
            axis=0,
        )
        for k in allresults[0][0].keys()
    }
    block_avg["weight"] = np.mean(weight)
    return block_avg, configs, weights


def branch(configs, weights):
    """
    Perform branching on a set of walkers using the 'stochastic comb'

    Walkers are resampled with probability proportional to the weights, and the new weights are all set to be equal to the average weight.

    :parameter configs: (nconfig,nelec,3) walker coordinates
    :parameter weights: (nconfig,) walker weights
    :returns: resampled walker configurations and weights all equal to average weight
    """

    nconfig = configs.configs.shape[0]
    if np.any(weights > 2.0):
        logging.warning("Some weights are larger than 2")
    probability = np.cumsum(weights)
    wtot = probability[-1]

    base = np.random.rand() * wtot
    newinds = np.searchsorted(
        probability, (base + np.linspace(0, wtot, nconfig, endpoint=False)) % wtot
    )
    unique, counts = np.unique(newinds, return_counts=True)

    configs.resample(newinds)
    weights.fill(wtot / nconfig)
    return (
        configs,
        weights,
        {
            "max branches": np.max(counts),
            "Number of walkers killed": nconfig - unique.shape[0],
        },
    )


def dmc_file(hdf_file, data, attr, configs, weights):
    import pyqmc.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
            if "weights" not in hdf.keys():
                hdf.create_dataset("weights", weights.shape)
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)
            hdf["weights"][:] = weights


def evaluate_energy_worker(configs, wf, en):
    wf.recompute(configs)
    return en(configs, wf)


def evaluate_energies(wf, configs, en, client, npartitions):
    if client is None:
        return evaluate_energy_worker(configs, wf, en)

    else:
        config = configs.split(npartitions)
        runs = [client.submit(evaluate_energy_worker, conf, wf, en) for conf in config]
        ret = {}
        data = [r.result() for r in runs]
        for k in data[0].keys():
            ret[k] = np.concatenate([d[k] for d in data])
        return ret


def rundmc(
    wf,
    configs,
    weights=None,
    tstep=0.01,
    nsteps=1000,
    branchtime=5,
    stepoffset=0,
    branchcut_start=10,
    verbose=False,
    accumulators=None,
    ekey=("energy", "total"),
    feedback=1.0,
    hdf_file=None,
    continue_from=None,
    client=None,
    npartitions=None,
    vmc_warmup=10,
    **kwargs,
):
    """
    Run DMC

    :parameter wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators
    :parameter configs: (nconfig, nelec, 3) - initial coordinates to start calculation.
    :parameter weights: (nconfig,) - initial weights to start calculation, defaults to uniform.
    :parameter nsteps: number of DMC steps to take
    :parameter tstep: Time step for move proposals. Introduces time step error.
    :parameter branchtime: number of steps to take between branching
    :parameter accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.
    :parameter ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]
    :parameter verbose: Print out step information
    :parameter stepoffset: If continuing a run, what to start the step numbering at.
    :parameter vmc_warmup: If starting a run, how many VMC warmup blocks to run
    :returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation

    """
    # Don't continue onto a file that's already there.
    if continue_from is not None and hdf_file is not None and os.path.isfile(hdf_file):
        raise RuntimeError(
            f"continue_from is set but hdf_file={hdf_file} already exists! Delete or rename {hdf_file} and try again."
        )

    # Restart if hdf_file is there
    if continue_from is None and hdf_file is not None and os.path.isfile(hdf_file):
        continue_from = hdf_file

    # Now we should be sure that there is a file
    # to continue from, if given.
    if continue_from is not None:
        with h5py.File(continue_from, "r") as hdf:
            stepoffset = hdf["step"][-1] + 1
            configs.load_hdf(hdf)
            weights = np.array(hdf["weights"])
            if "e_trial" not in hdf.keys():
                raise ValueError(
                    "Did not find e_trial in the restart file. This may mean that you are trying to restart from a different version of DMC"
                )
            e_trial = hdf["e_trial"][-1]
            e_est = hdf["e_est"][-1]
            esigma = hdf["esigma"][-1]
            if verbose:
                print(f"Restarting calculation {continue_from} from step {stepoffset}")
    else:
        df, configs = mc.vmc(
            wf,
            configs,
            client=client,
            npartitions=npartitions,
            verbose=verbose,
            nblocks=vmc_warmup,
        )
        en = evaluate_energies(wf, configs, accumulators[ekey[0]], client, npartitions)[
            ekey[1]
        ]
        eref = np.mean(en).real
        e_trial = eref
        e_est = eref
        esigma = np.std(en)
        if verbose:
            print("eref start", eref, "esigma", esigma)

    nconfig = configs.configs.shape[0]
    if weights is None:
        weights = np.ones(nconfig)

    npropagate = int(np.ceil(nsteps / branchtime))
    df = []
    for step in range(npropagate):
        if client is None:
            df_, configs, weights = dmc_propagate(
                wf,
                configs,
                weights,
                tstep,
                branchcut_start * esigma,
                e_trial=e_trial,
                e_est=e_est,
                nsteps=branchtime,
                accumulators=accumulators,
                ekey=ekey,
                **kwargs,
            )
        else:
            df_, configs, weights = dmc_propagate_parallel(
                wf,
                configs,
                weights,
                client,
                npartitions,
                tstep,
                branchcut_start * esigma,
                e_trial=e_trial,
                e_est=e_est,
                nsteps=branchtime,
                accumulators=accumulators,
                ekey=ekey,
                **kwargs,
            )

        df_["e_trial"] = e_trial
        df_["e_est"] = e_est
        df_["step"] = step + stepoffset
        df_["esigma"] = esigma
        df_["tstep"] = tstep
        df_["weight_std"] = np.std(weights)
        df_["nsteps"] = branchtime

        configs, weights, branch_info = branch(configs, weights)
        df_.update(branch_info)
        df.append(df_)
        dmc_file(hdf_file, df_, {}, configs, weights)
        
        e_est = estimate_energy(hdf_file, df, ekey)
        e_trial = e_est - feedback * np.log(np.mean(weights)).real

        if verbose:
            print(
                "energy",
                df_[ekey[0] + ekey[1]],
                "e_trial",
                e_trial,
                "e_est",
                e_est,
                "sigma(w)",
                df_["weight_std"],
            )
            print(branch_info)

    df_ret = {k: np.asarray([d[k] for d in df]) for k in df[0].keys()}
    return df_ret, configs, weights


def estimate_energy(hdf_file, df, ekey):
    if hdf_file is not None:
        with h5py.File(hdf_file, "r") as f:
            en = f[ekey[0] + ekey[1]][()]
            wt = f["weight"][()]
    else:
        en = np.asarray([d[ekey[0] + ekey[1]] for d in df])
        wt = np.asarray([d["weight"] for d in df])
    warmup = int(len(en) / 4)
    return np.average(en[warmup:], weights=wt[warmup:]).real
