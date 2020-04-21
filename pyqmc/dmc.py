# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pyqmc.mc as mc
import sys
import pandas as pd
import h5py


def limdrift(g, tau, acyrus=0.25):
    """
    Use Cyrus Umrigar's algorithm to limit the drift near nodes.

    Args:
      g: a [nconf,ndim] vector

      tau: time step
      
      acyrus: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    tot = np.linalg.norm(g, axis=1) * acyrus
    mask = tot > 1e-8
    taueff = np.ones(tot.shape) * tau
    taueff[mask] = (np.sqrt(1 + 2 * tau * tot[mask]) - 1) / tot[mask]
    return g * taueff[:, np.newaxis]


def limdrift_cutoff(g, tau, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    return mc.limdrift(g, cutoff) * tau


def dmc_propagate(
    wf,
    configs,
    weights,
    tstep,
    branchcut_start,
    branchcut_stop,
    eref,
    nsteps=5,
    accumulators=None,
    ekey=("energy", "total"),
    drift_limiter=limdrift,
    stepoffset=0,
):
    """
    Propagate DMC without branching
    
    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators

      configs: Configs object, (nconfig, nelec, 3) - initial coordinates to start calculation.

      weights: (nconfig,) - initial weights to start calculation

      tstep: Time step for move proposals. Introduces time step error.

      nsteps: number of DMC steps to take

      accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.

      ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]

      drift_limiter: a function that takes a gradient and a cutoff and returns an adjusted gradient

      stepoffset: what to start the step numbering at.

    Returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation
      
    """
    assert accumulators is not None, "Need an energy accumulator for DMC"
    nconfig, nelec = configs.configs.shape[0:2]
    wf.recompute(configs)

    eloc = accumulators[ekey[0]](configs, wf)[ekey[1]].real
    # eref_mean = np.mean(weights * eloc) / np.mean(weights)
    # eref = eref_mean
    df = []
    for step in range(nsteps):
        acc = np.zeros(nelec)
        for e in range(nelec):
            # Propose move
            grad = drift_limiter(np.real(wf.gradient(e, configs.electron(e)).T), tstep)
            gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconfig, 3))
            eposnew = configs.configs[:, e, :] + gauss + grad
            newepos = configs.make_irreducible(e, eposnew)

            # Compute reverse move
            new_grad = drift_limiter(np.real(wf.gradient(e, newepos).T), tstep)
            forward = np.sum(gauss ** 2, axis=1)
            backward = np.sum((gauss + grad + new_grad) ** 2, axis=1)
            # forward = np.sum((configs[:, e, :] + grad - eposnew) ** 2, axis=1)
            # backward = np.sum((eposnew + new_grad - configs[:, e, :]) ** 2, axis=1)
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))

            # Acceptance -- fixed-node: reject if wf changes sign
            wfratio = wf.testvalue(e, newepos)
            ratio = np.abs(wfratio) ** 2 * t_prob
            if not wf.iscomplex:
                ratio *= np.sign(wfratio)
            accept = ratio > np.random.rand(nconfig)

            # Update wave function
            configs.move(e, newepos, accept)
            wf.updateinternals(e, newepos, mask=accept)
            acc[e] = np.mean(accept)

        # weights
        elocold = eloc.copy()
        energydat = accumulators[ekey[0]](configs, wf)
        eloc = energydat[ekey[1]].real
        tdamp = limit_timestep(
            weights, eloc, elocold, eref, branchcut_start, branchcut_stop
        )
        wmult = np.exp(-tstep * 0.5 * tdamp * (elocold + eloc - 2 * eref))
        wmult[wmult > 2.0] = 2.0
        weights *= wmult
        wavg = np.mean(weights)

        avg = {}
        for k, accumulator in accumulators.items():
            if k != ekey[0]:
                dat = accumulator(configs, wf)
            else:
                dat = energydat
            for m, res in dat.items():
                avg[k + m] = np.einsum("...i,i...->...", weights, res) / (
                    nconfig * wavg
                )
        avg["weight"] = wavg
        avg["weightvar"] = np.std(weights)
        avg["weightmin"] = np.amin(weights)
        avg["weightmax"] = np.amax(weights)
        avg["acceptance"] = np.mean(acc)
        avg["step"] = stepoffset + step

        df.append(avg)
    return pd.DataFrame(df), configs, weights


def limit_timestep(weights, elocnew, elocold, eref, start, stop):
    """
    Stabilizes weights by scaling down the effective tstep if the local energy is too far from eref.

    Args:
      weights: (nconfigs,) array
        walker weights
      elocnew: (nconfigs,) array
        current local energy of each walker
      elocold: (nconfigs,) array
        previous local energy of each walker
      eref: scalar
        reference energy that fixes normalization
      start: scalar
        number of sigmas to start damping tstep
      stop: scalar
        number of sigmas where tstep becomes zero
    
    Return:
      tdamp: scalar
        Damping factor to multiply timestep; always between 0 and 1. The damping factor is 
            1 if eref-eloc < branchcut_start*sigma, 
            0 if eref-eloc > branchcut_stop*sigma,  
            decreases linearly inbetween.
    """
    if start is None or stop is None:
        return 1
    assert (
        stop > start
    ), "stabilize weights requires stop>start. Invalid stop={0}, start={1}".format(
        stop, start
    )
    eloc = np.stack([elocnew, elocold])
    fbet = np.amax(eref - eloc, axis=0)
    tdamp = np.clip((1 - (fbet - start)) / (stop - start), 0, 1)
    return tdamp


def branch(configs, weights):
    """
    Perform branching on a set of walkers  by stochastic reconfiguration

    Walkers are resampled with probability proportional to the weights, and the new weights are all set to be equal to the average weight.
    
    Args:
      configs: (nconfig,nelec,3) walker coordinates

      weights: (nconfig,) walker weights

    Returns:
      configs: resampled walker configurations

      weights: (nconfig,) all weights are equal to average weight
    """
    nconfig = configs.configs.shape[0]
    wtot = np.sum(weights)
    probability = np.cumsum(weights / wtot)
    base = np.random.rand()
    newinds = np.searchsorted(probability, (base + np.arange(nconfig) / nconfig) % 1.0)
    configs.resample(newinds)
    weights.fill(wtot / nconfig)
    return configs, weights


def dmc_file(hdf_file, data, attr, configs, weights):
    import pyqmc.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data.loc[0], attr)
                configs.initialize_hdf(hdf)
            if "weights" not in hdf.keys():
                hdf.create_dataset("weights", weights.shape)
            for i in range(len(data)):
                hdftools.append_hdf(hdf, data.loc[i])
            configs.to_hdf(hdf)
            hdf["weights"][:] = weights


def rundmc(
    wf,
    configs,
    weights=None,
    tstep=0.01,
    nsteps=1000,
    branchtime=5,
    stepoffset=0,
    branchcut_start=3,
    branchcut_stop=6,
    drift_limiter=limdrift,
    verbose=False,
    accumulators=None,
    ekey=("energy", "total"),
    propagate=dmc_propagate,
    feedback=1.0,
    hdf_file=None,
    **kwargs,
):
    """
    Run DMC 
    
    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators

      configs: (nconfig, nelec, 3) - initial coordinates to start calculation.

      weights: (nconfig,) - initial weights to start calculation, defaults to uniform.

      nsteps: number of DMC steps to take

      tstep: Time step for move proposals. Introduces time step error.

      branchtime: number of steps to take between branching

      accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.

      ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]

      verbose: Print out step information 

      drift_limiter: a function that takes a gradient and a cutoff and returns an adjusted gradient

      stepoffset: If continuing a run, what to start the step numbering at.

    Returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation
      
    """
    # Restart
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" in hdf.keys():
                stepoffset = hdf["step"][-1] + 1
                configs.load_hdf(hdf)
                weights = np.array(hdf["weights"])
                if verbose:
                    print("Restarted calculation")

    nconfig, nelec = configs.configs.shape[0:2]
    if weights is None:
        weights = np.ones(nconfig)

    npropagate = int(np.ceil(nsteps / branchtime))
    df = []

    df_, configs, weights = propagate(
        wf,
        configs,
        weights,
        tstep,
        branchcut_start=1e8,
        branchcut_stop=1e9,
        eref=0.0,
        nsteps=1,
        stepoffset=0,
        accumulators=accumulators,
        ekey=ekey,
        drift_limiter=drift_limiter,
        **kwargs,
    )

    df_ = pd.DataFrame(df_)
    eref = df_[ekey[0] + ekey[1]][0].real
    esigma = np.abs(eref) / 100
    for step in range(npropagate):
        if verbose:
            print("branch step", step, flush=True)
        df_, configs, weights = propagate(
            wf,
            configs,
            weights,
            tstep,
            branchcut_start * esigma,
            branchcut_stop * esigma,
            eref=eref,
            nsteps=branchtime,
            stepoffset=branchtime * step + stepoffset,
            accumulators=accumulators,
            ekey=ekey,
            drift_limiter=drift_limiter,
            **kwargs,
        )
        dmc_file(hdf_file, df_, dict(tstep=tstep), configs, weights)
        df_["eref"] = eref
        # print(df_)
        df.append(df_)
        eref = df_[ekey[0] + ekey[1]].values[-1].real - feedback * np.log(
            np.mean(weights)
        )
        configs, weights = branch(configs, weights)
    return pd.concat(df).reset_index(), configs, weights
