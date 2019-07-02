# This must be done BEFORE importing numpy or anything else. 
# Therefore it must be in your main script.
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import pyqmc.mc as mc
import sys

def limdrift(g,tau,acyrus=0.25):
    """
    Use Cyrus Umrigar's algorithm to limit the drift near nodes.

    Args:
      g: a [nconf,ndim] vector

      tau: time step
      
      acyrus: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    tot=np.linalg.norm(g,axis=1)*acyrus
    mask=tot > 1e-8
    taueff=np.ones(tot.shape)*tau
    taueff[mask]=(np.sqrt(1+2*tau*tot[mask])-1)/tot[mask]
    return g*taueff[:,np.newaxis]


def limdrift_cutoff(g,tau,cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    return mc.limdrift(g,cutoff)*tau


def dmc(wf,configs,weights=None, nsteps=1000,tstep=0.02,branchtime=5, stepoffset=0,
        verbose=False, **dmc_prop_kwargs):
        #accumulators=None,ekey=('energy','total'),
        #branchcut_start=3, branchcut_stop=6, drift_limiter=limdrift):
    """
    Run DMC (not parallel)
    
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
    #assert accumulators is not None, "Need an energy accumulator for DMC"
    nconfig, nelec=configs.shape[0:2]
    if weights is None:
        weights = np.ones(nconfig)

    wf.recompute(configs)
    
    npropagate = int(np.ceil(nsteps/branchtime))
    df=[]
    for step in range(npropagate):
        if verbose:
            print("branch step",step, flush=True)
        df_,configs,weights = dmc_propagate(wf,configs,weights, tstep,
                nsteps=branchtime, stepoffset=branchtime*step+stepoffset, verbose=verbose,
                **dmc_prop_kwargs)
                #accumulators=accumulators,ekey=ekey, drift_limiter=drift_limiter)
        df.extend(df_)
        configs, weights = branch(configs, weights)
    return df, configs, weights
    

def dmc_propagate(wf,configs,weights,tstep,nsteps=5,accumulators=None,ekey=('energy','total'), verbose=False, branchcut_start=5, branchcut_stop=10, drift_limiter=limdrift,stepoffset=0):
    """
    Propagate DMC without branching
    
    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators

      configs: (nconfig, nelec, 3) - initial coordinates to start calculation.

      weights: (nconfig,) - initial weights to start calculation

      tstep: Time step for move proposals. Introduces time step error.

      nsteps: number of DMC steps to take

      accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.

      ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]

      verbose: Print out step information 

      drift_limiter: a function that takes a gradient and a cutoff and returns an adjusted gradient

      stepoffset: what to start the step numbering at.

    Returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation
      
    """
    assert accumulators is not None, "Need an energy accumulator for DMC"
    nconfig, nelec=configs.shape[0:2]
    wf.recompute(configs)

    eloc = accumulators[ekey[0]](configs, wf)[ekey[1]]
    eref_mean = np.mean(weights*eloc)/np.mean(weights)
    eref_sigma = np.mean(weights*eloc/np.mean(weights))
    eref=eref_mean
    df=[]
    for step in range(nsteps):
        acc=np.zeros(nelec)
        for e in range(nelec):
            # Propose move
            grad=drift_limiter(wf.gradient(e, configs[:,e,:]).T,tstep)
            gauss = np.random.normal(scale=np.sqrt(tstep),size=(nconfig,3))
            eposnew=configs[:,e,:] + gauss + grad

            # Compute reverse move
            new_grad=drift_limiter(wf.gradient(e, eposnew).T,tstep) 
            forward=np.sum((configs[:,e,:]+grad-eposnew)**2,axis=1)
            backward=np.sum((eposnew+new_grad-configs[:,e,:])**2,axis=1)
            t_prob = np.exp(1/(2*tstep)*(forward-backward))

            # Acceptance -- fixed-node: reject if wf changes sign
            wfratio = wf.testvalue(e,eposnew)
            ratio=wfratio**2*t_prob
            accept=ratio*np.sign(wfratio) > np.random.rand(nconfig)
            
            # Update wave function
            configs[accept,e,:]=eposnew[accept,:]
            wf.updateinternals(e,configs[:,e,:],accept)
            acc[e]=np.mean(accept)

        # weights
        elocold = eloc.copy()
        energydat=accumulators[ekey[0]](configs, wf)
        eloc = energydat[ekey[1]] 
        tdamp = stabilize_weights(weights, eloc, elocold, eref, eref_sigma, branchcut_start, branchcut_stop)
        wmult=np.exp( -tstep*0.5*tdamp*(elocold+eloc-2*eref) )
        wmult[wmult > 2.0] = 2.0
        weights*=wmult
        wavg = np.mean(weights)

        avg = {}
        for k,accumulator in accumulators.items():
            if k!=ekey[0]:
                dat=accumulator(configs,wf)
            else:
                dat=energydat
            for m,res in dat.items():
                avg[k+m]=np.dot(weights,res)/(nconfig*wavg)
        avg['weight'] = wavg
        avg['weightvar'] = np.std(weights)
        avg['weightmin'] = np.amin(weights)
        avg['weightmax'] = np.amax(weights)
        avg['eref'] = eref
        avg['acceptance'] = np.mean(acc)
        avg['step']=stepoffset+step
        if verbose:
            print("step",stepoffset+step,'acceptance', avg['acceptance'],
                  'weight',avg['weight'],'weightstd',avg['weightvar'],'eref',eref)
            print("eref_mean",eref_mean,"logwavg",np.log(wavg),'wmax',avg['weightmax'],
                 'wmin',avg['weightmin'])
            
        eref = eref_mean-np.log(wavg)
        df.append(avg)
    return df, configs, weights
    
def stabilize_weights(weights, elocnew, elocold, eref, erefsigma, branchcut_start, branchcut_stop):
    """
    Stabilizes weights by scaling down the effective tstep if the local energy is too far from eref.
    The damping factor is 
        1 if eref-eloc < branchcut_start*sigma 
        0 if eref-eloc > branchcut_stop*sigma  
        decreases linearly inbetween.

    Args:
      weights: (nconfigs,) array
        walker weights
      elocnew: (nconfigs,) array
        current local energy of each walker
      elocold: (nconfigs,) array
        previous local energy of each walker
      eref: scalar
        reference energy that fixes normalization
      branchcut_start: scalar
        number of sigmas to start damping tstep
      branchcut_stop: scalar
        number of sigmas where tstep becomes zero
    """
    assert branchcut_stop>branchcut_start, "stabilize weights requires branchcut_stop>branchcut_start. Invalid branchcut_stop={0}, branchcut_start={1}".format(branchcut_stop, branchcut_start)
    eloc = np.stack([elocnew,elocold])
    sigma = erefsigma #np.mean(weights[np.newaxis]*eloc/np.mean(weights))
    fbet = np.amax((eref-eloc)/sigma, axis=0)
    tdamp = np.clip((1-(fbet-branchcut_start))/(branchcut_stop-branchcut_start), 0, 1)
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
    nconfig = configs.shape[0]
    wtot = np.sum(weights)
    probability = np.cumsum(weights/wtot)
    newinds = np.searchsorted(probability, np.random.random(nconfig))
    configs = configs[newinds]
    weights.fill(wtot/nconfig)
    return configs, weights


