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
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    tot=np.linalg.norm(g,axis=1)*acyrus
    mask=tot > 1e-8
    taueff=np.ones(tot.shape)*tau
    taueff[mask]=(np.sqrt(1+2*tau*tot[mask])-1)/tot[mask]
    g*=taueff[:,np.newaxis]
    return g


def limdrift_cutoff(g,tau):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    return mc.limdrift(g)*tau



def dmc(mol,wf,configs,nsteps=1000,tstep=0.02,branchtime=5,accumulators=None,verbose=False,
        drift_limiter=limdrift):
    if accumulators is None:
        accumulators={'energy':EnergyAccumulator(mol) } 
    nconfig, nelec=configs.shape[0:2]
    wf.recompute(configs)
    
    weights = np.ones(nconfig)
    npropagate = int(np.ceil(nsteps/branchtime))
    df=[]
    for step in range(npropagate):
        #print("branch step",step)
        df_,configs,weights = dmc_propagate(mol,wf,configs,weights,
                tstep,nsteps=branchtime,accumulators=accumulators,
                step_offset=branchtime*step,
                verbose=verbose)
        df.extend(df_)
        configs, weights = branch(configs, weights)
    return df
    

def dmc_propagate(mol,wf,configs,weights,tstep,nsteps=5,accumulators=None,verbose=False,
        drift_limiter=limdrift,step_offset=0):
    if accumulators is None:
        accumulators={'energy':EnergyAccumulator(mol) } 
    nconfig, nelec=configs.shape[0:2]
    wf.recompute(configs)

    E = accumulators['energy'](configs, wf)
    eloc = E['total']
    eref = np.mean(eloc) 
    df=[]
    for step in range(nsteps):
        acc=np.zeros(nelec)
        for e in range(nelec):
            # Propose move
            grad=limdrift(wf.gradient(e, configs[:,e,:]).T,tstep)
            gauss = np.random.normal(scale=np.sqrt(tstep),size=(nconfig,3))
            eposnew=configs[:,e,:] + gauss + grad

            # Compute reverse move
            new_grad=limdrift(wf.gradient(e, eposnew).T,tstep) 
            forward=np.sum((configs[:,e,:]+tstep*grad-eposnew)**2,axis=1)
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
        energydat=accumulators['energy'](configs, wf)
        eloc = energydat['total'] # TODO this is fragile
        #print(elocold, eloc, eref)
        weights *= np.exp( -tstep*0.5*(elocold+eloc-2*eref) )
        wavg = np.mean(weights)
        eref = np.dot(weights,eloc)/nconfig/wavg #-= np.log(wavg)

        avg = {}
        for k,accumulator in accumulators.items():
            if k!='energy':
                dat=accumulator(configs,wf)
            else:
                dat=energydat
            for m,res in dat.items():
                avg[k+m]=np.dot(weights,res)/nconfig/wavg
        avg['weight'] = wavg
        avg['weightvar'] = np.std(weights)
        avg['weightmin'] = np.amin(weights)
        avg['weightmax'] = np.amax(weights)
        avg['eref'] = eref
        avg['acceptance'] = np.mean(acc)
        avg['step']=step_offset+step
        if verbose:
            print("step",step,'acceptance', avg['acceptance'],
                  'weight',avg['weight'],'weightstd',avg['weightvar'])
        df.append(avg)
    return df, configs, weights
    
def branch(configs, weights):
    nconfig = configs.shape[0]
    wtot = np.sum(weights)
    probability = np.cumsum(weights/wtot)
    newinds = np.searchsorted(probability, np.random.random(nconfig))
    configs = configs[newinds]
    weights.fill(wtot/nconfig)
    return configs, weights


def test():
    from pyscf import lib, gto, scf
    
    import pandas as pd
    import time
    from pyqmc import PySCFSlaterRHF,JastrowSpin,MultiplyWF, EnergyAccumulator, initial_guess, vmc,ExpCuspFunction, GaussianFunction, optvariance
    
    mol = gto.M(atom='N 0. 0. 0.; N 0. 0. 2.0', ecp='bfd',basis='bfd_vtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()

    
    nconf=300
    basis=[ExpCuspFunction(2.0,1.5),GaussianFunction(0.5),
            GaussianFunction(2.0),GaussianFunction(.25),
            GaussianFunction(1.0),GaussianFunction(4.0),
            GaussianFunction(8.0)  ]
    wf1=PySCFSlaterRHF(mol,mf)
    wf=MultiplyWF(wf1,JastrowSpin(mol,a_basis=basis,b_basis=basis))
    optcoeff=[-0.000654322,-0.0007121276,0.1976771403,0.2249699815,-0.0005538429,-0.0023647576,0.236036959,0.2880757827,0.0900150779,0.0950073603,-0.0156273348,-0.0163297269,0.0036333132,0.0043803894,-0.0025742353,-0.0198104585,-0.002161816,-0.0806744038,-0.1426759301,-0.0970115406,-0.0412832563,-0.0447487546,-0.0344625724,0.0097164571,-0.0804903747,-0.0450287743,-0.0819815013,-0.1000744454,-0.0752387958,-0.0176527963,-0.0489049681,-0.0159655513,-0.0105098245,-0.1184213151,-0.0104448328]
    split=len(optcoeff)-3*len(basis)
    
    wf.parameters['wf2acoeff']=np.asarray(optcoeff[0:split]).reshape(wf.parameters['wf2acoeff'].shape)
    wf.parameters['wf2bcoeff']=np.asarray(optcoeff[split:]).reshape(wf.parameters['wf2bcoeff'].shape)

    configs = initial_guess(mol,nconf)

    nsteps=200
    tstart=time.process_time()
    dfvmc,configs_=vmc(wf,configs,nsteps=10)
    tend=time.process_time()
    dfvmc=pd.DataFrame(dfvmc)
    print("VMC took",tend-tstart)
    #dfvmc['method']=['vmc']*len(dfvmc)
    

    dfall=pd.DataFrame()
    for tstep in [0.01]:
        for limnm,dlimiter in {'cyrus':limdrift}.items():
            print(limnm,tstep)
            dfdmc=dmc(mol,wf,configs,nsteps=nsteps,accumulators={'energy':EnergyAccumulator(mol)},
                    tstep=tstep,drift_limiter=dlimiter,verbose=True)
            dfdmc=pd.DataFrame(dfdmc)
            dfdmc['method']=['dmc']*len(dfdmc)
            dfdmc['limiter']=[limnm]*len(dfdmc)
            dfdmc['tstep']=[tstep]*len(dfdmc)
            dfall=dfall.append(dfdmc).reset_index(drop=True)
            print(dfdmc)
            dfall.to_json("dmcdata.json")



if __name__=="__main__":
    import cProfile, pstats, io
    from pstats import Stats
    pr = cProfile.Profile()
    pr.enable()
    test()
    pr.disable()
    p=Stats(pr)
    print(p.sort_stats('cumulative').print_stats())
    
