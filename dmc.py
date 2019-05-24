# This must be done BEFORE importing numpy or anything else. 
# Therefore it must be in your main script.
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from pyscf import lib, gto, scf
from slater import PySCFSlaterRHF
from jastrow import Jastrow2B
from multiplywf import MultiplyWF
from accumulators import EnergyAccumulator
from mc import initial_guess, limdrift, vmc
from func3d import ExpCuspFunction, GaussianFunction
from optvariance import optvariance

def dmc(mol,wf,configs,nsteps=1000,tstep=0.02,branchtime=5,accumulators=None,verbose=False):
    if accumulators is None:
        accumulators={'energy':EnergyAccumulator(mol) } 
    nconfig, nelec=configs.shape[0:2]
    wf.recompute(configs)
    
    weights = np.ones(nconfig)
    npropagate = int(np.ceil(nsteps/branchtime))
    df=[]
    for step in range(npropagate):
        #print("branch step",step)
        df_,configs,weights = dmc_propagate(mol,wf,configs,weights,tstep,nsteps=branchtime,accumulators=accumulators)
        df.extend(df_)
        configs, weights = branch(configs, weights)
    return df
    

def dmc_propagate(mol,wf,configs,weights,tstep,nsteps=5,accumulators=None,verbose=False):
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
            grad=limdrift(wf.gradient(e, configs[:,e,:]).T)
            gauss = np.random.normal(scale=np.sqrt(tstep),size=(nconfig,3))
            eposnew=configs[:,e,:] + gauss + grad*tstep

            # Compute reverse move
            new_grad=limdrift(wf.gradient(e, eposnew).T) 
            forward=np.sum((configs[:,e,:]+tstep*grad-eposnew)**2,axis=1)
            backward=np.sum((eposnew+tstep*new_grad-configs[:,e,:])**2,axis=1)
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
        eloc = accumulators['energy'](configs, wf)['total'] # TODO we're computing the same total energy twice (again in accumulator loop)
        #print(elocold, eloc, eref)
        weights *= np.exp( -tstep*0.5*(elocold+eloc-2*eref) )
        wavg = np.mean(weights)
        eref = np.dot(weights,eloc)/nconfig/wavg #-= np.log(wavg)

        avg = {}
        for k,accumulator in accumulators.items():
            dat=accumulator(configs,wf)
            for m,res in dat.items():
                avg[k+m]=np.dot(weights,res)/nconfig/wavg
        avg['dmcweight'] = wavg
        avg['dmcweightvar'] = np.std(weights)
        avg['dmcweightmin'] = np.amin(weights)
        avg['dmcweightmax'] = np.amax(weights)
        avg['dmceref'] = eref
        avg['dmcacceptance'] = np.mean(acc)
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
    import pandas as pd
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', ecp='bfd',basis='bfd_vtz',unit='bohr',verbose=5)
    #mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5',basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=1000
    wf1=PySCFSlaterRHF(nconf,mol,mf)
    wf2=Jastrow2B(nconf,mol,
        basis=[ExpCuspFunction(.4,.6)]+[GaussianFunction(0.2*2**n) for n in range(1,4+1)])
    wf=MultiplyWF(nconf,wf1,wf2)
    #params0={'wf2coeff':np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
    params0={ 'wf2coeff':np.array([-0.33,  -0.8, 0.3,  -0.00 , -0.1])} # these were close to var-optimized values
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]
    configs = initial_guess(mol,nconf) 
    #import pickle
    #with open('configs_Li2_ecp.pickle','rb') as f:
    #    configs=pickle.load(f)
    tstart=time.process_time()
    print('Starting optimization')
    df1,configs = vmc(wf,configs,nsteps=100,accumulators={}) 
    df1=pd.DataFrame(df1)
    df1['method']=['op0']*len(df1)
    print('vmc ({0} steps) finished, {1}'.format(100,time.process_time()-tstart))
    #opt_var=optvariance(EnergyAccumulator(mol),wf,configs,params0)
    #print(params0)
    #print('optimization 1: {0}, {1}'.format(opt_var,time.process_time()-tstart))
    #nvmcsteps=30
    #df2,configs = vmc(wf,configs,nsteps=nvmcsteps,accumulators={}) 
    #df2=pd.DataFrame(df2)
    #df2['method']=['op1']*len(df2)
    #print('vmc ({0} steps) finished, {1}'.format(nvmcsteps,time.process_time()-tstart))
    #opt_var=optvariance(EnergyAccumulator(mol),wf,configs,params0)
    #print(params0)
    #print('optimization 2: {0}, {1}'.format(opt_var,time.process_time()-tstart))
    #df3,configs = vmc(wf,configs,nsteps=nvmcsteps,accumulators={}) 
    #df3=pd.DataFrame(df3)
    #df3['method']=['op2']*len(df3)
    #print('vmc ({0} steps) finished, {1}'.format(nvmcsteps,time.process_time()-tstart))
    #opt_var=optvariance(EnergyAccumulator(mol),wf,configs,params0)
    #print(params0)
    #print('optimization 3: {0}, {1}'.format(opt_var,time.process_time()-tstart))

    def dipole(configs,wf):
        return {'vec':np.sum(configs[:,:,:],axis=1) } 

    nsteps=300

    tstart=time.process_time()
    dfdmc=dmc(mol,wf,configs,nsteps=nsteps,accumulators={'energy':EnergyAccumulator(mol), 'dipole':dipole } )
    tend=time.process_time()
    print("DMC took",tend-tstart,"seconds") 
    dfdmc=pd.DataFrame(dfdmc)
    dfdmc['method']=['dmc']*len(dfdmc)

    tstart=time.process_time()
    dfvmc,configs_=vmc(wf,configs,nsteps=nsteps,accumulators={'energy':EnergyAccumulator(mol), 'dipole':dipole } )
    tend=time.process_time()
    print("VMC took",tend-tstart,"seconds") 
    dfvmc=pd.DataFrame(dfvmc)
    #with open('configs_Li2_ecp.pickle','wb') as f:
    #    pickle.dump(configs_,f)

    dfvmc['method']=['vmc']*len(dfvmc)

    df=pd.concat([dfvmc,dfdmc])
    df.to_csv("data.csv")
    warmup=30
    
    print('mean field',mf.energy_tot(),
          'vmc estimation', np.mean(dfvmc['energytotal'][warmup:]),np.std(dfvmc['energytotal'][warmup:])/np.sqrt(len(dfvmc)-warmup),
          'dmc estimation', np.mean(dfdmc['energytotal'][warmup:]),np.std(dfdmc['energytotal'][warmup:])/np.sqrt(len(dfdmc)-warmup))
    print('dipole',np.mean(np.asarray(dfdmc['dipolevec'][warmup:]),axis=0))
    
    g = sns.FacetGrid(df, hue='method')
    g.map(plt.plot,'energytotal')
    plt.legend()
    plt.show()
    

if __name__=="__main__":
    import cProfile, pstats, io
    from pstats import Stats
    pr = cProfile.Profile()
    pr.enable()
    test()
    pr.disable()
    p=Stats(pr)
    #print(p.sort_stats('cumulative').print_stats())
    
