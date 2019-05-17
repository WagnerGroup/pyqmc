# This must be done BEFORE importing numpy or anything else. 
# Therefore it must be in your main script.
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from pyscf import lib, gto, scf
from slater import PySCFSlaterRHF
from accumulators import EnergyAccumulator

def initial_guess(mol,nconfig,r=1.0):
    """ Generate an initial guess by distributing electrons near atoms
    proportional to their charge."""
    nelec=np.sum(mol.nelec)
    configs=np.zeros((nconfig,nelec,3))
    wts=mol.atom_charges()
    wts=wts/np.sum(wts)

    ### This is not ideal since we loop over configs 
    ### Should figure out a way to throw configurations
    ### more efficiently.
    for c in range(nconfig):
        count=0
        for s in [0,1]:
            neach=np.floor(mol.nelec[s]*wts)
            nassigned=np.sum(neach)
            nleft=mol.nelec[s]*wts-neach
            tot=int(np.sum(nleft))
            gets=np.random.choice(len(wts),p=nleft/tot,size=tot,replace=False) 
            for i in gets:
                neach[i]+=1
            for n,coord in zip(neach,mol.atom_coords()):
                for i in range(int(n)):
                    configs[c,count,:]=coord+r*np.random.randn(3)
                    count+=1
    return configs
    

def initial_guess_vectorize(mol,nconfig,r=1.0):
    """ Generate an initial guess by distributing electrons near atoms
    proportional to their charge."""
    nelec=np.sum(mol.nelec)
    epos=np.zeros((nconfig,nelec,3))
    wts=mol.atom_charges()
    wts=wts/np.sum(wts)

    # assign electrons to atoms based on atom charges
    # assign the minimum number first, and assign the leftover ones randomly
    # this algorithm chooses atoms *with replacement* to assign leftover electrons

    for s in [0,1]:
        neach=np.array(np.floor(mol.nelec[s]*wts),dtype=int) # integer number of elec on each atom
        nleft=mol.nelec[s]*wts-neach # fraction of electron unassigned on each atom
        nassigned=np.sum(neach) # number of electrons assigned
        totleft=int(mol.nelec[s]-nassigned) # number of electrons not yet assigned
        bins=np.cumsum(nleft)/totleft
        inds = np.argpartition(np.random.random((nconfig,len(wts))), totleft, axis=1)[:,:totleft]
        ind0=s*mol.nelec[0]
        epos[:,ind0:ind0+nassigned,:] = np.repeat(mol.atom_coords(),neach,axis=0)[np.newaxis] # assign core electrons
        epos[:,ind0+nassigned:ind0+mol.nelec[s],:] = mol.atom_coords()[inds] # assign remaining electrons
    epos+=r*np.random.randn(*epos.shape) # random shifts from atom positions
    return epos


def limdrift(g,cutoff=1):
    tot=np.linalg.norm(g,axis=1)
    mask=tot > cutoff
    g[mask,:]=g[mask,:]/tot[mask,np.newaxis]
    return g
    

def vmc(mol,wf,coords,nsteps=10000,tstep=0.5,accumulators=None,verbose=False):
    if accumulators is None:
        accumulators={'energy':EnergyAccumulator(mol) } 
    nconf=coords.shape[0]
    nelec=np.sum(mol.nelec)
    df=[]
    wf.recompute(coords)
    for step in range(nsteps):
        if verbose:
            print("step",step)
        acc=[]
        for e in range(nelec):

            # Calculate gradient
            grad=limdrift(wf.gradient(e, coords[:,e,:]).T)

            # Calculate new coordinates
            newcoorde=coords[:,e,:]+np.random.normal(scale=np.sqrt(tstep),size=(nconf,3))\
                      + grad*tstep

            # Calculate new gradient
            new_grad=limdrift(wf.gradient(e, newcoorde).T) 

            # PDF for forward transition
            forward=np.sum((coords[:,e,:]+tstep*grad-newcoorde)**2,axis=1)

            # PDF for backward transition
            backward=np.sum((newcoorde+tstep*new_grad-coords[:,e,:])**2,axis=1)

            # Transition probability from distribution
            t_prob = np.exp(1/(2*tstep)*(forward-backward))

            # Compute transition probabilities and which moves to accept
            ratio=np.multiply(wf.testvalue(e,newcoorde)**2, t_prob)
            accept=ratio > np.random.rand(nconf)
            
            # Original MC code
            coords[accept,e,:]=newcoorde[accept,:]
            wf.updateinternals(e,coords[:,e,:],accept)
            acc.append(np.mean(accept))
        avg={}
        for k,accumulator in accumulators.items():
            dat=accumulator(coords,wf)
            for m,res in dat.items():
                avg[k+m]=np.mean(res,axis=0)
        avg['acceptance']=np.mean(acc)
        df.append(avg)
    return df, coords 
    

def test():
    import pandas as pd
    
    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    #mol = gto.M(atom='C 0. 0. 0.', ecp='bfd', basis='bfd_vtz')
    mf = scf.RHF(mol).run()
    import pyscf2qwalk
    pyscf2qwalk.print_qwalk(mol,mf)
    nconf=5000
    wf=PySCFSlaterRHF(nconf,mol,mf)
    coords = initial_guess_vectorize(mol,nconf) 
    def dipole(coords,wf):
        return {'vec':np.sum(coords[:,:,:],axis=1) } 

    import time
    tstart=time.process_time()
    df,coords=vmc(mol,wf,coords,nsteps=100,accumulators={'energy':EnergyAccumulator(mol), 'dipole':dipole } )
    tend=time.process_time()
    print("VMC took",tend-tstart,"seconds")

    df=pd.DataFrame(df)
    df.to_csv("data.csv")
    warmup=30
    print('mean field',mf.energy_tot(),'vmc estimation', np.mean(df['energytotal'][warmup:]),np.std(df['energytotal'][warmup:]))
    print('dipole',np.mean(np.asarray(df['dipolevec'][warmup:]),axis=0))
    
    
def test_compare_init_guess():
    import time
    mol1 = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=2)
    mol2 = gto.M(atom=';'.join(['H 0. 0. {0}'.format(x) for x in np.arange(10)*3]), basis='cc-pvtz',unit='bohr',verbose=2)
    for mol in [mol1, mol2]:
        print(mol.atom)
        mf = scf.RHF(mol).run()
        nconf=5000
        wf=PySCFSlaterRHF(nconf,mol,mf)
        for i,func in enumerate([initial_guess_vectorize, initial_guess]):
            for j in range(5):
                start = time.time()
                configs = func(mol,nconf) 
                assert np.isnan(configs).sum()==0
                assert np.isinf(configs).sum()==0
                logval = wf.recompute(configs)[1]
                print(i, 'min', np.amin(logval), 'max', np.amax(logval), 'median', np.median(logval), 'mean', np.mean(logval))
                hist= np.histogram(logval, range=(-65,-10), bins=14)
                print('hist', hist[0])
                #print('bins', np.round(hist[1],1))
            print(time.time()-start, configs.shape)
    

if __name__=="__main__":
    #test_compare_init_guess();
    import cProfile, pstats, io
    from pstats import Stats
    pr = cProfile.Profile()
    pr.enable()
    test()
    pr.disable()
    p=Stats(pr)
    print(p.sort_stats('cumulative').print_stats())
    
