import numpy as np
from numpy import linalg as la
from pyscf import lib, gto, scf
from pyqmc.slater import PySCFSlaterRHF
from pyqmc.energy import energy


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
            gets=np.random.choice(len(wts),p=nleft,size=tot,replace=False) 
            for i in gets:
                neach[i]+=1
            for n,coord in zip(neach,mol.atom_coords()):
                for i in range(int(n)):
                    configs[c,count,:]=coord+r*np.random.randn(3)
                    count+=1
    return configs
    

def vmc(mol,wf,coords,nsteps=10000,tstep=0.5,accumulators=None):
    if accumulators is None:
        accumulators={'energy':energy } 
    nconf=coords.shape[0]
    nelec=np.sum(mol.nelec)
    df=[]
    wf.recompute(coords)
    for step in range(nsteps):
        print("step",step)
        acc=[]
        for e in range(nelec):

            # Create current value of wavefunction
            current_val=np.exp(wf.value()[0][:,np.newaxis]) * \
                        np.exp(wf.value()[1][:,np.newaxis])

            # Calculate gradient
            grad=wf.gradient(e, coords[:,e,:]).T * current_val

            # Calculate new coordinates
            newcoorde=coords[:,e,:]+np.random.normal(scale=np.sqrt(tstep),size=(nconf,3))\
                      - grad*tstep

            # Calculate new gradient
            new_grad=wf.gradient(e, newcoorde).T * wf.testvalue(e, newcoorde)[:,np.newaxis]\
                     * current_val

            # Numerator in transition probability
            first=np.linalg.norm((coords[:,e,:] + tstep*grad - newcoorde), 2, axis=1)**2

            # Denominator in transition probability
            second=np.linalg.norm((newcoorde + tstep*new_grad - coords[:,e,:]), 2, axis=1)**2

            # Transition probability from distribution
            t_prob = np.exp(1/(2*tstep**2)*(first-second))

            # Original MC code
            ratio=np.multiply(wf.testvalue(e,newcoorde)**2, t_prob)
            accept=ratio > np.random.rand(nconf)
            coords[accept,e,:]=newcoorde[accept,:]
            wf.updateinternals(e,coords[:,e,:],accept)
            acc.append(np.mean(accept))
        avg={}
        for k,accumulator in accumulators.items():
            dat=accumulator(mol,coords,wf)
            for m,res in dat.items():
                avg[k+m]=np.mean(res,axis=0)
        avg['acceptance']=np.mean(acc)
        df.append(avg)
    return df #should return back new coordinates
    

def old_vmc(mol,wf,coords,nsteps=10000,tstep=0.5,accumulators=None):
    if accumulators is None:
        accumulators={'energy':energy } 
    nconf=coords.shape[0]
    nelec=np.sum(mol.nelec)
    df=[]
    wf.recompute(coords)
    for step in range(nsteps):
        print("step",step)
        acc=[]
        #if(step==1000):
        #    break
        for e in range(nelec):
            newcoorde=coords[:,e,:]+np.random.normal(scale=tstep,size=(nconf,3))
            ratio=wf.testvalue(e,newcoorde)
            accept=ratio**2 > np.random.rand(nconf)
            coords[accept,e,:]=newcoorde[accept,:]
            wf.updateinternals(e,coords[:,e,:],accept)
            acc.append(np.mean(accept))
        avg={}
        for k,accumulator in accumulators.items():
            dat=accumulator(mol,coords,wf)
            for m,res in dat.items():
                avg[k+m]=np.mean(res,axis=0)
        avg['acceptance']=np.mean(acc)
        df.append(avg)
    return df #should return back new coordinates



def test(tstep):
    import pandas as pd
    
    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=0)
    old_mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=0)
    mf = scf.RHF(mol).run()
    old_mf = scf.RHF(mol).run()
    nconf=300

    wf=PySCFSlaterRHF(nconf,mol,mf)
    old_wf=PySCFSlaterRHF(nconf,old_mol,old_mf)

    coords = initial_guess(mol,nconf) 
    old_coords = initial_guess(old_mol,nconf)
    def dipole(mol,coords,wf):
        return {'vec':np.sum(coords[:,:,:],axis=1) } 
    df=vmc(mol,wf,coords,nsteps=10000,tstep=tstep,
           accumulators={'energy':energy, 'dipole':dipole } )

    df=pd.DataFrame(df)
    df.to_csv("../test_pyqmc/long_test/data"+str(tstep)+".csv")

    print()
    old_df=old_vmc(old_mol,old_wf,old_coords,nsteps=10000,tstep=tstep,
               accumulators={'energy':energy, 'dipole':dipole } )
    old_df=pd.DataFrame(old_df)
    old_df.to_csv("../test_pyqmc/long_test/old_data"+str(tstep)+".csv")

    warmup=100
    #print('mean field',mf.energy_tot(),'vmc estimation', np.mean(df['energytotal'][warmup:]),np.std(df['energytotal'][warmup:]))
    #print('dipole',np.mean(np.asarray(df['dipolevec'][warmup:]),axis=0))
    
if __name__=="__main__":
    import cProfile, pstats, io
    from pstats import Stats
    pr = cProfile.Profile()
    pr.enable()
    test(0.5)
    pr.disable()
    p=Stats(pr)
    #print(p.sort_stats('cumulative').print_stats())
    
