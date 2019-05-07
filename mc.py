import numpy as np
from pyscf import lib, gto, scf
from slater import PySCFSlaterRHF
from energy import energy

def vmc(mol,wf,coords,nsteps=10000,tstep=0.5):
    nconf=coords.shape[0]
    nelec=np.sum(mol.nelec)
    
    df=[]
    wf.recompute(coords)
    for step in range(nsteps):
        print("step",step)
        acc=[]
        for e in range(nelec):
            newcoorde=coords[:,e,:]+np.random.normal(scale=tstep,size=(nconf,3))
            ratio=wf.testvalue(e,newcoorde)
            accept=ratio**2 > np.random.rand(nconf)
            coords[accept,e,:]=newcoorde[accept,:]
            wf.updateinternals(e,coords[:,e,:],accept)
            acc.append(np.mean(accept))
        dat=energy(mol,coords,wf)
        avg={}
        for k in dat:
            avg[k]=np.mean(dat[k])
        avg['acceptance']=np.mean(acc)
        df.append(avg)
        #print(df[-1])
    return df #should return back new coordinates
    

def test():
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=5000
    nelec=np.sum(mol.nelec)
    wf=PySCFSlaterRHF(nconf,mol,mf)
    coords = np.random.normal(scale=1.,size=(nconf,nelec,3))

    df=vmc(mol,wf,coords,nsteps=100)


    import pandas as pd
    df=pd.DataFrame(df)
    df.to_csv("data.csv")
    warmup=30
    print(np.mean(df['total_energy'][warmup:]),np.std(df['total_energy'][warmup:]))
    
if __name__=="__main__":
    test()
