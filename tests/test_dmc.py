# This must be done BEFORE importing numpy or anything else. 
# Therefore it must be in your main script.
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import sys
import numpy as np
import pyqmc.testwf as testwf
import pytest
import pyblock.pd_utils as pyblock

def test():
    """ Ensure that DMC obtains the exact result for a hydrogen atom """
    from pyscf import lib, gto, scf
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.dmc import limdrift,dmc
    from pyqmc.mc import vmc
    from pyqmc.accumulators import EnergyAccumulator
    from pyqmc.func3d import ExpCuspFunction
    from pyqmc.multiplywf import MultiplyWF
    import pandas as pd
    
    mol = gto.M(atom='H 0. 0. 0.', basis='sto-3g',unit='bohr',spin=1)
    mf = scf.UHF(mol).run()
    nconf=1000
    configs=np.random.randn(nconf,1,3)
    wf1 = PySCFSlaterUHF(mol,mf)
    wf=wf1
    wf2 = JastrowSpin(mol,a_basis=[ExpCuspFunction(5,.2)],b_basis=[])
    wf2.parameters['acoeff']=np.asarray([[-1.0,0]]) 
    wf=MultiplyWF(wf1,wf2)
    
    dfvmc,configs_=vmc(wf,configs,nsteps=50, accumulators={'energy':EnergyAccumulator(mol)})
    dfvmc = pd.DataFrame(dfvmc)
    print('vmc energy', np.mean(dfvmc['energytotal']), np.std(dfvmc['energytotal'])/np.sqrt(len(dfvmc)))

    dfdmc,configs_,weights_=dmc(wf,configs,nsteps=5000,branchtime=5,
        accumulators={'energy':EnergyAccumulator(mol)}, ekey=('energy','total'),
        tstep=0.01,drift_limiter=limdrift,verbose=True)
    
    dfdmc = pd.DataFrame(dfdmc)
    dfdmc.sort_values('step', inplace=True)
    
    warmup=200
    dfprod=dfdmc[dfdmc.step > warmup]
    
    reblock=pyblock.reblock(dfprod[['energytotal','energyei']])
    print(reblock[1])
    dfoptimal=reblock[1][reblock[1][('energytotal','optimal block')]!='']
    energy=dfoptimal[('energytotal','mean')].values[0]
    err=dfoptimal[('energytotal','standard error')].values[0]
    print("energy",energy, "+/-",err) 
    assert np.abs(energy + 0.5)<5*err,"energy not within {0} of -0.5: energy {1}".format(5*err,np.mean(energy)) 


if __name__=='__main__':
    test()
