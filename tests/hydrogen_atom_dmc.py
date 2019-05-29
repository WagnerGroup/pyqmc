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

def test():
    from pyscf import lib, gto, scf
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.dmc import limdrift,dmc
    from pyqmc.mc import vmc
    from pyqmc.accumulators import EnergyAccumulator
    from pyqmc.func3d import ExpCuspFunction
    from pyqmc.multiplywf import MultiplyWF
    import pandas as pd
    
    mol = gto.M(atom='H 0. 0. 0.', basis='cc-pvtz',unit='bohr',spin=1)
    mf = scf.UHF(mol).run()
    epsilon=1e-3
    nconf=5000
    configs=np.random.randn(nconf,1,3)
    wf1 = PySCFSlaterUHF(mol,mf)
    wf=wf1
    wf2 = JastrowSpin(mol,a_basis=[ExpCuspFunction(5,.2)],b_basis=[])
    wf2.parameters['acoeff']=[[.25,0]]
    wf=MultiplyWF(wf1,wf2)
    
    dfvmc,configs_=vmc(wf,configs,nsteps=1000, accumulators={'energy':EnergyAccumulator(mol)})
    dfvmc = pd.DataFrame(dfvmc)
    print('vmc energy', np.mean(dfvmc['energytotal']), np.std(dfvmc['energytotal'])/len(dfvmc))

    dfdmc,configs_,weights_=dmc(wf,configs,nsteps=1000,branchtime=5,
        accumulators={'energy':EnergyAccumulator(mol)}, ekey=('energy','total'),
        tstep=0.01,drift_limiter=limdrift)
    
    dfdmc = pd.DataFrame(dfdmc)
    dfdmc.sort_values('step', inplace=True)
    
    warmup=0
    energy = dfdmc['energytotal'].values[warmup:]
    print('energy', np.mean(energy), np.std(energy)/len(energy)**.5)
    import matplotlib.pyplot as plt
    plt.plot(dfdmc['energytotal'])
    plt.show()
    assert np.abs(np.mean(energy) + 0.5)<epsilon,"energy not within {0} of -0.5: energy {1}".format(epsilon,np.mean(energy)) 


if __name__=='__main__':
    test()
