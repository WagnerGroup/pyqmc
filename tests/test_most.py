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


def test_wfs():
    from pyscf import lib, gto, scf
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.jastrow import Jastrow2B
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.multiplywf import MultiplyWF
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    mf_rohf = scf.ROHF(mol).run()
    mf_uhf = scf.UHF(mol).run()
    epsilon=1e-5
    nconf=10
    epos=np.random.randn(nconf,4,3)
    for wf in [PySCFSlaterRHF(mol,mf),JastrowSpin(mol),Jastrow2B(mol),
               MultiplyWF(PySCFSlaterRHF(mol,mf),JastrowSpin(mol)), 
               MultiplyWF(PySCFSlaterRHF(mol,mf),Jastrow2B(mol)), 
               PySCFSlaterUHF(mol,mf_uhf),PySCFSlaterUHF(mol,mf), 
               PySCFSlaterUHF(mol,mf_rohf)]:
        for k in wf.parameters:
            wf.parameters[k]=np.random.rand(*wf.parameters[k].shape)
        assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon 
        assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon 
        assert testwf.test_wf_pgradient(wf, epos, delta=1e-5)[0] < epsilon
        
        for k,item in testwf.test_updateinternals(wf,epos).items():
            assert item < epsilon


def test_func3d():
    from pyqmc.func3d import PadeFunction,GaussianFunction,test_func3d_gradient,test_func3d_laplacian
    test_functions = {'Pade':PadeFunction(0.2), 'Gaussian':GaussianFunction(0.4)}
    delta=1e-6
    epsilon=1e-5
    
    for name, func in test_functions.items():
        assert test_func3d_gradient(func,delta=delta)[0] < epsilon
        assert test_func3d_laplacian(func,delta=delta)[0] < epsilon
        
    

def test_vmc():
    import pandas as pd
    from pyqmc.mc import vmc,initial_guess
    from pyscf import lib, gto, scf
    
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.accumulators import EnergyAccumulator
    
    
    nconf=5000
    mol   = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=1)

    mf    =scf.RHF(mol).run()
    mf_uhf=scf.UHF(mol).run()
    nsteps=100
    warmup=30

    for wf,mf in [(PySCFSlaterRHF(mol,scf.RHF(mol).run()), scf.RHF(mol).run()) , 
                  (PySCFSlaterUHF(mol,scf.UHF(mol).run()),scf.UHF(mol).run())]:
       
        coords = initial_guess(mol,nconf) 
        df,coords=vmc(wf,coords,nsteps=nsteps,
                accumulators={'energy':EnergyAccumulator(mol)})

        df=pd.DataFrame(df)
        df.to_csv("data.csv")
        en=np.mean(df['energytotal'][warmup:])
        err=np.std(df['energytotal'][warmup:])/np.sqrt(nsteps-warmup)
        assert en-mf.energy_tot() < 10*err

def test_accumulator_rhf():
    import pandas as pd
    from pyqmc.mc import vmc,initial_guess
    from pyscf import gto,scf
    from pyqmc.energy import energy
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.accumulators import EnergyAccumulator,PGradAccumulator

    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=5000
    wf=PySCFSlaterRHF(mol,mf)
    coords = initial_guess(mol,nconf) 

    df,coords=vmc(wf,coords,nsteps=30,accumulators={'energy':EnergyAccumulator(mol)} )
    df=pd.DataFrame(df)
    eaccum=EnergyAccumulator(mol)
    eaccum_energy=eaccum(coords,wf)
    pgrad=PGradAccumulator(eaccum)
    pgrad_dat=pgrad(coords,wf)
    df=pd.DataFrame(df)
    print(df['energytotal'][29] == np.average(eaccum_energy['total']))

    assert df['energytotal'][29] == np.average(eaccum_energy['total'])

def test_ecp():
    import pandas as pd
    from pyqmc.mc import vmc,initial_guess
    from pyscf import lib,gto,scf
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.accumulators import EnergyAccumulator

    mol = gto.M(atom='C 0. 0. 0.', ecp='bfd', basis='bfd_vtz')
    mf = scf.RHF(mol).run()
    nconf=5000
    wf=PySCFSlaterRHF(mol,mf)
    coords = initial_guess(mol,nconf)
    df,coords=vmc(wf,coords,nsteps=100,accumulators={'energy':EnergyAccumulator(mol)} )
    df=pd.DataFrame(df)
    df.to_csv("data.csv")
    warmup=30
    print('mean field',mf.energy_tot(),'vmc estimation', np.mean(df['energytotal'][warmup:]),np.std(df['energytotal'][warmup:]))
    
    assert abs(mf.energy_tot()-np.mean(df['energytotal'][warmup:])) <= np.std(df['energytotal'][warmup:])
