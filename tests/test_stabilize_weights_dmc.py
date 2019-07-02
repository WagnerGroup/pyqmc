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

def use_parsl():
    import parsl
    from pyscf import lib, gto, scf 
    import numpy as np
    import pandas as pd
    import logging

    from parsl.config import Config
    from parsl.providers import LocalProvider
    from parsl.channels import LocalChannel
    from parsl.launchers import SimpleLauncher
    from parsl.executors import ExtremeScaleExecutor
    ncore=4
    config = Config(
        executors=[
            ExtremeScaleExecutor(
                label="Extreme_Local",
                worker_debug=True,
                ranks_per_node=ncore,
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=1,
                    max_blocks=1,
                    launcher=SimpleLauncher()
                )
            )
        ],
        strategy=None,
    )   
    
    parsl.load(config)

def clear_pyscf_io(mol, mf):
    mol.output=None
    mol.stdout=None
    mf.output=None
    mf.stdout=None
    mf.chkfile=None

def timestr(start):
    return 'time={0}'.format(time.time()-start)

def test():
    from pyscf import gto,scf   
    from pyqmc.linemin import line_minimization
    import pyqmc
    import pandas as pd
    import time
    start=time.time()
    use_parsl()
    mol = gto.M(atom='O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587',basis='bfd_vtz',ecp='bfd')     
    mf = scf.RHF(mol).run()
    clear_pyscf_io(mol, mf)

    wf=pyqmc.slater_jastrow(mol,mf)
    acc=pyqmc.gradient_generator(mol,wf,['wf2acoeff','wf2bcoeff'])
    nconf=1000
    configs = pyqmc.initial_guess(mol, nconf)
    #wf,df=pyqmc.gradient_descent(wf,configs,acc,vmcoptions={'nsteps':30}, step=0.2 )
    wf,datagrad,datatest=line_minimization(wf,configs,acc,vmcoptions={'nsteps':20}, warmup=5 )
    print('Optimization finished',timestr(start))
    dfdmc,configs,weights=pyqmc.dmc(wf,configs,accumulators={'energy':pyqmc.EnergyAccumulator(mol)},verbose=True)
    print('DMC finished', timestr(start))

    dfdmc = pd.DataFrame(dfdmc)
    dfdmc.sort_values('step', inplace=True)

    warmup=200
    dfprod=dfdmc[dfdmc.step > warmup]

    reblock=pyblock.reblock(dfprod[['energytotal','energyei']])
    print(reblock[1])
    dfoptimal=reblock[1][reblock[1][('energytotal','optimal block')]!='']
    energy=dfoptimal[('energytotal','mean')].values[0]
    err=dfoptimal[('energytotal','standard error')].values[0]
    print("energy",energy, "+/-",err, timestr(start))


if __name__=='__main__':
    test()
