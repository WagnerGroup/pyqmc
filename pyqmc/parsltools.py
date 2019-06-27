# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import parsl
from parsl.app.app import python_app

import numpy as np
import time

@python_app
def vmcparsl(wf,lastrun,nsteps,accumulators,stepoffset=0):
    import os
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    from pyqmc.mc import vmc
    import copy
    import numpy as np
    
    print("running")
    df,coords=vmc(copy.copy(wf),np.asarray(lastrun[1]).copy(),
                 nsteps=nsteps,
                 accumulators=copy.copy(accumulators),
                 stepoffset=stepoffset)
    return df,coords.tolist()


        
def distvmc(wf,coords,accumulators=None,nsteps=100,npartitions=2,nsteps_per=20):
    """ 
    Args: 
    wf: a wave function object

    coords: nconf x nelec x 3 

    nsteps: how many steps to move each walker


    """
    
    if accumulators is None:
        accumulators={}
        if verbose:
            print("WARNING: running VMC with no accumulators")
            
    allruns=[]
    niterations=int(nsteps/nsteps_per)
    coord=np.split(coords,npartitions)
    for epoch in range(niterations):
        for p in range(npartitions):
            if epoch==0:
                allruns.append(vmcparsl(wf,([],coord[p]),nsteps_per,accumulators))
            else:
                allruns.append(vmcparsl(wf,allruns[-npartitions],
                               nsteps_per,
                               accumulators,
                               stepoffset=epoch*nsteps_per))
    import pandas as pd
    import time
    while True:
        print("Job done:",[r.done() for r in  allruns])
        df=[]
        done=[]
        for r in allruns:
            if r.done():
                df.extend(r.result()[0])
        pd.DataFrame(df).to_json("data.json")
        sys.stdout.flush()
        if np.all([r.done() for r in  allruns]):
            break
        time.sleep(10)

    coords=np.asarray(np.concatenate([x.result()[1] for x in allruns[-npartitions:]]))
    
    return df,coords

        
                


def test():
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

    mol=gto.M(atom='H 0. 0. 0.; H 0. 0. 2.0',unit='bohr',
                ecp='bfd', basis='bfd_vtz')
    mf = scf.RHF(mol).run()
    mol.output=None
    mol.stdout=None
    mf.output=None
    mf.stdout=None
    mf.chkfile=None
    from pyqmc import ExpCuspFunction,GaussianFunction,MultiplyWF,PySCFSlaterRHF,JastrowSpin,initial_guess,EnergyAccumulator
    from pyqmc.accumulators import PGradTransform,LinearTransform
    
    nconf=1600
    basis=[ExpCuspFunction(2.0,1.5),GaussianFunction(0.5),GaussianFunction(2.0),GaussianFunction(.25),GaussianFunction(1.0),GaussianFunction(4.0),GaussianFunction(8.0)  ]
    wf=MultiplyWF(PySCFSlaterRHF(mol,mf),JastrowSpin(mol,basis,basis))
    coords = initial_guess(mol,nconf)
    energy_acc=EnergyAccumulator(mol)
    pgrad_acc=PGradTransform(energy_acc,LinearTransform(wf.parameters,['wf2acoeff','wf2bcoeff']))
    
    from pyqmc.optsr import gradient_descent
    gradient_descent(wf,coords,pgrad_acc,vmc=distvmc,
            vmcoptions={'npartitions':ncore,'nsteps':100,'nsteps_per':100}
            )
            

if __name__=="__main__":
    test()
    
                
                

            
