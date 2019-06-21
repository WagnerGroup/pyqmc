import numpy as np
import pandas as pd
from pyqmc.reblock import optimally_reblocked
from pyqmc.bootstrap import bootstrap_derivs
from mc import vmc
from line_minimization import line_minimization

def gradient_lines(wf, configs, pgrad, warmup=10,
        iters=10, vmc=None,vmcoptions=None,shortvmc=20,
        datafile=None,verbose=1):
    """Optimizes energy using line minimizations.

    Args: 
      wf: initial wave function 
      coords: initial configurations 
      pgrad_acc: A PGradAccumulator-like object 
      vmc: A function that works like mc.vmc() 
      vmcoptions: a dictionary of options for the vmc method 
      warmup: warmup cutoff for VMC steps 
      iters: maximum number of steps in the gradient descent 
      datafile: a file in which the current progress can be dumped in JSON format.

    Returns: 
      wf: optimized wave function 
      data: dictionary with gradient descent data

    """
    if vmc is None:
        import pyqmc.mc
        vmc=pyqmc.mc.vmc
    
    if vmcoptions is None:
        vmcoptions={}
        
    def grad_energy_function(params, **vmckwargs):
        wf.parameters.update(pgrad.transform.deserialize(params)) 
        df, configs_ = vmc(wf, configs, {'pgrad':pgrad}, **vmckwargs)
        df = pd.DataFrame(df)[warmup:]

        reblocked_vals = optimally_reblocked(df['pgradtotal'])
        if verbose>1: print(reblocked_vals)
        try:
            val = reblocked_vals['mean']
            err = reblocked_vals['standard error']
        except KeyError:
            val = df['pgradtotal'].mean()
            err = np.inf
        if verbose>0:
            print('vmc acceptance', df['acceptance'].values.mean(), df['acceptance'].values.std())
            print('vmc val', val, 'err', err)

        grad = np.mean( 2*np.real(np.stack(df['pgraddpH'])) -\
                        2*np.real(np.stack(df['pgraddppsi']))*val , axis=0)
        return val, grad

    if verbose>0: 
        import time
        start=time.time()
        print('starting line minimization', time.process_time()-start)
    params = pgrad.transform.serialize_parameters(wf.parameters)
    data={'iter':[], 'params':[],'pgrad':[], 'totalen':[], 'totalen_err':[], 'fitted_en':[]}
    for i in range(iters): 
        params, fitval, (val,grad) = line_minimization(grad_energy_function, params, shortvmc=shortvmc, **vmcoptions)
        data['iter'].append(i)
        data['params'].append(params.copy())
        data['pgrad'].append(grad)
        data['totalen'].append(val)
        data['fitted_en'].append(fitval)
        if verbose>0: 
            print('iteration %i'%i, 'fitval', fitval, 'val', val, '|grad|', np.linalg.norm(grad), 'time',time.time()-start)
            print('='*30)
        if datafile is not None:
            pd.DataFrame(data).to_json(datafile)
    val, grad = grad_energy_function(params, **vmcoptions)
    if verbose>0: print('lin min\n', params, '\ntime', time.time()-start)
    print(data['totalen'])
    
    return wf, data

 

def test():
    """
    Generate a wave function object with some parameters
    Use it to create WFoptimizer
    Test that the energy is minimized
    """

    import copy
    import time
    from pyqmc import PySCFSlaterRHF,JastrowSpin,MultiplyWF, EnergyAccumulator, initial_guess, vmc,ExpCuspFunction, GaussianFunction, optvariance, gradient_descent, PGradTransform, LinearTransform
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
    ncore=2
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
    df, coords = vmc(wf,coords,nsteps=30) #warmup
    energy_acc=EnergyAccumulator(mol)
    pgrad_acc=PGradTransform(energy_acc,LinearTransform(wf.parameters,['wf2acoeff','wf2bcoeff']))
    
    from pyqmc.optsr import gradient_descent
    from pyqmc.parsltools import distvmc
    gradient_lines(wf,coords,pgrad_acc,warmup=4, iters=5,vmc=distvmc,verbose=1,
            vmcoptions={'npartitions':ncore,'nsteps':100,'nsteps_per':100}
            )
    #gradient_descent(wf,coords,pgrad_acc,vmc=distvmc,verbose=1,maxiters=10,
    #        vmcoptions={'npartitions':ncore,'nsteps':100,'nsteps_per':100}
    #        )
            
if __name__=='__main__':
    test()
