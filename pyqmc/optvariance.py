import numpy as np
from scipy.optimize import minimize
from pyqmc.energy import kinetic

def optvariance(energy,wf,coords,params=None,method='Powell',method_options=None):
    """Optimizes variance of wave function against parameters indicated by params.
    
    Does not use gradient information, and assumes that only the kinetic energy changes.
    
    Args:
      energy: An Accumulator object that returns total energy in 'total' and kinetic energy in 'ke'

      coords: (nconfig,nelec,3)

      params: dictionary with parameters to optimize

      method: An optimization method usable by scipy.optimize
      
    Returns:
      opt_variance, modifying params into optimized values.
      
    """
    if params is None:
        params={}
        for k,p in wf.parameters.items():
            params[k]=p
    #print(params)
    
    # scipy.minimize() needs 1d argument 
    x0=np.concatenate([ params[k].flatten() for k in params ])
    shapes=np.array([ params[k].shape for k in params ])
    slices=np.array([ np.prod(s) for s in shapes ])
    Enref=energy(coords,wf)

    def variance_cost_function(x):
        x_sliced=np.split(x,slices)
        for i,k in enumerate(params):
            wf.parameters[k]=x_sliced[i]
        wf.recompute(coords)
        ke=kinetic(coords,wf)
        #Here we assume the ecp is fixed and only recompute
        #kinetic energy
        En=Enref['total']-Enref['ke']+ke
        return np.std(En)**2


    res=minimize(variance_cost_function, x0=x0, method=method, options=method_options)

    # Modfies params dictionary with optimized parameters (preserving original shape)
    opt_pars=np.split(res.x,slices[:-1])
    for i,k in enumerate(params):
        params[k]=opt_pars[i].reshape(shapes[i])
    
    return res.fun


def test_single_opt():
    from pyqmc.accumulators import EnergyAccumulator
    from pyscf import lib, gto, scf
    
    import pandas as pd
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.jastrow import Jastrow2B
    from pyqmc.func3d import GaussianFunction
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.jastrow import Jastrow2B
    
    from pyqmc.mc import initial_guess,vmc
    
    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='bfd_vtz',ecp='bfd',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=1000
    nsteps=10

    coords = initial_guess(mol,nconf)
    basis={'wf2coeff':[GaussianFunction(1.0),GaussianFunction(2.0)]}
    wf=MultiplyWF(nconf,PySCFSlaterRHF(nconf,mol,mf),Jastrow2B(nconf,mol,basis['wf2coeff']))
    params0={'wf2coeff':np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]
    #params0=None
    
    vmc(wf,coords,nsteps=nsteps,
            accumulators={'energy':EnergyAccumulator(mol)})

    opt_var=optvariance(EnergyAccumulator(mol),wf,coords,params0)
    print('Optimized parameters:\n',params0)
    print('Final variance:',opt_var)
    
    return 0



    
if __name__=="__main__":
    test_single_opt()
