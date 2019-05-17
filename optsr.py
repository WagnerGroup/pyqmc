import numpy as np
from scipy.optimize import minimize

from slater import PySCFSlaterRHF
from multiplywf import MultiplyWF
from jastrow import Jastrow2B
from energy import energy
from mc import initial_guess,vmc
from func3d import GaussianFunction

def optenergy(mol,wf,coords,params=None,method='Powell',method_options=None,vmcsteps=50):
    """Optimizes energy (not yet using parameter gradients) doing stochastic reconfiguration.
    Args:
      mol: Mole object.
      coords: (nconfig,nelec,3).
      params: dictionary with parameters to optimize.
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
    
    def energy_cost_function(x):
        wf.wf2.parameters['coeff']=x
        wf.recompute(coords)
        vmc(mol,wf,coords,nsteps=vmcsteps) # Stochastic reconfiguration
        En=energy(mol,coords,wf)['total']
        print('E =',np.mean(En),'pars =',x)
        return np.mean(En)

    res=minimize(energy_cost_function, x0=x0, method=method, options=method_options)

    # Modfies params dictionary with optimized parameters (preserving original shape)
    opt_pars=np.split(res.x,slices[:-1])
    for i,k in enumerate(params):
        params[k]=opt_pars[i].reshape(shapes[i])

    return res.fun





def test():
    import pandas as pd
    from pyscf import lib, gto, scf

    mol = gto.M(atom='H 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=300
    nsteps=30

    coords = initial_guess(mol,nconf)
    basis={'wf2coeff':[GaussianFunction(1.0),GaussianFunction(2.0)]}
    wf=MultiplyWF(nconf,PySCFSlaterRHF(nconf,mol,mf),Jastrow2B(nconf,mol,basis['wf2coeff']))
    params0={'wf2coeff':np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]
    #params0=None
    
    vmc(mol,wf,coords,nsteps=nsteps)
    En=energy(mol,coords,wf)['total']

    print('Initial parameters:\n',params0)
    print('Initial energy:',np.mean(En))
    opt_en=optenergy(mol,wf,coords,params0,vmcsteps=nsteps)
    print('Optimized parameters:\n',params0)
    print('Final energy:',opt_en)

    return 0

    
if __name__=="__main__":
    test()
