import numpy as np
from scipy.optimize import minimize
from pyscf import lib, gto, scf
from slater import PySCFSlaterRHF
from multiplywf import MultiplyWF
from jastrow import Jastrow2B
from energy import energy,kinetic
from mc import initial_guess_vectorize,vmc
from func3d import GaussianFunction

def optvariance(mol,wf,coords,params=None,method='Powell',method_options=None):
    """Optimizes variance of wave function against parameters indicated by params.
    Need to implement variance gradient with respect to parameters.
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
    Enref=energy(mol,coords,wf)

    def variance_cost_function(x):
        x_sliced=np.split(x,slices)
        for i,k in enumerate(params):
            wf.parameters[k]=x_sliced[i]
        wf.recompute(coords)
        ke=kinetic(coords,wf)
        #Here we assume the ecp is fixed and only recompute
        #kinetic energy
        En=Enref['total']-Enref['ke']+ke
        #En=energy(mol,coords,wf)['total']
        return np.std(En)**2


    res=minimize(variance_cost_function, x0=x0, method=method, options=method_options)

    # Modfies params dictionary with optimized parameters (preserving original shape)
    opt_pars=np.split(res.x,slices[:-1])
    for i,k in enumerate(params):
        params[k]=opt_pars[i].reshape(shapes[i])
    
    return res.fun





def test_single_opt():
    import pandas as pd
    from multiplywf import MultiplyWF
    from jastrow import Jastrow2B
    
    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=1000
    nsteps=10

    coords = initial_guess_vectorize(mol,nconf)
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
    print('Initial variance:',np.std(En)**2)
    opt_var=optvariance(mol,wf,coords,params0)
    print('Optimized parameters:\n',params0)
    print('Final variance:',opt_var)
    
    return 0



def test_multiple_opt():
    import pandas as pd
    from multiplywf import MultiplyWF
    from jastrow import Jastrow2B
    
    mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=5000
    nsteps=30

    coords = initial_guess_vectorize(mol,nconf)
    basis={'wf2coeff':[GaussianFunction(0.1),GaussianFunction(0.32)]}
    wf=MultiplyWF(nconf,PySCFSlaterRHF(nconf,mol,mf),Jastrow2B(nconf,mol,basis['wf2coeff']))
    params0={'wf2coeff':np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]

    # Cycle of VMC followed by variance optimization
    for j in range(4):
        print('--> Iteration %d <--'%j)
        vmc(mol,wf,coords,nsteps=nsteps)
        En=energy(mol,coords,wf)['total']

        print('Initial parameters:\n',params0)
        print('Initial variance:',np.std(En)**2)
        opt_var=optvariance(mol,wf,coords,params0)
        print('Optimized parameters:\n',params0)
        print('Final variance:',opt_var)
    
    return 0

    
if __name__=="__main__":
    test_single_opt()
    #test_multiple_opt()
