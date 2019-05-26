import numpy as np
from scipy.optimize import minimize
from energy import kinetic
from mc import vmc

def optenergy_bare(mol,energy,wf,coords,params=None,method='Powell',method_options=None,vmcsteps=50,warmup=15):
    """Optimizes energy (not yet using parameter gradients) against parameters indicated by params.
    
    It doesn't yet use parameter gradients.

    Args:
      mol: 

      energy: An Accumulator object that returns total energy in 'total' and kinetic energy in 'ke'

      coords: (nconfig,nelec,3)

      params: dictionary with parameters to optimize

      method: An optimization method usable by scipy.optimize

      vmcsteps: steps for the vmc
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

    import pandas as pd
    
    def energy_cost_function(x):
        wf.wf2.parameters['coeff']=x
        wf.recompute(coords)
        data,coords2=vmc(wf,coords,nsteps=vmcsteps)
        df=pd.DataFrame(data)['energytotal']
        #En=energy(coords,wf)['total']
        #print('E =',np.mean(En),'pars =',x)
        print('x --->',x,'E =',np.mean(df[warmup:]),'+-',np.std(df[warmup:]))
        return np.mean(df[warmup:])

    res=minimize(energy_cost_function, x0=x0, method=method, options=method_options)

    # Modfies params dictionary with optimized parameters (preserving original shape)
    if len(slices)>1:
        opt_pars=np.split(res.x,slices[:-1])
    else:
        opt_pars=[res.x]
    for i,k in enumerate(params):
        params[k]=opt_pars[i].reshape(shapes[i])

    return res.fun, res.success




def optenergy(mol,energy,pgrad,wf,coords,params=None,method='CG',method_options=None,vmcsteps=50,warmup=10,accumulators=None):
    """Optimizes energy (not yet using parameter gradients) against parameters indicated by params.
    
    It doesn't yet use parameter gradients.

    Args:
      mol: 

      energy: An Accumulator object that returns total energy in 'total' and kinetic energy in 'ke'

      coords: (nconfig,nelec,3)

      params: dictionary with parameters to optimize

      method: An optimization method usable by scipy.optimize

      vmcsteps: steps for the vmc
    Returns:
      opt_variance, modifying params into optimized values.

    """
    import pandas as pd
    
    if params is None:
        params={}
        for k,p in wf.parameters.items():
            params[k]=p
    #print(params)
    
    # scipy.minimize() needs 1d argument 
    x0=np.concatenate([ params[k].flatten() for k in params ])
    shapes=np.array([ params[k].shape for k in params ])
    slices=np.array([ np.prod(s) for s in shapes ])
    
    def energy_function(x):
        wf.wf2.parameters['coeff']=x
        wf.recompute(coords)
        data,coords2=vmc(wf,coords,nsteps=vmcsteps,accumulators=accumulators)
        df=pd.DataFrame(data)
        return np.mean(df['energytotal'][warmup:])#, np.std(df[warmup:])

    def gradient_energy_function(x):
        wf.wf2.parameters['coeff']=x
        wf.recompute(coords)
        data,confs=vmc(wf,coords,nsteps=vmcsteps,accumulators=accumulators)
        df=pd.DataFrame(data)
        grad = 2*(np.mean(df['pgraddpH_wf2coeff'][warmup:],axis=0) - np.mean(df['energytotal'][warmup:]) * np.mean(df['pgraddppsi_wf2coeff'][warmup:],axis=0))
        return grad
        

    gtol=1e-4
    step=1.0
    xx=x0
    pgrad=gradient_energy_function(xx)
    print(xx,pgrad,np.linalg.norm(pgrad))
    while np.linalg.norm(pgrad) > gtol:
        xx-=step*pgrad
        pgrad=gradient_energy_function(xx)
        print(xx,np.linalg.norm(pgrad))
    print('Gradient descent converged:')
    print('x =',xx,'grad(x) =',pgrad,'norm(grad(x)) =',np.linalg.norm(pgrad))
    
    #print('\n1D function:')
    #for x in np.linspace(-0.5,-0.3,30):
    #    #en,enstd,pgen=gradient_energy_function(np.array([x]))
    #    pgen=gradient_energy_function(np.array([x]))
    #    print('x=%.3f: pgraden ='%x,pgen)
    #    #print('i=%d:'%i)
    
    res=minimize(energy_function, x0=x0, method=method, jac=gradient_energy_function, options={'disp':True, 'gtol':1e-4})
    print(res.x,res.fun)
    
    # Modfies params dictionary with optimized parameters (preserving original shape)
    print(res.x)
    opt_pars=np.split(res.x,slices[:-1])
    for i,k in enumerate(params):
        params[k]=opt_pars[i].reshape(shapes[i])
    print('opt_pars:',opt_pars,'Sucess:',res.success)
    
    return res.fun, res.success





def test():
    from accumulators import EnergyAccumulator, PGradAccumulator
    from pyscf import lib, gto, scf
    
    import pandas as pd
    from multiplywf import MultiplyWF
    from jastrow import Jastrow2B
    from func3d import GaussianFunction
    from slater import PySCFSlaterRHF
    from multiplywf import MultiplyWF
    from jastrow import Jastrow2B

    from mc import initial_guess
    
    mol = gto.M(atom='H 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=1500
    nsteps=70

    coords = initial_guess(mol,nconf)
    basis={'wf2coeff':[GaussianFunction(0.2),GaussianFunction(0.4)]}
    wf=MultiplyWF(nconf,PySCFSlaterRHF(nconf,mol,mf),Jastrow2B(nconf,mol,basis['wf2coeff']))
    params0={'wf2coeff':np.array([0.2,0.5])}#np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]
    #params0=None

    
    from time import time

    t0=time()
    energy_acc=EnergyAccumulator(mol)
    pgrad_acc=PGradAccumulator(energy_acc)
    vmc(wf,coords,nsteps=nsteps,accumulators={'energy':energy_acc,'pgrad':pgrad_acc})

    #exit()
    #print('Initial parameters:\n',params0)
    ##print('Initial energy:',np.mean(En))
    #opt_en,success=optenergy_bare(mol,EnergyAccumulator(mol),wf,coords,params0,vmcsteps=nsteps)
    #print('Optimized parameters (no gradient):\n',params0)
    #print('Final energy:',opt_en,'Success:',success)    

    t1=time()

    res={'cycle':[],'converged':[],'opt_en':[],'par1':[]}#,'par2':[]}
    for j in range(1):
        params0={'wf2coeff':np.array([0.52,-1.1])}#np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
        for k,p in wf.parameters.items():
            if k in params0:
                wf.parameters[k]=params0[k]
        #energy_acc=EnergyAccumulator(mol)
        #pgrad_acc=PGradAccumulator(energy_acc)
        vmc(wf,coords,nsteps=nsteps,accumulators={'energy':energy_acc,'pgrad':pgrad_acc})
        print('Initial parameters:\n',params0)
        #print('Initial energy:',np.mean(En))
        opt_en,success=optenergy(mol,energy_acc,pgrad_acc,wf,coords,params0,method='CG',method_options={'gtol':1e-6,'disp':True,'eps':1e-2},vmcsteps=nsteps,warmup=20,accumulators={'energy':energy_acc,'pgrad':pgrad_acc})
        #opt_en,success=optenergy(mol,EnergyAccumulator(mol),wf,coords,params0,method='Powell',method_options={'gtol':1e-4,'disp':True,'eps':1e-2},vmcsteps=nsteps)
        print('Optimized parameters (with grad):\n',params0)
        print('Final energy:',opt_en)

        res['cycle'].append(j)
        res['converged'].append(success)
        res['opt_en'].append(opt_en)
        res['par1'].append(params0['wf2coeff'][0])
        #res['par2'].append(params0['wf2coeff'][1])

    import pandas as pd
    df=pd.DataFrame(res)
    df.to_csv("optsr_data.csv")
    
    t2=time()

    print('time(no grad) =',t1-t0)
    print('time(with grad) =',t2-t1)
    
    return 0



def test_bare():
    from accumulators import EnergyAccumulator
    from pyscf import lib, gto, scf
    
    import pandas as pd
    from multiplywf import MultiplyWF
    from jastrow import Jastrow2B
    from func3d import GaussianFunction
    from slater import PySCFSlaterRHF
    from multiplywf import MultiplyWF
    from jastrow import Jastrow2B

    from mc import initial_guess
    
    mol = gto.M(atom='H 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=5000
    nsteps=100

    coords = initial_guess(mol,nconf)
    basis={'wf2coeff':[GaussianFunction(0.2)]}
    wf=MultiplyWF(nconf,PySCFSlaterRHF(nconf,mol,mf),Jastrow2B(nconf,mol,basis['wf2coeff']))
    params0={'wf2coeff':np.random.normal(loc=0.,scale=.1,size=len(wf.parameters['wf2coeff']))}
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]
    #params0=None
    
    vmc(wf,coords,nsteps=nsteps)
    
    print('Initial parameters:\n',params0)
    enacc=EnergyAccumulator(mol)
    opt_en=optenergy_bare(mol,enacc,wf,coords,params0,vmcsteps=nsteps)
    print('Optimized parameters:\n',params0)
    print('Final energy:',opt_en)

    for x in np.linspace(-2,2,50):
        wf.wf2.parameters['coeff']=np.array([x])
        wf.recompute(coords)
        vmc(wf,coords,nsteps=nsteps)
        En=enacc(coords,wf)['total']
        print(x,np.mean(En))
        

    return 0


    
if __name__=="__main__":
    test()
    #test_bare()
