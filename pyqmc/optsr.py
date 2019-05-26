import numpy as np
import pandas as pd

def gradient_descent(wf,coords,params=None,warmup=10,accumulators=None,
        step=0.5,eps=0.1,maxiters=50,
        vmc=None,vmcoptions=None,
        datafile=None):
    """Optimizes energy using gradient descent with stochastic reconfiguration.

    Args:

      wf: initial wave function

      coords: initial configurations

      params: list of dictionary entries in wf.parameters to optimize

      vmc: A function that works like mc.vmc()

      vmcoptions: a dictionary of options for the vmc method

      warmup: warmup cutoff for VMC steps

      accumulators: accumulators to store during VMC

      step: gradient descent step size

      eps: stabilizing constant for the stochastic reconfiguration matrix

      maxiters: maximum number of steps in the gradient descent

      datafile: a file in which the current progress can be dumped in JSON format.

    Returns:

      wf: optimized wave function

      data: dictionary with gradient descent data

    """
    import pandas as pd
    if vmc is None:
        import pyqmc.mc
        vmc=pyqmc.mc.vmc
    
    if vmcoptions is None:
        vmcoptions={}
    if params is None:
        params=list(wf.parameters.keys())
        print('Error: Parameter derivatives for Slater coefficients not implemented.')
        exit()
        
    # Gradient takes 1d argument of parameters 
    x0=np.concatenate([ wf.parameters[k].flatten() for k in params ])
    shapes=np.array([ wf.parameters[k].shape for k in params ])
    slices=np.array([ np.prod(s) for s in shapes ])


    def gradient_energy_function(x):
        x_sliced=np.split(x,slices[:-1])
        for i,k in enumerate(params):
            wf.parameters[k]=x_sliced[i].reshape(shapes[i])    
        wf.recompute(coords)
        data,confs=vmc(wf,coords,accumulators=accumulators, **vmcoptions)
        df=pd.DataFrame(data)[warmup:]
        
        en=np.mean(df['energytotal'])
        en_std=np.std(df['energytotal'])
        
        grad={}
        grad_std={}
        dp={}
        dpdp={}
        for k in params:
            if 'pgraddpH_'+k in df.keys():
                dpH = np.mean(df['pgraddpH_'+k],axis=0)
                dpH_std = np.std(df['pgraddpH_'+k].values,axis=0)
                dp[k] = np.mean(df['pgraddppsi_'+k],axis=0)
                dp_std = np.std(df['pgraddppsi_'+k].values,axis=0)
                grad[k] = 2*( dpH - en * dp[k])
                grad_std[k] = 2*np.sqrt( dpH_std**2 + (en*dp[k])**2 * ( (en_std/en)**2 + (dp_std/dp[k])**2 ) )
                for k2 in params:
                    if 'pgraddpH_'+k2 in df.keys():
                        dpdp[k+k2] = np.mean(df['pgraddpidpj_'+k+k2],axis=0)
        # Concatenates types of gradients
        grad=np.concatenate([ grad[k].reshape(-1) for k in params if 'pgraddpH_'+k in df.keys() ])
        grad_std=np.concatenate([ grad_std[k].reshape(-1) for k in params if 'pgraddpH_'+k in df.keys() ])
        dp=np.concatenate([ dp[k] for k in params if 'pgraddpH_'+k in df.keys() ])
        # Concatenates Sij sub-matrices into one matrix
        dpdp=np.array([ [ dpdp[k+k2] for k2 in params if 'pgraddpH_'+k2 in df.keys() ] for k in params if 'pgraddpH_'+k in df.keys() ])
        dpdp=np.concatenate( np.array([ np.concatenate(d,axis=1) for d in dpdp ]) , axis=0)
        # Sij matrix with stabilizing diagonal
        Sij = dpdp - np.einsum('i,j->ij',dp,dp) + eps*np.eye(dpdp.shape[0])
        invSij=np.linalg.inv(Sij)
        
        return grad, grad_std, invSij, en, en_std


    data={'iter':[],'params':[],'pgrad':[],'pgrad_err':[],'totalen':[],'totalen_err':[]}
    pgrad,pgrad_std,invSij,en,en_std=gradient_energy_function(x0)
    data['iter'].append(0)
    data['params'].append(x0)
    data['pgrad'].append(pgrad)
    data['pgrad_err'].append(pgrad_std)
    data['totalen'].append(en)
    data['totalen_err'].append(en_std)
    print('p =',x0,'grad =',pgrad,'|grad|=%.6f'%np.linalg.norm(pgrad),'E=%.5f+-%.5f'%(en,en_std))
    
    # Gradient descent cycles
    for it in range(maxiters):
        x0 -= np.einsum('ij,j->i',invSij,pgrad) * step/(it/10+1)
        pgrad,pgrad_std,invSij,en,en_std = gradient_energy_function(x0)
        print('i=%d: p ='%it,x0,'grad =',pgrad,'|grad|=%.6f'%np.linalg.norm(pgrad),'E=%.5f+-%.5f'%(en,en_std))
        data['iter'].append(it+1)
        data['params'].append(x0)
        data['pgrad'].append(pgrad)
        data['pgrad_err'].append(pgrad_std)
        data['totalen'].append(en)
        data['totalen_err'].append(en_std)
        if not (datafile is None):
            pd.DataFrame(data).to_json(datafile)


    print('\nGradient descent terminated.')
    print('p =',x0,'grad =',pgrad,'|grad|=%.6f'%np.linalg.norm(pgrad),'E=%.5f+-%.5f'%(en,en_std))
            
    return wf, data






    

def test():    
    from pyscf import lib, gto, scf
    from pyqmc.accumulators import EnergyAccumulator, PGradAccumulator    
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.jastrow import Jastrow2B
    from pyqmc.func3d import GaussianFunction
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.mc import initial_guess
    
    mol = gto.M(atom='H 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr',verbose=5)
    mf = scf.RHF(mol).run()
    nconf=2500
    nsteps=70
    warmup=20

    coords = initial_guess(mol,nconf)
    basis={'wf2coeff':[GaussianFunction(0.2),GaussianFunction(0.4),GaussianFunction(0.6)]}
    wf=MultiplyWF(PySCFSlaterRHF(mol,mf),Jastrow2B(mol,basis['wf2coeff']))
    params0={'wf2coeff':np.array([-0.8,-0.2,0.4])}
    for k,p in wf.parameters.items():
        if k in params0:
            wf.parameters[k]=params0[k]
    
    energy_acc=EnergyAccumulator(mol)
    pgrad_acc=PGradAccumulator(energy_acc)
    
    # Gradient descent
    wf,data=gradient_descent(wf,coords,params=list(params0.keys()),
            vmcoptions={'nsteps':nsteps},warmup=warmup,
            accumulators={'energy':energy_acc,'pgrad':pgrad_acc},
            step=0.5,eps=0.1,maxiters=50,datafile='sropt.json')

    # GD data plot
    import pandas as pd
    import matplotlib.pyplot as plt
    df=pd.DataFrame(data)
    df2=pd.DataFrame(df['pgrad'].values.tolist())
    df3=pd.DataFrame(df['pgrad_err'].values.tolist())
    df2['iter']=df['iter']
    fig, ax = plt.subplots(1,2)
    for c in df2.keys():
        if c!='iter':
            ax[0].errorbar(df2['iter'],df2[c],yerr=df3[c],label=c)
    ax[0].set_xlabel('Gradient descent step')
    ax[0].set_ylabel('PGradient (Ha)')
    ax[1].errorbar(df['iter'],df['totalen'],yerr=df['totalen_err'])
    ax[1].set_xlabel('Gradient descent step')
    ax[1].set_ylabel('Total Energy (Ha)')
    ax[0].legend(title='PGrad')
    plt.tight_layout()
    plt.savefig("gradesc_Enmin.png")
    df.to_csv("gradesc_Enmin.csv")
       
    


    
if __name__=="__main__":
    test()

