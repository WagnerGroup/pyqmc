import numpy as np
import pandas as pd
import json
import itertools
import scipy


def sr_update(pgrad,Sij,step,eps=0.1):
    invSij=np.linalg.inv(Sij+eps*np.eye(Sij.shape[0]))
    v=np.einsum('ij,j->i',invSij,pgrad)
    return -v * step/np.linalg.norm(v)
    
def sd_update(pgrad,Sij,step,eps=0.1):
    return -pgrad*step/np.linalg.norm(pgrad)

def sr12_update(pgrad,Sij,step,eps=0.1):
    invSij=scipy.linalg.sqrtm(np.linalg.inv(Sij+eps*np.eye(Sij.shape[0])))
    v=np.einsum('ij,j->i',invSij,pgrad)
    return -v * step/np.linalg.norm(v)
    



def line_minimization(wf,coords,pgrad_acc,warmup=0,
        steprange=0.5,maxiters=50,
        vmc=None,vmcoptions=None,
        dataprefix="",
        update=sr_update,
        update_kws=None,
        verbose=2):
    """Optimizes energy using gradient descent with stochastic reconfiguration.

    Args:

      wf: initial wave function

      coords: initial configurations

      pgrad_acc: A PGradAccumulator-like object

      steprange: How far to search in the line minimization

      vmc: A function that works like mc.vmc()

      vmcoptions: a dictionary of options for the vmc method

      update: A function that generates a parameter change 

      update_kws: Any keywords 

      maxiters: maximum number of steps in the gradient descent

      dataprefix: A base filename in which to save datafileline.json and datafilegrad.json, which contain information about the optimization

    Returns:

      wf: optimized wave function

      datagrad: dictionary with gradient descent data

      dataline: dictionary with line minimization data

    """
    if vmc is None:
        import pyqmc.mc
        vmc=pyqmc.mc.vmc
    
    if vmcoptions is None:
        vmcoptions={}
    if update_kws is None:
        update_kws={}
        

    def gradient_energy_function(x):
        newparms=pgrad_acc.transform.deserialize(x)
        for k in newparms:
            wf.parameters[k]=newparms[k]
        data,newcoords=vmc(wf,coords,accumulators={'pgrad':pgrad_acc}, **vmcoptions)
        df=pd.DataFrame(data)[warmup:]
        nsteps=len(df)
        en=np.mean(df['pgradtotal'])
        
        en_std=np.std(df['pgradtotal'])
        
        # Sij matrix with stabilizing diagonal
        dpH=np.mean(df['pgraddpH'],axis=0)
        dp=np.mean(df['pgraddppsi'],axis=0)
        dpdp=np.mean(df['pgraddpidpj'],axis=0)
        grad=2*(dpH-en*dp)
        Sij = dpdp - np.einsum('i,j->ij',dp,dp) #+ eps*np.eye(dpdp.shape[0])
        grad_std=0
        return grad, Sij, en, en_std, len(df)


        
    x0=pgrad_acc.transform.serialize_parameters(wf.parameters)
    datagrad=[]
    datatest=[]
    
    # Gradient descent cycles
    for it in range(maxiters):
        pgrad,Sij,en,en_std,nsteps=gradient_energy_function(x0)
        datagrad.append({'pgrad':pgrad,
            'S':Sij,
            'en':en,
            'en_err':en_std/np.sqrt(nsteps),
            'iter':it,
            'params':x0.copy()
            })

        print("descent en",en,en_std/np.sqrt(nsteps))
        print("descent grad",pgrad,flush=True)

        
        xfit=[]
        yfit=[]
        xfit.append(0.0)
        yfit.append(np.linalg.norm(pgrad)**2)
        npts=8
        steps=np.linspace(0,steprange,npts)
        steps[0]=-steprange/npts


        for step in steps:
            x = x0+update(pgrad,Sij,step,**update_kws)
            pgradp,Sijp,enp,en_stdp,nstepsp=gradient_energy_function(x)
            en_stdp/=np.sqrt(nstepsp)

            print("descent step",step,enp,en_stdp,flush=True)
            xfit.append(step)
            yfit.append(np.linalg.norm(pgradp)**2)
            datatest.append({
                'en':enp,
                'en_err':en_stdp,
                'iter':it,
                'step':step,
                'eps':0.0,
                'params':x.copy(),
                'pgrad':pgradp
                })
                
        p=np.polyfit(xfit,yfit,2)
        print("polynomial fit",p)
        est_min=-p[1]/(2*p[0])
        print("estimated minimum",est_min,flush=True)
        if est_min > step:
            est_min=step
        if est_min < 0:
            est_min=0.0
        x0+=update(pgrad,Sij,est_min,**update_kws)


            

        pd.DataFrame(datagrad).to_json(dataprefix+"grad.json")
        pd.DataFrame(datatest).to_json(dataprefix+"line.json")


    return wf, datagrad, datatest



