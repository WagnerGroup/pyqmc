import numpy as np

def test_updateinternals(wf,configs):
    """
    Parameters:
    wf: a wave function object to be tested
    configs: nconf x nelec x 3 position array

    Returns: 
    tuple which 

    """

    ne=configs.shape[1]
    delta=1e-2
    updatevstest=np.zeros((ne,configs.shape[0]))
    recomputevstest=np.zeros((ne,configs.shape[0]))
    recomputevsupdate=np.zeros((ne,configs.shape[0]))
    for e in range(ne):
        val1=wf.recompute(configs)
        configs[:,e,:]+=delta
        ratio=wf.testvalue(e,configs[:,e,:])
        wf.updateinternals(e,configs[:,e,:])
        update=wf.value()
        recompute=wf.recompute(configs)
        updatevstest[e,:]=update[0]/val1[0]*np.exp(update[1]-val1[1])-ratio
        recomputevsupdate[e,:]=update[0]/val1[0]*np.exp(update[1]-val1[1])\
                               -recompute[0]/val1[0]*np.exp(recompute[1]-val1[1])
        recomputevstest[e,:]=recompute[0]/val1[0]*np.exp(recompute[1]-val1[1])-ratio

        
    return {'updatevstest':np.max(updatevstest),
            'recomputevstest':np.max(recomputevstest),
            'recomputevsupdate':np.max(recomputevsupdate)} 


def test_wf_gradient(wf, configs, delta=1e-5):
    """ 
    Parameters:
        wf: a wavefunction object with functions wf.recompute(configs), wf.testvalue(e,configs) and wf.gradient(e,configs)
        configs: nconf x nelec x 3 position array to set the wf object
        delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision
    Tests wf.gradient(e,configs) against numerical derivatives of wf.testvalue(e,configs)
    For gradient and testvalue:
        e is the electron index
        epos is nconf x 3 positions of electron e
    wf.testvalue(e,epos) should return a ratio: the wf value at the position where electron e is moved to epos divided by the current value
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    
    """
    nconf, nelec = configs.shape[0:2]
    wf.recompute(configs)
    maxerror=0
    grad = np.zeros(configs.shape)
    numeric = np.zeros(configs.shape)
    for e in range(nelec):
        grad[:,e,:] = wf.gradient(e, configs[:,e,:]).T
        for d in range(0,3):
            configsnew=configs.copy()
            configsnew[:,e,d]+=delta
            plusval=wf.testvalue(e,configsnew[:,e,:])
            configsnew[:,e,d]-=2*delta
            minuval=wf.testvalue(e,configsnew[:,e,:])
            numeric[:,e,d] = (plusval - minuval)/(2*delta)
    maxerror = np.amax(np.abs(grad-numeric))
    normerror = np.mean(np.abs(grad-numeric))
    
    #print('maxerror', maxerror, np.log10(maxerror))
    #print('normerror', normerror, np.log10(normerror))
    return(maxerror,normerror)



def test_wf_pgradient(wf,configs,delta=1e-5):
    pkeys=wf.parameters.keys()
    baseval=wf.recompute(configs)
    gradient=wf.pgradient()
    error={}
    #This is a little tricky; you cannot assign wf.parameters[k] to a numpy array
    #because it breaks multiplywf (since wf.parameters are a reference to self.wf1.parameters
    #and self.wf2.parameters, resetting the reference breaks it.)
    #
    for k in gradient.keys(): #We only check the gradients that are exposed.
        flt=wf.parameters[k].ravel()
        #print(flt.shape,wf.parameters[k].shape,gradient[k].shape)
        nparms=len(flt)
        numgrad=np.zeros((configs.shape[0],nparms))
        for i,c in enumerate(flt):
            flt[i]+=delta
            plusval=wf.recompute(configs)
            flt[i]-=2*delta
            minuval=wf.recompute(configs)
            numgrad[:,i] = (plusval[0]*baseval[0]*np.exp(plusval[1]-baseval[1]) 
                    - minuval[0]*baseval[0]*np.exp(minuval[1]-baseval[1]))/(2*delta)
            flt[i]+=delta
        #print(gradient[k],numgrad)            
        error[k]=(np.amax(np.abs(gradient[k].reshape((-1,nparms))-numgrad)),
                  np.mean(np.abs(gradient[k].reshape((-1,nparms))-numgrad)))
    return error
            
        
def test_wf_laplacian(wf, configs, delta=1e-5):
    """ 
    Parameters:
        wf: a wavefunction object with functions wf.recompute(configs), wf.gradient(e,configs) and wf.laplacian(e,configs)
        configs: nconf x nelec x 3 position array to set the wf object
        delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision
    Tests wf.laplacian(e,epos) against numerical derivatives of wf.gradient(e,epos)
    For gradient and laplacian:
        e is the electron index
        epos is nconf x 3 positions of electron e
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    wf.laplacian(e,epos) should behave the same as gradient, except lap(\Psi(epos))/Psi(epos)
    """
    nconf, nelec = configs.shape[0:2]
    wf.recompute(configs)
    maxerror=0
    lap = np.zeros(configs.shape[:2])
    numeric = np.zeros(configs.shape[:2])

    for e in range(nelec):
        lap[:,e] = wf.laplacian(e, configs[:,e,:])
        
        for d in range(0,3):
            configsnew=configs.copy()
            configsnew[:,e,d]+=delta
            plusval=wf.testvalue(e,configsnew[:,e,:])
            plusgrad=wf.gradient(e,configsnew[:,e,:])[d]*plusval
            configsnew[:,e,d]-=2*delta
            minuval=wf.testvalue(e,configsnew[:,e,:])
            minugrad=wf.gradient(e,configsnew[:,e,:])[d]*minuval
            numeric[:,e] += (plusgrad - minugrad)/(2*delta)
    
    maxerror = np.amax(np.abs(lap-numeric))
    normerror = np.mean(np.abs((lap-numeric)/numeric))
    #print('maxerror', maxerror, np.log10(maxerror))
    #print('normerror', normerror, np.log10(normerror))
    return (maxerror,normerror)


if __name__=='__main__':
    from pyscf import lib, gto, scf
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.jastrow import Jastrow2B
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    wf=PySCFSlaterRHF(10,mol,mf)
    #wf=Jastrow2B(10,mol)
    for i in range(5):
        configs=np.random.randn(10,4,3)
        print("testing gradient: errors", test_wf_gradient(wf, configs, delta=1e-5))
    for i in range(5):
        configs=np.random.randn(10,4,3)
        print("testing laplacian: errors", test_wf_laplacian(wf, configs, delta=1e-5))
