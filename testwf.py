import numpy as np

def test_wf_gradient(wf, epos, delta=1e-5):
    """ 
    Parameters:
        wf: a wavefunction object with functions wf.recompute(epos), wf.testvalue(e,epos) and wf.gradient(e,epos)
        epos: nconf x nelec x 3 position array to set the wf object
        delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision
    Tests wf.gradient(e,epos) against numerical derivatives of wf.testvalue(e,epos)
    For gradient and testvalue:
        e is the electron index
        epos is nconf x 3 positions of electron e
    wf.testvalue(e,epos) should return a ratio: the wf value at the position where electron e is moved to epos divided by the current value
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    
    """
    nconf, nelec = epos.shape[0:2]
    wf.recompute(epos)
    maxerror=0
    grad = np.zeros(epos.shape)
    numeric = np.zeros(epos.shape)
    for e in range(nelec):
        grad[:,e,:] = wf.gradient(e, epos[:,e,:]).T
        for d in range(0,3):
            eposnew=epos.copy()
            eposnew[:,e,d]+=delta
            plusval=wf.testvalue(e,eposnew[:,e,:])
            eposnew[:,e,d]-=2*delta
            minuval=wf.testvalue(e,eposnew[:,e,:])
            numeric[:,e,d] = (plusval - minuval)/(2*delta)
    maxerror = np.amax(np.abs(grad-numeric))
    normerror = np.mean(np.abs(grad-numeric))
    
    #print('maxerror', maxerror, np.log10(maxerror))
    #print('normerror', normerror, np.log10(normerror))
    return(maxerror,normerror)

def test_wf_laplacian(wf, epos, delta=1e-5):
    """ 
    Parameters:
        wf: a wavefunction object with functions wf.recompute(epos), wf.gradient(e,epos) and wf.laplacian(e,epos)
        epos: nconf x nelec x 3 position array to set the wf object
        delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision
    Tests wf.laplacian(e,epos) against numerical derivatives of wf.gradient(e,epos)
    For gradient and laplacian:
        e is the electron index
        epos is nconf x 3 positions of electron e
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    wf.laplacian(e,epos) should behave the same as gradient, except lap(\Psi(epos))/Psi(epos)
    """
    nconf, nelec = epos.shape[0:2]
    wf.recompute(epos)
    maxerror=0
    lap = np.zeros(epos.shape[:2])
    numeric = np.zeros(epos.shape[:2])

    for e in range(nelec):
        lap[:,e] = wf.laplacian(e, epos[:,e,:])
        
        for d in range(0,3):
            eposnew=epos.copy()
            eposnew[:,e,d]+=delta
            plusval=wf.testvalue(e,eposnew[:,e,:])
            plusgrad=wf.gradient(e,eposnew[:,e,:])[d]*plusval
            eposnew[:,e,d]-=2*delta
            minuval=wf.testvalue(e,eposnew[:,e,:])
            minugrad=wf.gradient(e,eposnew[:,e,:])[d]*minuval
            numeric[:,e] += (plusgrad - minugrad)/(2*delta)
    
    maxerror = np.amax(np.abs(lap-numeric))
    normerror = np.mean(np.abs((lap-numeric)/numeric))
    #print('maxerror', maxerror, np.log10(maxerror))
    #print('normerror', normerror, np.log10(normerror))
    return (maxerror,normerror)


if __name__=='__main__':
    from pyscf import lib, gto, scf
    from slater import PySCFSlaterRHF
    from jastrow import Jastrow2B
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    wf=PySCFSlaterRHF(10,mol,mf)
    #wf=Jastrow2B(10,mol)
    for i in range(5):
        epos=np.random.randn(10,4,3)
        print("testing gradient: errors", test_wf_gradient(wf, epos, delta=1e-5))
    for i in range(5):
        epos=np.random.randn(10,4,3)
        print("testing laplacian: errors", test_wf_laplacian(wf, epos, delta=1e-5))
