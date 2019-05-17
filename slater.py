import numpy as np
#from pyscf import lib, gto, scf



def sherman_morrison_row(e,inv,vec):
    ratio=np.einsum("ij,ij->i",vec,inv[:,:,e])
    tmp=np.einsum("ek,ekj->ej",vec,inv)
    invnew=inv-np.einsum("ki,kj->kij",inv[:,:,e],tmp)/ratio[:,np.newaxis,np.newaxis]
    invnew[:,:,e]=inv[:,:,e]/ratio[:,np.newaxis]
    return ratio,invnew

class PySCFSlaterRHF:
    """A wave function object has a state defined by a reference configuration of electrons.
    The functions recompute() and updateinternals() change the state of the object, and 
    the rest compute and return values from that state. """
    def __init__(self,nconfig,mol,mf):
        self.occ=np.asarray(mf.mo_occ > 1.0)
        nmo=np.sum(self.occ)
        self.parameters={}
        
        self.parameters['mo_coeff']=mf.mo_coeff[:,self.occ]
        self._nconfig=nconfig
        self._mol=mol
        self._nelec=np.sum(mol.nelec)
        self._nup=int(self._nelec/2)
        assert self._nup==nmo
        self._movals=np.zeros((nconfig,2,self._nup,nmo)) # row is electron, column is mo
        self._inverse=np.zeros((nconfig,2,self._nup,self._nup))
            
    def recompute(self,configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        mycoords=configs.reshape((configs.shape[0]*configs.shape[1],configs.shape[2]))
        ao = self._mol.eval_gto('GTOval_sph', mycoords)
        mo = ao.dot(self.parameters['mo_coeff'])
        self._movals=mo.reshape((self._nconfig,2,self._nup,self._nup))
        self.dets=np.linalg.slogdet(self._movals)
        self._inverse=np.linalg.inv(self._movals)
        return self.dets[0][:,0]*self.dets[0][:,1],self.dets[1][:,0]+self.dets[1][:,1]


    def updateinternals(self,e,epos,mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        if mask is None:
            mask=[True]*self._inverse.shape[0]
        s=int(e>=self._nup)
        eeff=e-s*self._nup
        ao=self._mol.eval_gto('GTOval_sph',epos)
        mo=ao.dot(self.parameters['mo_coeff'])
        self._movals[:,s,eeff,:]=mo
        ratio,self._inverse[mask,s,:,:]=sherman_morrison_row(eeff,self._inverse[mask,s,:,:],mo[mask,:])
        self._updateval(ratio,s,mask)

    ### not state-changing functions

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        return self.dets[0][:,0]*self.dets[0][:,1],self.dets[1][:,0]+self.dets[1][:,1]
        

    def _updateval(self,ratio,s,mask):
        self.dets[0][mask,s]*=np.sign(ratio) #will not work for complex!
        self.dets[1][mask,s]+=np.log(np.abs(ratio))
    
    def _testrow(self,e,vec):
        """vec is a nconfig,nmo vector which replaces row e"""
        s=int(e>= self._nup)
        ratio=np.einsum("ij,ij->i",vec,self._inverse[:,s,:,e-s*self._nup])
        return ratio
        
    def _testcol(self,i,s,vec):
        """vec is a nconfig,nmo vector which replaces column i"""
        ratio=np.einsum("ij,ij->i",vec,self._inverse[:,s,i,:]) #need to test this!
        return ratio
    
    def gradient(self,e,epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        aograd=self._mol.eval_gto('GTOval_ip_sph',epos)
        mograd=aograd.dot(self.parameters['mo_coeff'])
        ratios=[self._testrow(e,x) for x in mograd]
        return np.asarray(ratios)/self.testvalue(e,epos)[np.newaxis,:]

    def laplacian(self,e,epos):
        """ Compute the laplacian Psi/ Psi. """
        aolap=np.sum(self._mol.eval_gto('GTOval_sph_deriv2',epos)[[4,7,9]], axis=0)
        molap=aolap.dot(self.parameters['mo_coeff'])
        ratios=self._testrow(e,molap)
        return ratios/self.testvalue(e,epos)

    def testvalue(self,e,epos):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        ao=self._mol.eval_gto('GTOval_sph',epos)
        mo=ao.dot(self.parameters['mo_coeff'])
        return self._testrow(e,mo)

    def pgradient(self):
        """Compute the parameter gradient of Psi. 
        Returns d_p \Psi/\Psi as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.
        """
        d={}
#        ao = self._mol.eval_gto('GTOval_sph', mycoords)
        
        #use testcol() to update determinant values for each mo_coeff
        return d
        
        
def test(): 
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    nconf=10
    nelec=np.sum(mol.nelec)
    slater=PySCFSlaterRHF(nconf,mol,mf)
    configs=np.random.randn(nconf,nelec,3)
    import testwf
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        print('delta', delta, "Testing gradient",testwf.test_wf_gradient(slater,configs,delta=delta))
        print('delta', delta, "Testing laplacian", testwf.test_wf_laplacian(slater,configs,delta=delta))
        print('delta', delta, "Testing pgradient", testwf.test_wf_pgradient(slater,configs,delta=delta))


    print("testing internals:", testwf.test_updateinternals(slater,configs))
    quit()
    #Test the internal update
    e=3
    slater.recompute(configs)
    inv=slater._inverse.copy()
    configsnew=configs.copy()
    configsnew[:,e,:]+=0.1
    slater.updateinternals(e,configsnew[:,e,:])
    inv_update=slater._inverse.copy()
    slater.recompute(configsnew)
    inv_recalc=slater._inverse.copy()
    print(inv_recalc-inv_update)

    


if __name__=="__main__":
    test()




