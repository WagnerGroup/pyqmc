import numpy as np
from pyscf import lib, gto, scf



def sherman_morrison_row(e,inv,vec):
    ratio=np.einsum("ij,ij->i",vec,inv[:,:,e])
    tmp=np.einsum("ek,ekj->ej",vec,inv)
    invnew=inv-np.einsum("ki,kj->kij",inv[:,:,e],tmp)/ratio[:,np.newaxis,np.newaxis]
    invnew[:,:,e]=inv[:,:,e]/ratio[:,np.newaxis]
    return ratio,invnew

class PySCFSlaterRHF:
    def __init__(self,nconfig,mol,mf):
        self.occ=np.asarray(mf.mo_occ > 1.0)
        nmo=np.sum(self.occ)
        print("nmo",nmo)
        self.mo_coeff=mf.mo_coeff[:,self.occ]
        self.nconfig=nconfig
        self.mol=mol
        self.nelec=np.sum(mol.nelec)
        self.nup=int(self.nelec/2)
        print("nup",self.nup)
        assert self.nup==nmo
        self.movals=np.zeros((nconfig,2,self.nup,nmo)) # row is electron, column is mo
        self.inverse=np.zeros((nconfig,2,self.nup,self.nup))
            
    def value(self,epos):
        """This computes the value from scratch. Returns the logarithm of the wave function"""
        mycoords=epos.reshape((epos.shape[0]*epos.shape[1],epos.shape[2]))
        ao = self.mol.eval_gto('GTOval_sph', mycoords)
        mo = ao.dot(self.mo_coeff)
        self.movals=mo.reshape((self.nconfig,2,self.nup,self.nup))
        self.dets=np.linalg.slogdet(self.movals)
        self.inverse=np.linalg.inv(self.movals)
        return self.dets[0][:,0]*self.dets[0][:,1],self.dets[1][:,0]+self.dets[1][:,1]


    def updateinternals(self,e,epos,mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        if mask is None:
            mask=[True]*self.inverse.shape[0]
        s=int(e>=self.nup)
        eeff=e-s*self.nup
        ao=self.mol.eval_gto('GTOval_sph',epos)
        mo=ao.dot(self.mo_coeff)
        self.movals[:,s,eeff,:]=mo
        ratio,self.inverse[mask,s,:,:]=sherman_morrison_row(eeff,self.inverse[mask,s,:,:],mo[mask,:])


    def testrow(self,e,vec):
        """vec is a nconfig,nmo vector which replaces row e"""
        s=int(e>= self.nup)
        ratio=np.einsum("ij,ij->i",vec,self.inverse[:,s,:,e-s*self.nup])
        return ratio
        
    def gradient(self,e,epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        aograd=self.mol.eval_gto('GTOval_ip_sph',epos)
        mograd=aograd.dot(self.mo_coeff)
        ratios=[self.testrow(e,x) for x in mograd]
        return np.asarray(ratios)

    def laplacian(self,e,epos):
        """ Compute the laplacian Psi/ Psi. """
        aograd=self.mol.eval_gto('GTOval_sph_deriv2',epos)
        mograd=aograd.dot(self.mo_coeff)
        ratios=[self.testrow(e,x) for x in mograd]
        return ratios[4]+ratios[7]+ratios[9]


    def testvalue(self,e,epos):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        ao=self.mol.eval_gto('GTOval_sph',epos)
        mo=ao.dot(self.mo_coeff)
        return self.testrow(e,mo)

    def parameter_gradient(self):
        """Compute the parameter gradient of Psi"""
        pass
        
        
def test(): 
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    slater=PySCFSlaterRHF(10,mol,mf)
    epos=np.random.randn(10,4,3)
    baseval=slater.value(epos)
    e=3
    grad=slater.gradient(e,epos[:,e,:])
    print(grad)

    delta=1e-9
    for d in range(0,3):
        eposnew=epos.copy()
        eposnew[:,e,d]+=delta
        baseval=slater.value(epos)
        testval=slater.testvalue(e,eposnew[:,e,:])
        valnew=slater.value(eposnew)
        print("updated value",testval-np.exp(valnew[1]-baseval[1]))
        print('derivative',d,'analytic',grad[d,:],'numerical',(valnew[1]-baseval[1])/delta)

    #Test the internal update
    slater.value(epos)
    inv=slater.inverse.copy()
    eposnew=epos.copy()
    eposnew[:,e,:]+=0.1
    slater.updateinternals(e,eposnew[:,e,:])
    inv_update=slater.inverse.copy()
    slater.value(eposnew)
    inv_recalc=slater.inverse.copy()
    print(inv_recalc-inv_update)

    


if __name__=="__main__":
    test()




