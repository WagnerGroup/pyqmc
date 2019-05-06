import numpy as np
from pyscf import lib, gto, scf


class Jastrow:
    """
    I=ion index, i,j=electron index
    One-body: sum_iI sum_k c_k a_k(r_iI)
    Two-body: sum_ij sum_k c_k b_k(r_ij)
    Basis functions:
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    b_k(r) = (beta_k*r/(1+beta_k*r))^2
    alpha_k = alpha/2^k, k starting at 0
    beta_k = beta/2^k, k starting at 0
    """
    def __init__(self,nconfig,mol, eeparms=(0,0,0), ieparms=(0,0,0,0)):
        self.pade_a = Pade(0.2, len(ieparms))
        self.pade_b = Pade(0.5, len(eeparms))
        self.ieparms = ieparms
        self.eeparms = eeparms
        self.value = None
        self.grad = None
        self.lap = None
        self.parmDeriv = None
            
    def value(self,epos):
        """This computes the value from scratch. Returns the logarithm of the wave function"""
        nconf, nelec = epos.shape[0:2]
        # EE distances
        eepairs = np.zeros((nconf, nelec*(nelec-1)/2))
        counter=0
        for i in range(nelec):
            for j in range(i):
                eepairs[:,counter] = np.linalg.norm(epos[:,i,:]-epos[:,j,:], axis=1)
        eipairs = np.linalg.norm(epos[:,:,np.newaxis,:] - mol.atom_coords()[np.newaxis,np.newaxis,:,:], axis=-1)
        self.value = np.sum(pade_a.value(self.eiparms, eipairs), axis=(1,2)) \
                    +np.sum(pade_b.value(self.eeparms, eepairs), axis=1)

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
        
class Pade:
    def __init__(self, alpha, nbasis):
        self.alpha = alpha
        self.nbasis = nbasis
        self.ak = 2.**(-np.arange(nbasis))

    def update_alpha(self, alpha):
        self.alpha = alpha
        self.ak = 2.**(-np.arange(nbasis))
        
    def parameter_gradient(self, parms, r):
        """
        Parameters:
          r: nconf x npairs, or nconf x nelec x natom
        Return:
          nbasis x nconf x npairs
        """
        a = np.multiply.outer(self.ak, r)
        funcs = (a/(1+a))**2
        return funcs

    def basis_gradient(self, r):
        a = np.multiply.outer(self.ak, r)
        dfuncs = 2*a**2/(1+a)**3/self.alpha
        a0deriv = np.tensordot(parms, dfuncs, axes=1)
        return a0deriv

    def value(self, parms, r):
        """
        Parameters:
          parms: pade coefficients
          r: nconf x npairs, or nconf x nelec x natom
        Return:
          nconf x npairs
        """
        a = np.multiply.outer(self.ak, r)
        funcs = (a/(1+a))**2
        return np.tensordot(parms, funcs, axes=1)

    def grad(self, parms, rvec):
        """
        Parameters:
          parms: pade coefficients
          rvec: nconf x npairs x 3, or nconf x nelec x natom x 3, displacement between particles
        Return:
          nconf x npairs x 3
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = np.multiply.outer(self.ak, r)
        ak = np.reshape(self.ak, (self.nbasis, *([1]*len(r.shape))))
        rfuncs = 2* ak**2/(1+a)**3 #/ r[np.newaxis]**2
        funcs = np.expand_dims(rfuncs,-1)*rvec[np.newaxis]
        return np.tensordot(parms, funcs, axes=1)
        
    def lap(self, parms, r):
        """
        Parameters:
          parms: pade coefficients
          r: nconf x npairs, or nconf x nelec x natom
        Return:
          nconf x npairs
        """
        a = np.multiply.outer(self.ak, r)
        ak = np.reshape(self.ak, (self.nbasis, *([1]*len(r.shape))))
        funcs = 6*ak**2 * (1+a)**(-4)
        return np.tensordot(parms, funcs, axes=1)
        
class Cutoff_Cusp:
    def __init__(self)

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

def test_pade():
    pade = Pade(0.2, 4)
    parms = np.random.random(4)*2-1
    epos = np.random.random((1, 8, 3))
    atoms = np.array([[0,0,0],[0,0,1.5],[1,1,0]]) 
    rvec = epos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
    print('rvec', rvec.shape)
    val = pade.value(parms, np.linalg.norm(rvec, axis=-1))
    grad = pade.grad(parms, rvec)
    lap = pade.lap(parms, np.linalg.norm(rvec, axis=-1))
    delta=1e-5
    e=2
    testlap = 0
    for i in range(3):
      pos = epos.copy()
      pos[:,e,i] += delta
      plusvec = np.linalg.norm(pos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:], axis=-1)
      plusval = pade.value(parms, plusvec)
      pos[:,e,i] -= 2*delta
      minuvec = np.linalg.norm(pos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:], axis=-1)
      minuval = pade.value(parms, minuvec)
      deriv = (plusval - minuval)/(2*delta)
      testlap += (plusval + minuval - 2*val)/delta**2
      print(i, np.linalg.norm(grad[:,e,:,i]-deriv[:,e,:]))
      print(i, grad[:,e,:,i])
      print(i, deriv[:,e,:])
    print('lap', np.linalg.norm(lap[:,e,:] - testlap[:,e,:]))

if __name__=="__main__":
    test_pade()




