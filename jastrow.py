import numpy as np

class GaussianFunction:
    def __init__(self,exponent):
        self.parameters={}
        self.parameters['exponent']=exponent

    def value(self,x): 
        """Return the value of the function. 
        x should be a (nconfig,3) vector """
        r2=np.sum(x**2,axis=1)
        return np.exp(-self.parameters['exponent']*r2)
        
    def gradient(self,x):
        """ return gradient of the function """
        v=self.value(x)
        return -2*self.parameters['exponent']*x*v[:,np.newaxis]

    def laplacian(self,x):
        """ laplacian """
        v=self.value(x)
        alpha=self.parameters['exponent']
        return (4*alpha*alpha*x*x-2*alpha)*v[:,np.newaxis]


    def pgradient(self,x):
        """ parameter gradient """
        
class PadeFunction:
    """
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    alpha_k = alpha/2^k, k starting at 0
    """
    def __init__(self, alphak):
        self.parameters={}
        self.parameters['alphak'] = alphak

    def value(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3 (number of inner dimensions doesn't matter)
        Return:
          func: same dimensions as rvec, but the last one removed 
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = self.parameters['alphak']* r
        return (a/(1+a))**2

    def gradient(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3, displacement between particles
            For example, nconf x n_elec_pairs x 3, where n_elec_pairs could be all pairs of electrons or just the pairs that include electron e for the purpose of updating one electron.
            Or it could be nconf x nelec x natom x 3 for electron-ion displacements
        Return:
          grad: same dimensions as rvec
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        a = self.parameters['alphak']* r
        grad = 2* self.parameters['alphak']**2/(1+a)**3 *rvec
        return grad
        
    def laplacian(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3
        Return:
          lap: same dimensions as rvec, d2/dx2, d2/dy2, d2/dz2 separately
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        a = self.parameters['alphak']* r
        #lap = 6*self.parameters['alphak']**2 * (1+a)**(-4) #scalar formula
        lap = 2*self.parameters['alphak']**2 * (1+a)**(-3) \
              *(1 - 3*a/(1-a)*(rvec/r)**2)
        return lap

    def pgradient(self, rvec):
        """ Return gradient of value with respect to parameter alphak
        Parameters:
          rvec: nconf x ... x 3
        Return:
          akderiv: same dimensions as rvec, but the last one removed 
        
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = self.parameters['alphak']* r
        akderiv = 2*a/(1+a)**3 * r
        return akderiv


def eedist(epos):
    """returns a list of electron-electron distances within a collection """
    ne=epos.shape[1]
    d=np.zeros((epos.shape[0],int(ne*(ne-1)/2),3))
    c=0
    for i in range(ne):
        for j in range(i+1,ne):
            d[:,c,:]=epos[:,j,:]-epos[:,i,:]
            c+=1
    return d
    

def eedist_i(epos,vec):
    """returns a list of electron-electron distances from an electron at position 'vec'
    epos will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
    """
    ne=epos.shape[1]
    return vec[:,np.newaxis,:]-epos
    
    

class Jastrow2B:
    """A simple two-body Jastrow factor that is written as 
    :math:`\ln \Psi_J  = \sum_k c_k \sum_{i<j} b(r_{ij})`
    b are function objects
    """
    def __init__(self,nconfig,mol):
        self.parameters={}
        nexpand=2
        self._nelec=np.sum(mol.nelec)
        self._mol=mol
        self.basis=[GaussianFunction(0.2*2**n) for n in range(1,nexpand)]
        self.parameters['coeff']=np.zeros(nexpand)
        self._bvalues=np.zeros((nconfig,nexpand))
        self._eposcurrent=np.zeros((nconfig,self._nelec,3))

    def recompute(self,epos):
        """ """
        u=0.0
        self._eposcurrent=epos.copy()
        #We will save the b sums over i,j in _bvalues
        
        #package the electron-electron distances into a 1d array
        d=eedist(epos)
        d=d.reshape((-1,3))

        for i,b in enumerate(self.basis):
            self._bvalues[:,i]=np.sum(b.value(d).reshape( (epos.shape[0],-1) ),axis=1) 
        print(self._bvalues.shape)
        u=np.einsum("ij,j->i",self._bvalues,self.parameters['coeff'])
        print(u)
        return (1,u)

    def updateinternals(self,e,epos,mask=None):
        """  """
        #update b and c sums. This overlaps with testvalue()
        if mask is None:
            mask=[True]*self._eposcurrent.shape[0]
        self._eposcurrent[mask,e,:]=epos[mask,e,:]

    def value(self): 
        """  """

    def gradient(self,e,epos):
        """We compute the gradient for electron e as 
        :math:`\grad_e \ln \Psi_J = \sum_k c_k \sum_{j > e} \grad_e b_k(r_{ej})  + \sum_{i < e} \grad_e b_k(r_{ie}) `
        So we need to compute the gradient of the b's for these indices. 
        Note that we need to compute distances between electron position given and the current electron distances.
        We will need this for laplacian() as well"""
        nconf=epos.shape[0]
        ne=self._eposcurrent.shape[1]
        dnew=eedist_i(self._eposcurrent,epos)

        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]
        dnew=dnew.reshape(-1,3)
        
        delta=np.zeros((3,nconf))
        for c,b in zip(self.parameters['coeff'],self.basis):
            delta+=c*np.sum(b.gradient(dnew).reshape(nconf,-1,3),axis=1).T
        return delta


    def laplacian(self,e,epos):
        """ """
        nconf=epos.shape[0]
        ne=self._eposcurrent.shape[1]
        dnew=eedist_i(self._eposcurrent,epos)
        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]
        dnew=dnew.reshape(-1,3)
        delta=np.zeros(nconf)
        for c,b in zip(self.parameters['coeff'],self.basis):
            delta+=c*np.sum(b.laplacian(dnew).reshape(nconf,-1),axis=1).T
        return delta
        

    def testvalue(self,e,epos):
        """
        here we will evaluate the b's for a given electron (both the old and new) 
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        dnew=eedist_i(self._eposcurrent,epos).reshape((-1,3))
        dold=eedist_i(self._eposcurrent,self._eposcurrent[:,e,:]).reshape((-1,3))
        delta=np.zeros(nconf)
        for c,b in zip(self.parameters['coeff'],self.basis):
            delta+=c*np.sum((b.value(dnew)-b.value(dold)).reshape(nconf,-1),axis=1)
        return np.exp(delta)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the exponent derivatives, we will have to compute the derivative of all the b's 
        and redo the sums, similar to recompute() """
        return {'coeff':self._bvalues}



def test(): 
    from pyscf import lib, gto, scf
    
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    nconf=1
    epos=np.random.randn(nconf,np.sum(mol.nelec),3)
    
    jastrow=Jastrow2B(nconf,mol)
    jastrow.parameters['coeff']=np.random.random(jastrow.parameters['coeff'].shape)
    print('coefficients',jastrow.parameters['coeff'])
    baseval=jastrow.recompute(epos)
    e=0
    grad=jastrow.gradient(e,epos[:,e,:])
    delta=1e-9
    for d in range(0,3):
        eposnew=epos.copy()
        eposnew[:,e,d]+=delta
        baseval=jastrow.recompute(epos)
        testval=jastrow.testvalue(e,eposnew[:,e,:])
        valnew=jastrow.recompute(eposnew)
        print("testval",testval,valnew,baseval)
        print("updated value",testval-np.exp(valnew[1]-baseval[1]))
        print('derivative',d,'analytic',grad[d,:],'numerical',(valnew[1]-baseval[1])/delta)

    jastrow.laplacian(e,epos[:,e,:])
    

def test_pade():
    pade = PadeFunction(0.2)
    parms = np.random.random(4)*2-1
    epos = np.random.random((1, 8, 3))
    atoms = np.array([[0,0,0],[0,0,1.5],[1,1,0]]) 
    rvec = epos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
    print('rvec', rvec.shape)
    val = pade.value(rvec)
    grad = pade.gradient(rvec)
    lap = pade.laplacian(rvec)
    delta=1e-5
    e=2
    testlap = 0
    for i in range(3):
      pos = epos.copy()
      pos[:,e,i] += delta
      plusvec = pos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
      plusval = pade.value(plusvec)
      pos[:,e,i] -= 2*delta
      minuvec = pos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
      minuval = pade.value(minuvec)
      deriv = (plusval - minuval)/(2*delta)
      testlap += (plusval + minuval - 2*val)/delta**2
      print(i, np.linalg.norm(grad[:,e,:,i]-deriv[:,e,:]))
      print(i, grad[:,e,:,i])
      print(i, deriv[:,e,:])
    print('lap', np.linalg.norm(lap[:,e,:] - testlap[:,e,:]))

if __name__=="__main__":
    test()

