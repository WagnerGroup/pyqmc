import numpy as np
from func3d import GaussianFunction


def eedist(configs):
    """returns a list of electron-electron distances within a collection """
    ne=configs.shape[1]
    d=np.zeros((configs.shape[0],int(ne*(ne-1)/2),3))
    c=0
    for i in range(ne):
        for j in range(i+1,ne):
            d[:,c,:]=configs[:,j,:]-configs[:,i,:]
            c+=1
    return d
    

def eedist_i(configs,vec):
    """returns a list of electron-electron distances from an electron at position 'vec'
    configs will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
    """
    ne=configs.shape[1]
    return vec[:,np.newaxis,:]-configs
    
    

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
        self._configscurrent=np.zeros((nconfig,self._nelec,3))

    def recompute(self,configs):
        """ """
        u=0.0
        self._configscurrent=configs.copy()
        #We will save the b sums over i,j in _bvalues
        
        #package the electron-electron distances into a 1d array
        d=eedist(configs)
        d=d.reshape((-1,3))

        for i,b in enumerate(self.basis):
            self._bvalues[:,i]=np.sum(b.value(d).reshape( (configs.shape[0],-1) ),axis=1) 
        u=np.einsum("ij,j->i",self._bvalues,self.parameters['coeff'])
        return (1,u)

    def updateinternals(self,e,epos,mask=None):
        """  """
        #update b and c sums. This overlaps with testvalue()
        if mask is None:
            mask=[True]*self._eposcurrent.shape[0]
        self._configscurrent[mask,e,:]=epos[mask,:]

    def value(self): 
        """  """
        u=np.einsum("ij,j->i",self._bvalues,self.parameters['coeff'])
        return (1,u)       

    def gradient(self,e,epos):
        """We compute the gradient for electron e as 
        :math:`\grad_e \ln \Psi_J = \sum_k c_k \sum_{j > e} \grad_e b_k(r_{ej})  + \sum_{i < e} \grad_e b_k(r_{ie}) `
        So we need to compute the gradient of the b's for these indices. 
        Note that we need to compute distances between electron position given and the current electron distances.
        We will need this for laplacian() as well"""
        nconf=epos.shape[0]
        ne=self._configscurrent.shape[1]
        dnew=eedist_i(self._configscurrent,epos)

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
        ne=self._configscurrent.shape[1]
        dnew=eedist_i(self._configscurrent,epos)
        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]
        dnew=dnew.reshape(-1,3)
        delta=np.zeros(nconf)
        for c,b in zip(self.parameters['coeff'],self.basis):
            delta+=c*np.sum(b.laplacian(dnew).reshape(nconf,-1),axis=1)
        g=self.gradient(e,epos)
        return delta + np.sum(g**2,axis=0)
        

    def testvalue(self,e,epos):
        """
        here we will evaluate the b's for a given electron (both the old and new) 
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        dnew=eedist_i(self._configscurrent,epos).reshape((-1,3))
        dold=eedist_i(self._configscurrent,self._configscurrent[:,e,:]).reshape((-1,3))
        delta=np.zeros(nconf)
        for c,b in zip(self.parameters['coeff'],self.basis):
            delta+=c*np.sum((b.value(dnew)-b.value(dold)).reshape(nconf,-1),axis=1)
        return np.exp(delta)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative of all the b's 
        and redo the sums, similar to recompute() """
        return {'coeff':self._bvalues}

def test(): 
    from pyscf import lib, gto, scf
    np.random.seed(10)
    
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    nconf=20
    configs=np.random.randn(nconf,np.sum(mol.nelec),3)
    
    jastrow=Jastrow2B(nconf,mol)
    jastrow.parameters['coeff']=np.random.random(jastrow.parameters['coeff'].shape)
    print('coefficients',jastrow.parameters['coeff'])
    import testwf
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        print('delta', delta, "Testing gradient",testwf.test_wf_gradient(jastrow,configs,delta=delta))
        print('delta', delta, "Testing laplacian", testwf.test_wf_laplacian(jastrow,configs,delta=delta))
        print('delta', delta, "Testing pgradient", testwf.test_wf_pgradient(jastrow,configs,delta=delta))
    
if __name__=="__main__":
    test()

