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


def eidist(configs, coords):
    """returns a list of electron-ion distances"""
    ne=configs.shape[1]
    ni=len(coords)
    d=np.zeros((configs.shape[0],ne,ni,3))
    for i in range(ne):
        for j in range(ni):
            d[:,i,j,:]=configs[:,i,:]-coords[j]
    return d
    

def eedist_i(configs,vec):
    """returns a list of electron-electron distances from an electron at position 'vec'
    configs will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
    """
    return vec[:,np.newaxis,:]-configs


def eidist_i(coords,vec):
    """returns a list of electron-electron distances from an electron at position 'vec'
    configs will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
    """
    return vec[:,np.newaxis,:]-coords
    

class Jastrow2B:
    """A simple two-body Jastrow factor that is written as
    :math:`\ln \Psi_J  = \sum_k c_k \sum_{i<j} b_k(r_{ij})`
    b are function objects
    """
    def __init__(self,nconfig,mol,basis=None):
        if basis is None:
            nexpand=4
            self.basis=[GaussianFunction(0.2*2**n) for n in range(1,nexpand+1)]
        else:
            nexpand=len(basis)
            self.basis=basis
        self.parameters={}
        self._nelec=np.sum(mol.nelec)
        self._mol=mol
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
            mask=[True]*self._configscurrent.shape[0]
        self._bvalues[mask,:]+=self._get_deltab(e,epos)[mask,:]
        self._configscurrent[mask,e,:]=epos[mask,:] #order matters here!

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

    def _get_deltab(self,e,epos):
        """
        here we will evaluate the b's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        ne=self._configscurrent.shape[1]
        mask=[True]*ne
        mask[e]=False

        dnew=eedist_i(self._configscurrent,epos)[:,mask,:].reshape((-1,3))
        dold=eedist_i(self._configscurrent,self._configscurrent[:,e,:])[:,mask,:].reshape((-1,3))
        delta=np.zeros((nconf,len(self.basis)))

        for i,b in enumerate(self.basis):
            delta[:,i]+=np.sum((b.value(dnew)-b.value(dold)).reshape(nconf,-1),axis=1)
        return delta

    def testvalue(self,e,epos):
        return np.exp(np.einsum('j,ij->i',self.parameters['coeff'],self._get_deltab(e,epos)))

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative of all the b's
        and redo the sums, similar to recompute() """
        return {'coeff':self._bvalues}


class Jastrow:
    '''
    1 body and 2 body jastrow factor
    '''
    def __init__(self,nconfig,mol,basis=None):
        if basis is None:
            nexpand=5
            aexpand=8
            self.b_basis=[GaussianFunction(0.2*2**n) for n in range(1,nexpand+1)]
            self.a_basis=[GaussianFunction(0.2*2**n) for n in range(1,aexpand+1)]
        else:
            nexpand=len(basis)
            self.basis=basis
        self.parameters={}
        self._nelec=np.sum(mol.nelec)
        self._mol=mol
        self.parameters['bcoeff']=np.zeros(nexpand)
        self.parameters['acoeff']=np.zeros(aexpand)
        self._bvalues=np.zeros((nconfig,nexpand))
        self._configscurrent=np.zeros((nconfig,self._nelec,3))
        
        # First using gaussian, later change to obey cusp condition
        self._avalues=np.zeros((nconfig,mol.natm,aexpand))
        

    def recompute(self,configs):
        """ """
        u=0.0
        self._configscurrent=configs.copy()
        #We will save the b sums over i,j in _bvalues
        
        #package the electron-electron distances into a 1d array
        d=eedist(configs)
        d=d.reshape((-1,3))

        # Package the electron-ion distances into a 1d array
        di = eidist(configs, self._mol.atom_coords())
        di = di.reshape((-1, 3))
        
        for i,b in enumerate(self.b_basis):
            self._bvalues[:,i]=np.sum(b.value(d).reshape( (configs.shape[0],-1) ),axis=1)

        for i,a in enumerate(self.a_basis):
            self._avalues[:,:,i] = np.sum(a.value(di).reshape((configs.shape[0],
                                                               self._mol.natm, -1)), axis=2)

        u=np.einsum("ij,j->i",self._bvalues,self.parameters['bcoeff']) +\
          np.sum(self._avalues*self.parameters['acoeff'], axis=(2,1))

        return (1,u)


    def updateinternals(self,e,epos,mask=None):
        """  """
        #update b and c sums. This overlaps with testvalue()
        if mask is None:
            mask=[True]*self._configscurrent.shape[0]
        self._bvalues[mask,:]+=self._get_deltab(e,epos)[mask,:]
        self._avalues[mask,:]+=self._get_deltaa(e,epos)[mask,:]
        self._configscurrent[mask,e,:]=epos[mask,:]


    def value(self): 
        """  """
        u=np.einsum("ij,j->i",self._bvalues,self.parameters['bcoeff'])+\
          np.sum(self._avalues*self.parameters['acoeff'], axis=(2,1))
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

        dinew=eidist_i(self._mol.atom_coords(),epos)
        dinew=dinew.reshape(-1,3)

        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]
        dnew=dnew.reshape(-1,3)
        
        delta=np.zeros((3,nconf))

        for c,b in zip(self.parameters['bcoeff'],self.b_basis):
            delta+=c*np.sum(b.gradient(dnew).reshape(nconf,-1,3),axis=1).T

        for c,a in zip(self.parameters['acoeff'],self.a_basis):
            delta+=c*np.sum(a.gradient(dinew).reshape(nconf,-1,3),axis=1).T

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

        dinew=eidist_i(self._mol.atom_coords(),epos)
        dinew=dinew.reshape(-1,3)


        delta=np.zeros(nconf)

        for c,b in zip(self.parameters['bcoeff'],self.b_basis):
            delta+=c*np.sum(b.laplacian(dnew).reshape(nconf,-1),axis=1)

        for c,a in zip(self.parameters['acoeff'],self.a_basis):
            delta+=c*np.sum(a.laplacian(dinew).reshape(nconf,-1),axis=1)

        g=self.gradient(e,epos)
        return delta + np.sum(g**2,axis=0)
        

    def _get_deltab(self,e,epos):
        """
        here we will evaluate the b's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        ne=self._configscurrent.shape[1]
        mask=[True]*ne
        mask[e]=False

        dnew=eedist_i(self._configscurrent,epos)[:,mask,:]
        dnew=eedist_i(self._configscurrent,epos)[:,mask,:].reshape((-1,3))
        dold=eedist_i(self._configscurrent,self._configscurrent[:,e,:])[:,mask,:].reshape((-1,3))
        delta=np.zeros((nconf,len(self.b_basis)))

        for i,b in enumerate(self.b_basis):
            delta[:,i]+=np.sum((b.value(dnew)-b.value(dold)).reshape(nconf,-1),axis=1)
        return delta

    # I THINK THE BUG IS IN HERE
    def _get_deltaa(self,e,epos):
        """
        here we will evaluate the a's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        # Gets number of configurations
        nconf=epos.shape[0]

        # Gets number of ions
        ni=self._mol.natm

        # Gets new e-i distance
        #dnew=eidist_i(self._mol.atom_coords(),epos)
        dnew=eidist_i(self._mol.atom_coords(),epos).reshape((-1,3))

        # Gets old e-i distance
        dold=eidist_i(self._mol.atom_coords(),self._configscurrent[:,e,:]).reshape((-1,3))

        # Change
        delta=np.zeros((nconf,ni,len(self.a_basis)))

        for i,a in enumerate(self.a_basis):
            #delta[:,j,i]+=np.sum((a.value(dnew)-a.value(dold)).reshape(nconf, -1),axis=1)
            delta[:,:,i]+=(a.value(dnew)-a.value(dold)).reshape((nconf, -1))
        return delta


    def testvalue(self,e,epos):
        b_val = np.einsum('j,ij->i',self.parameters['bcoeff'],self._get_deltab(e,epos))
        #a_val = np.einsum('j,ij->i',self.parameters['acoeff'],self._get_deltaa(e,epos))
        a_val = np.sum(self.parameters['acoeff']*self._get_deltaa(e,epos), axis=(2,1))
        return np.exp(b_val + a_val)


    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute() """
        return {'bcoeff':self._bvalues, 'acoeff':np.sum(self._avalues,axis=1)}


def test(): 
    from pyscf import lib, gto, scf
    np.random.seed(10)
    
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    nconf=20
    configs=np.random.randn(nconf,np.sum(mol.nelec),3)
    
    #jastrow=Jastrow2B(nconf,mol)
    #jastrow.parameters['coeff']=np.random.random(jastrow.parameters['coeff'].shape)
    jastrow=Jastrow(nconf,mol)
    jastrow.parameters['bcoeff']=np.random.random(jastrow.parameters['bcoeff'].shape)
    jastrow.parameters['acoeff']=np.random.random(jastrow.parameters['acoeff'].shape)
    import testwf
    #print(testwf.test_updateinternals(jastrow,configs))
    for key, val in testwf.test_updateinternals(jastrow, configs).items():
        print(key, val)

    print()
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        print('delta', delta, "Testing gradient",
              testwf.test_wf_gradient(jastrow,configs,delta=delta))
        print('delta', delta, "Testing laplacian",
              testwf.test_wf_laplacian(jastrow,configs,delta=delta))
        print('delta', delta, "Testing pgradient",
              testwf.test_wf_pgradient(jastrow,configs,delta=delta))
        print()
    
if __name__=="__main__":
    test()

