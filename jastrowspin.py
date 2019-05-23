import numpy as np
from func3d import GaussianFunction


def eedist_spin(configs, nup, ndown):
    """returns a separate list of electron-electron distances within a collection 
    according to the spin state of the electron pair, assumes forst nup electrons are spin up
    and the remaining ndown electrons are spin down"""
    ne=configs.shape[1]
    d1=np.zeros((configs.shape[0],int(nup*(nup-1)/2),3))      # up-up case
    d2=np.zeros((configs.shape[0],int(nup*ndown),3))          # up-down case
    d3=np.zeros((configs.shape[0],int(ndown*(ndown-1)/2),3))  # down-down case

    # First electrons are spin up by convenction
    for i in range(int(nup*(nup-1)/2)): # Both spins up
        for j in range(i+1, nup):
            d1[:,i,:] = configs[:,j,:]-configs[:,i,:]

    for i in range(nup): # One spin up, one spin down
        for j in range(ndown):
            d2[:,i*ndown+j,:] = configs[:,j+nup,:]-configs[:,i,:]

    c=0
    for i in range(nup,ne):
        for j in range(i+1,ne):
            d3[:,c,:]=configs[:,j,:]-configs[:,i,:]
            c+=1

    return d1, d2, d3

    

def eidist_spin(configs, coords, nup, ndown):
    """returns a separate list of electron-ion distances according to the spin
    of each electron assumes the first nup electrons are spin up and the remaining spin
    down electrons are spin down"""
    ne=configs.shape[1]
    ni=len(coords)
    d1=np.zeros((configs.shape[0],nup,ni,3))   # up case
    d2=np.zeros((configs.shape[0],ndown,ni,3)) # down case

    # First electrons are spin up by convenction
    for i in range(nup): # spin up case
        for j in range(ni):
            d1[:,i,j,:] = configs[:,i,:]-coords[j]
            
    for i in range(ndown): # spin down case
        for j in range(ni):
            d2[:,i,j,:] = configs[:,i+nup,:]-coords[j]

    return d1, d2


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
    



class JastrowSpin:
    '''
    1 body and 2 body jastrow factor
    '''
    def __init__(self,nconfig,mol,a_basis=None,b_basis=None):
        if b_basis is None:
            nexpand=5
            self.b_basis=[GaussianFunction(0.2*2**n) for n in range(1,nexpand+1)]
        else:
            nexpand=len(b_basis)
            self.b_basis=b_basis

        if a_basis is None:
            aexpand=4
            self.a_basis=[GaussianFunction(0.2*2**n) for n in range(1,aexpand+1)]
        else:
            aexpand=len(a_basis)
            self.a_basis=a_basis
            
        self.parameters={}
        self._nelec=np.sum(mol.nelec)
        self._mol=mol
        self.parameters['bcoeff']=np.zeros((nexpand, 3))
        self.parameters['acoeff']=np.zeros((aexpand, 2))
        self._bvalues=np.zeros((nconfig,nexpand, 3))
        self._configscurrent=np.zeros((nconfig,self._nelec,3))
        self._avalues=np.zeros((nconfig,mol.natm,aexpand, 2))
        

    def recompute(self,configs):
        """ """
        u=0.0
        self._configscurrent=configs.copy()
        elec = self._mol.nelec
        
        #package the electron-electron distances into a 1d array
        d1, d2, d3 = eedist_spin(configs, elec[0], elec[1])
        d1=d1.reshape((-1,3))
        d2=d2.reshape((-1,3))
        d3=d3.reshape((-1,3))

        # Package the electron-ion distances into a 1d array
        di1, di2 = eidist_spin(configs, self._mol.atom_coords(), elec[0], elec[1])
        di1 = di1.reshape((-1, 3))
        di2 = di2.reshape((-1, 3))
        
        # Update bvalues according to spin case
        for i,b in enumerate(self.b_basis):
            self._bvalues[:,i,0]=np.sum(b.value(d1).reshape( (configs.shape[0], -1) ),axis=1)
            self._bvalues[:,i,1]=np.sum(b.value(d2).reshape( (configs.shape[0], -1) ),axis=1)
            self._bvalues[:,i,2]=np.sum(b.value(d3).reshape( (configs.shape[0], -1) ),axis=1)

        # Update avalues according to spin case
        for i,a in enumerate(self.a_basis):
            self._avalues[:,:,i,0] = np.sum(a.value(di1).reshape((configs.shape[0],
                                                               self._mol.natm, -1)), axis=2)
            self._avalues[:,:,i,1] = np.sum(a.value(di2).reshape((configs.shape[0],
                                                               self._mol.natm, -1)), axis=2)

        u=np.sum(self._bvalues*self.parameters['bcoeff'], axis=(2,1)) +\
          np.sum(self._avalues*self.parameters['acoeff'], axis=(3,2,1))

        return (1,u)


    def updateinternals(self,e,epos,mask=None):
        """ Update a, b, and c sums. """
        if mask is None:
            mask=[True]*self._configscurrent.shape[0]
        self._bvalues[mask,:,:]+=self._get_deltab(e,epos)[mask,:,:]
        self._avalues[mask,:,:,:]+=self._get_deltaa(e,epos)[mask,:,:,:]
        self._configscurrent[mask,e,:]=epos[mask,:]


    def value(self): 
        """Compute the current log value of the wavefunction"""
        u=np.sum(self._bvalues*self.parameters['bcoeff'], axis=(2,1)) +\
          np.sum(self._avalues*self.parameters['acoeff'], axis=(3,2,1))
        return (1,u)       


    def gradient(self,e,epos):
        """We compute the gradient for electron e as
        :math:`grad_e ln Psi_J = sum_k c_k sum_{j > e} grad_e b_k(r_{ej}) + sum_{i < e} grad_e b_k(r_{ie}) `
        So we need to compute the gradient of the b's for these indices.
        Note that we need to compute distances between electron position given and the current electron distances.
        We will need this for laplacian() as well"""
        nconf=epos.shape[0]
        ne=self._configscurrent.shape[1]
        nup = self._mol.nelec[0]
        dnew=eedist_i(self._configscurrent,epos)

        dinew=eidist_i(self._mol.atom_coords(),epos)
        dinew=dinew.reshape(-1,3)

        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]

        delta=np.zeros((3,nconf))

        # Check if selected electron is spin up or down
        eup = int(e<nup)
        edown = int(e>=nup)

        dnewup= dnew[:,:nup-eup,:].reshape(-1,3) # Other electron is spin up
        dnewdown= dnew[:,nup-eup:,:].reshape(-1,3) # Other electron is spin down

        for c,b in zip(self.parameters['bcoeff'],self.b_basis):
            delta += c[edown]*np.sum(b.gradient(dnewup).reshape(nconf,-1,3),axis=1).T
            delta += c[1+edown]*np.sum(b.gradient(dnewdown).reshape(nconf,-1,3),axis=1).T

        for c,a in zip(self.parameters['acoeff'],self.a_basis):
            delta+=c[edown]*np.sum(a.gradient(dinew).reshape(nconf,-1,3),axis=1).T

        return delta


    def laplacian(self,e,epos):
        """ """
        nconf=epos.shape[0]
        nup = self._mol.nelec[0]
        ne=self._configscurrent.shape[1]
        
        # Get and break up eedist_i
        dnew=eedist_i(self._configscurrent,epos)
        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]

        eup = int(e<nup)
        edown = int(e>=nup)
        dnewup = dnew[:,:nup-eup,:].reshape(-1,3) # Other electron is spin up
        dnewdown = dnew[:,nup-eup:,:].reshape(-1,3) # Other electron is spin down

        # Get and reshape eidist_i
        dinew = eidist_i(self._mol.atom_coords(),epos)
        dinew = dinew.reshape(-1,3)

        delta=np.zeros(nconf)

        # b-value component
        for c,b in zip(self.parameters['bcoeff'],self.b_basis):
            delta += c[edown]*np.sum(b.laplacian(dnewup).reshape(nconf,-1),axis=1)
            delta += c[1+edown]*np.sum(b.laplacian(dnewdown).reshape(nconf,-1),axis=1)

        # a-value component
        for c,a in zip(self.parameters['acoeff'],self.a_basis):
            delta += c[edown]*np.sum(a.laplacian(dinew).reshape(nconf,-1),axis=1)

        g=self.gradient(e,epos)
        return delta + np.sum(g**2,axis=0) 


    def _get_deltab(self,e,epos):
        """
        here we will evaluate the b's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        ne=self._configscurrent.shape[1]
        nup = self._mol.nelec[0]
        mask=[True]*ne
        mask[e]=False

        dnew=eedist_i(self._configscurrent,epos)[:,mask,:]
        dold=eedist_i(self._configscurrent,self._configscurrent[:,e,:])[:,mask,:]

        eup = int(e<nup)
        edown = int(e>=nup)
        # This is the point at which we switch between up and down
        # We subtract eup because we have removed e from the set
        sep= nup-eup         
        dnewup = dnew[:,:sep,:].reshape((-1,3)) 
        dnewdown = dnew[:,sep:,:].reshape((-1,3)) 
        doldup = dold[:,:sep,:].reshape((-1,3)) 
        dolddown = dold[:,sep:,:].reshape((-1,3)) 

        delta=np.zeros((nconf,len(self.b_basis), 3))
        for i,b in enumerate(self.b_basis):
            delta[:,i,edown]+=np.sum((b.value(dnewup)-b.value(doldup)).reshape(nconf,-1),
                                     axis=1)
            delta[:,i,1+edown]+=np.sum((b.value(dnewdown)-b.value(dolddown)).reshape(nconf,-1),
                                       axis=1)
        return delta


    def _get_deltaa(self,e,epos):
        """
        here we will evaluate the a's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        ni=self._mol.natm
        nup = self._mol.nelec[0]
        dnew=eidist_i(self._mol.atom_coords(),epos).reshape((-1,3))
        dold=eidist_i(self._mol.atom_coords(),self._configscurrent[:,e,:]).reshape((-1,3))
        delta=np.zeros((nconf,ni,len(self.a_basis), 2))

        for i,a in enumerate(self.a_basis):
            delta[:,:,i,int(e>=nup)]+=(a.value(dnew)-a.value(dold)).reshape((nconf, -1))

        return delta


    def testvalue(self,e,epos):
        b_val = np.sum(self._get_deltab(e,epos)*self.parameters['bcoeff'],
                       axis=(2,1))
        a_val = np.sum(self._get_deltaa(e,epos)*self.parameters['acoeff'],
                       axis=(3,2,1))
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
    l = dir(mol)
    nconf=20
    configs=np.random.randn(nconf,np.sum(mol.nelec),3)
    
    abasis=[GaussianFunction(0.2),GaussianFunction(0.4)]
    bbasis=[GaussianFunction(0.2),GaussianFunction(0.4)]
    jastrow=JastrowSpin(nconf,mol,a_basis=abasis,b_basis=bbasis)
    jastrow.parameters['bcoeff']=np.random.random(jastrow.parameters['bcoeff'].shape)
    jastrow.parameters['acoeff']=np.random.random(jastrow.parameters['acoeff'].shape)
    import testwf
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

