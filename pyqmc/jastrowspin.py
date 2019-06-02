import numpy as np
from pyqmc.func3d import GaussianFunction
from pyqmc.distance import RawDistance

class JastrowSpin:
    '''
    1 body and 2 body jastrow factor
    '''
    def __init__(self,mol,a_basis=None,b_basis=None,dist=RawDistance() ):
        """ 
        Args: 

        mol : a pyscf molecule object

        a_basis : list of func3d objects that comprise the electron-ion basis

        b_basis : list of func3d objects that comprise the electron-electron basis

        dist: a distance calculator

        """
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
        self._dist=dist
        self.parameters['bcoeff']=np.zeros((nexpand, 3))
        self.parameters['acoeff']=np.zeros((aexpand, 2))
        

    def recompute(self,configs):
        """ """
        u=0.0
        self._configscurrent=configs.copy()
        elec = self._mol.nelec
        nconfig=configs.shape[0]
        nexpand=len(self.b_basis)
        aexpand=len(self.a_basis)
        self._bvalues=np.zeros((nconfig,nexpand, 3))
        self._avalues=np.zeros((nconfig,self._mol.natm,aexpand, 2))
        
        nup=elec[0]
        d1,ij=self._dist.dist_matrix(configs[:,:nup,:])
        d2,ij=self._dist.pairwise(configs[:,:nup,:],configs[:,nup:,:])
        d3,ij=self._dist.dist_matrix(configs[:,nup:,:])
        
        d1=d1.reshape((-1,3))
        d2=d2.reshape((-1,3))
        d3=d3.reshape((-1,3))
        r1=np.linalg.norm(d1,axis=1)
        r2=np.linalg.norm(d2,axis=1)
        r3=np.linalg.norm(d3,axis=1)

        # Package the electron-ion distances into a 1d array
        di1,ij=self._dist.pairwise(self._mol.atom_coords(),configs[:,:nup,:])
        di2,ij=self._dist.pairwise(self._mol.atom_coords(),configs[:,nup:,:])
        di1 = di1.reshape((-1, 3))
        di2 = di2.reshape((-1, 3))
        ri1=np.linalg.norm(di1,axis=1)
        ri2=np.linalg.norm(di2,axis=1)
        

        # Update bvalues according to spin case
        for i,b in enumerate(self.b_basis):
            self._bvalues[:,i,0]=np.sum(b.value(d1,r1).reshape( (configs.shape[0], -1) ),axis=1)
            self._bvalues[:,i,1]=np.sum(b.value(d2,r2).reshape( (configs.shape[0], -1) ),axis=1)
            self._bvalues[:,i,2]=np.sum(b.value(d3,r3).reshape( (configs.shape[0], -1) ),axis=1)

        # Update avalues according to spin case
        for i,a in enumerate(self.a_basis):
            self._avalues[:,:,i,0] = np.sum(a.value(di1,ri1).reshape((configs.shape[0],
                                                               self._mol.natm, -1)), axis=2)
            self._avalues[:,:,i,1] = np.sum(a.value(di2,ri2).reshape((configs.shape[0],
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
        dnew=self._dist.dist_i(self._configscurrent,epos)
        dinew=self._dist.dist_i(self._mol.atom_coords(),epos)
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
        dnew=self._dist.dist_i(self._configscurrent,epos)
        mask=[True]*ne
        mask[e]=False
        dnew=dnew[:,mask,:]

        eup = int(e<nup)
        edown = int(e>=nup)
        dnewup = dnew[:,:nup-eup,:].reshape(-1,3) # Other electron is spin up
        dnewdown = dnew[:,nup-eup:,:].reshape(-1,3) # Other electron is spin down

        #Electron-ion distances
        dinew = self._dist.dist_i(self._mol.atom_coords(),epos)
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
        tmpconfigs=self._configscurrent[:,mask,:]

        dnew=self._dist.dist_i(tmpconfigs,epos)
        dold=self._dist.dist_i(tmpconfigs,self._configscurrent[:,e,:])

        eup = int(e<nup)
        edown = int(e>=nup)
        # This is the point at which we switch between up and down
        # We subtract eup because we have removed e from the set
        sep= nup-eup         
        dnewup = dnew[:,:sep,:].reshape((-1,3)) 
        dnewdown = dnew[:,sep:,:].reshape((-1,3)) 
        doldup = dold[:,:sep,:].reshape((-1,3)) 
        dolddown = dold[:,sep:,:].reshape((-1,3)) 

        rnewup=np.linalg.norm(dnewup,axis=1)
        rnewdown=np.linalg.norm(dnewdown,axis=1)
        roldup=np.linalg.norm(doldup,axis=1)
        rolddown=np.linalg.norm(dolddown,axis=1)

        delta=np.zeros((nconf,len(self.b_basis), 3))
        for i,b in enumerate(self.b_basis):
            delta[:,i,edown]+=np.sum((b.value(dnewup,rnewup)-b.value(doldup,roldup)).reshape(nconf,-1),axis=1)
            delta[:,i,1+edown]+=np.sum((b.value(dnewdown,rnewdown)-b.value(dolddown,rolddown)).reshape(nconf,-1),axis=1)
        return delta


    def _get_deltaa(self,e,epos):
        """
        here we will evaluate the a's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf=epos.shape[0]
        ni=self._mol.natm
        nup = self._mol.nelec[0]
        dnew=self._dist.dist_i(self._mol.atom_coords(),epos).reshape((-1,3))
        dold=self._dist.dist_i(self._mol.atom_coords(),self._configscurrent[:,e,:]).reshape((-1,3))
        delta=np.zeros((nconf,ni,len(self.a_basis), 2))

        rnew=np.linalg.norm(dnew,axis=1)
        rold=np.linalg.norm(dold,axis=1)

        for i,a in enumerate(self.a_basis):
            delta[:,:,i,int(e>=nup)]+=(a.value(dnew,rnew)-a.value(dold,rold)).reshape((nconf, -1))

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
    jastrow=JastrowSpin(mol,a_basis=abasis,b_basis=bbasis)
    jastrow.parameters['bcoeff']=np.random.random(jastrow.parameters['bcoeff'].shape)
    jastrow.parameters['acoeff']=np.random.random(jastrow.parameters['acoeff'].shape)
    import pyqmc.testwf as testwf
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

