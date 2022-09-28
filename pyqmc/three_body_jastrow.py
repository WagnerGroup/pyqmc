import numpy as np
class Three_Body_JastrowSpin:

  def __init__(self, mol, a_basis, b_basis):
    self.a_basis = a_basis
    self.b_basis = b_basis
    self.parameters = {}
    self._nelec = np.sum(mol.nelec)
    self._mol = mol

    self.parameters["ccoeff"] = np.zeros((self._mol.natm,len(a_basis),len(a_basis),len(b_basis),3))
   
   # self.parameters["ccoeef_diagonal"] = np.diagonal(self.C,axis1=1,axis2=2)

    self.iscomplex = False
  
  def recompute(self, configs):
    """returns phase, log value """
    nconf, nelec = configs.configs.shape[:2]
    #n_elec in first axis to match di format
    self.sum_j = np.zeros((nelec,nconf,self._mol.natm,len(self.a_basis),len(self.a_basis),len(self.b_basis),2))
    #can dot product this with the coefficent vector and get the final U
    #order of spin channel: upup,updown,downdown
    self.sum_ij = np.zeros((nconf,self._mol.natm,len(self.a_basis),len(self.a_basis),len(self.b_basis),3))
    

    notmask = np.ones(nconf, dtype=bool)

    #electron-electron distances
    #d_upup dim is  nconf, nup(nup-1)/2,3
    #d_downdown dim is nconf, ndown(ndown-1)/2,3
    #d_updown dim is nconf, nup*ndown,3
    nup = int(self._mol.nelec[0])
    ndown=int(self._mol.nelec[1])
    d_upup, ij_upup = configs.dist.dist_matrix(configs.configs[:, :nup])
    d_updown, ij_updown = configs.dist.pairwise(
        configs.configs[:, :nup], configs.configs[:, nup:]
    )
    d_downdown, ij_downdown = configs.dist.dist_matrix(configs.configs[:, nup:]) 

    d_upup=np.swapaxes(d_upup,0,1)
    d_updown=np.swapaxes(d_updown,0,1)
    d_downdown=np.swapaxes(d_downdown,0,1)

    r_upup= np.linalg.norm(d_upup, axis=-1)
    r_updown= np.linalg.norm(d_updown, axis=-1)
    r_downdown= np.linalg.norm(d_downdown, axis=-1)

    #electron-ion distances
    di = np.zeros((nelec, nconf, self._mol.natm, 3))
    for e in range(nelec):
        di[e] = np.asarray(
            configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, e, :])
        )
    ri = np.linalg.norm(di, axis=-1)

    #bvalues are the evaluations of b bases. bm(rij)

    b_upup_values = np.zeros((int(nup*(nup-1)/2),nconf,len(self.b_basis)))
    b_updown_values = np.zeros((nup*ndown,nconf,len(self.b_basis)))
    b_downdown_values = np.zeros((int(ndown*(ndown-1)/2),nconf,len(self.b_basis)))

    #set b_values
    for i, b in enumerate(self.b_basis):
        #swap axes: nconf and nelec. for now doing it here. check if this swap works.
        b_upup_values[:,:,i] = b.value(d_upup,r_upup)
        b_updown_values[:,:,i]= b.value(d_updown,r_updown)
        b_downdown_values[:,:,i]= b.value(d_downdown,r_downdown)

    b_upup_2d_values = np.zeros((nup,nup,nconf,len(self.b_basis)))
    inds = tuple(zip(*ij_upup))
    b_upup_2d_values[inds] = self.b_upup_values 
    print('upup 2d',b_upup_2d_values.shape)

    b_updown_2d_values = np.zeros((nup,ndown,nconf,len(self.b_basis)))
    inds = tuple(zip(*ij_updown))
    b_updown_2d_values[inds] = self.b_updown_values 
    print('nelec',self._mol.nelec)
    print('updown 2d',b_updown_2d_values.shape)


    b_downdown_2d_values = np.zeros((ndown,ndown,nconf,len(self.b_basis)))
    inds = tuple(zip(*ij_downdown))
    b_downdown_2d_values[inds] = self.b_downdown_values 
    print('downdown 2d',b_downdown_2d_values.shape)
        
    #evaluate a_values
    #a_values are a evaluations ak(rIi)
    #might not need all of these, but have them defined here for now. idealy use as few as possible
    a_values = np.zeros((self._nelec,nconf,self._mol.natm,len(self.a_basis)))
    for i, a in enumerate(self.a_basis):
        #di dim nconf,I,nelec
        a_values[:,:,:,i]=a.value(di,ri)

    self.sum_ij[:,:,:,:,:,0] = np.einsum('inIk,jnIl,ijnm->nIklm',a_values[:nup],a_values[:nup],b_upup_2d_values)
    self.sum_ij[:,:,:,:,:,1] = np.einsum('inIk,jnIl,ijnm->nIklm',a_values[:nup],a_values[nup:],b_updown_2d_values)
    self.sum_ij[:,:,:,:,:,2] = np.einsum('inIk,jnIl,ijnm->nIklm',a_values[nup:],a_values[nup:],b_downdown_2d_values)


    self.C = self.parameters['ccoeff']+ self.parameters['ccoeff'].swapaxes(1,2)
    val =np.einsum('Iklms,nIklms->n',self.C,self.sum_ij)  


    #nconf,I,k,nelec , nconf,I,l,nelec , nconf,m,nelec*(nelec-1)/2 --> nconf,I,k,l,m)
    #collapse the two a_basis into one with size n(n-1)/2
    return np.ones(len(val)),val

  def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
    nconf, nelec = configs.configs.shape[:2]
    #calculate new sum overj and set it
    nup = self._mol.nelec[0]
    ndown=self._mol.nelec[1]

    configs[:,e,:] = epos
    #di has shape nconf,natoms,3
    di = np.zeros((nelec, nconf, self._mol.natm, 3))
    for j in range(nelec):
        di[j] = np.asarray(
            configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, j, :])
        )
    ri = np.linalg.norm(di, axis=-1)

    #d has shape configs,nelec-1,3
    not_e = np.arange(self._nelec) != e

    d = configs.dist.dist_i(configs[:,not_e,...],epos)
    d_swapped = np.swapaxes(d,0,1)
    d_up_swapped = d_swapped[:nup]
    d_down_swapped = d_swapped[nup:]
    r_swapped= np.linalg.norm(d_swapped, axis=-1)
    r_up_swapped = r_swapped[:nup]
    r_down_swapped = r_swapped[nup:]

    is_up = e<nup

    a_up_values = np.zeros((nup,nconf,self._mol.natm,len(self.a_basis)))
    a_down_values = np.zeros((ndown,nconf,self._mol.natm,len(self.a_basis)))
    b_upup_values = np.zeros((self.nup*(self.nup-1)/2,nconf,len(self.b_basis)))
    b_updown_values = np.zeros((self.nup*self.ndown,nconf,len(self.b_basis)))
    b_downdown_values = np.zeros((self.ndown*(self.ndown-1)/2,nconf,len(self.b_basis)))
    # a_i_values = np.zeros(nconf,self._mol.natm,len(self.a_basis))
  
    # for i, a in enumerate(self.a_basis):
    #   a_i_values[:,:,i] = a.value(di,ri)
    #can probably remove this if and replace with integer stuff. but will do that last since it decreases readability for me

    if is_up:
      sep = nup-1
      for i, a in enumerate(self.a_basis):
        a_up_values[e,:,:,i]=a.value(di,ri)
      for j, b in enumerate(self.b_basis):
        #unflatten i,j electron axes for this to work. Also think of way such that you dont have to unflatten it. 
        b_updown_values[e,:e,:,j] = b.value(d_down_swapped,r_down_swapped)[:e]
        b_updown_values[e,e+1:,:,j] = b.value(d_down_swapped,r_down_swapped)[e:]
        b_upup_values[e,:,:,j]= b.value(d_up_swapped,r_up_swapped)
    else:
      sep=nup
      for i, a in enumerate(self.a_basis):
        a_down_values[e-nup,:,:,i]=a.value(di,ri)
      for j, b in enumerate(self.b_basis):
        b_updown_values[e,:,:,j] = b.value(d_down_swapped,r_down_swapped)
        b_upup_values[e,:,:,j]= b.value(d_up_swapped,r_up_swapped)
      
    old_sum_e = self.sum_j[e]
    self.sum_j[e,:,:,:,:,:,0] = np.einsum('',a_values[e],a_up_values,b_values)
    old_sum_ij = self.sum_ij  
    self.sum_ij =  old_sum_ij - old_sum_e + self.sum_j[e]
      
  def value(self):
      #sum over i
    val =np.dot(self.parameters['coeff'],self.sumij)
      
    return np.ones(len(val)),val

  def testvalue(self, e, epos, mask=None):
    e_partial_old = self.sum_j[e]
    e_partial_new = 
    sum_ij_old = self.sum_ij
    sum_ij_new = sum_ij_old - e_partial_old + e_partial_new 
    old_value = self.value()
    new_value = np.dot(sum_ij_new,self.parameters['coeff'])
    val = np.exp(new_value-old_value)
    return val, None





