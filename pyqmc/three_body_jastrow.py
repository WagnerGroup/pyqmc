import numpy as np


class Three_Body_JastrowSpin:
    def __init__(self, mol, a_basis, b_basis):
        self.a_basis = a_basis
        self.b_basis = b_basis
        self.parameters = {}
        self._nelec = np.sum(mol.nelec)
        self._mol = mol

        self.parameters["ccoeff"] = np.zeros(
            (self._mol.natm, len(a_basis), len(a_basis), len(b_basis), 3)
        )

        # self.parameters["ccoeef_diagonal"] = np.diagonal(self.C,axis1=1,axis2=2)

        self.iscomplex = False

    def recompute(self, configs):
        """returns phase, log value """
        nconf, nelec = configs.configs.shape[:2]
        #n_elec in first axis to match di format
        self.sum_j = np.zeros((nelec,nconf,self._mol.natm,len(self.a_basis),len(self.a_basis),len(self.b_basis),2))
        #can dot product this with the coefficent vector and get the final U
        #order of spin channel: upup,updown,downdown

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
        d_all = [d_upup, d_updown, d_downdown]
        ij_all = [ij_upup, ij_updown, ij_downdown]
        r_all = [np.linalg.norm(d, axis=-1) for d in d_all]

        #electron-ion distances
        di = np.zeros((nelec, nconf, self._mol.natm, 3))
        for e in range(nelec):
            di[e] = np.asarray(
                configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, e, :])
            )
        ri = np.linalg.norm(di, axis=-1)

        #bvalues are the evaluations of b bases. bm(rij)
        b_2d_values = []
        for s, shape in enumerate([(nup, nup), (nup, ndown), (ndown, ndown)]):
            bvalues = np.stack([b.value(d_all[s], r_all[s]) for i, b in enumerate(self.b_basis)], axis=-1)
            b_2d_values_s = np.zeros((*shape,nconf,len(self.b_basis)))
            inds = tuple(zip(*ij_all[s]))
            b_2d_values_s[inds] = bvalues.swapaxes(0, 1)
            b_2d_values.append(b_2d_values_s)
            print(shape,b_2d_values_s.shape)

        #evaluate a_values
        #a_values are a evaluations ak(rIi)
        #might not need all of these, but have them defined here for now. idealy use as few as possible
        a_values = np.zeros((self._nelec,nconf,self._mol.natm,len(self.a_basis)))
        for i, a in enumerate(self.a_basis):
            #di dim nconf,I,nelec
            a_values[:,:,:,i]=a.value(di,ri)

        self.sum_ij = np.zeros((nconf,self._mol.natm,len(self.a_basis),len(self.a_basis),len(self.b_basis),3))
        self.sum_ij[:,:,:,:,:,0] = np.einsum('inIk,jnIl,ijnm->nIklm',a_values[:nup],a_values[:nup],b_2d_values[0])
        self.sum_ij[:,:,:,:,:,1] = np.einsum('inIk,jnIl,ijnm->nIklm',a_values[:nup],a_values[nup:],b_2d_values[1])
        self.sum_ij[:,:,:,:,:,2] = np.einsum('inIk,jnIl,ijnm->nIklm',a_values[nup:],a_values[nup:],b_2d_values[2])

        self.C = self.parameters['ccoeff']+ self.parameters['ccoeff'].swapaxes(1,2)
        val =np.einsum('Iklms,nIklms->n',self.C,self.sum_ij)  

        return np.ones(len(val)),val

    # def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
    # nconf, nelec = configs.configs.shape[:2]
    # #calculate new sum overj and set it
    # nup = self._mol.nelec[0]
    # ndown=self._mol.nelec[1]

    # configs[:,e,:] = epos
    # #di has shape nconf,natoms,3
    # di = np.zeros((nelec, nconf, self._mol.natm, 3))
    # for j in range(nelec):
    #     di[j] = np.asarray(
    #         configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, j, :])
    #     )
    # ri = np.linalg.norm(di, axis=-1)

    # diup = di[:nup]
    # didown=di[nup:]
    # riup = ri[:nup]
    # ridown=ri[nup:]

    # #d has shape configs,nelec-1,3
    # not_e = np.arange(self._nelec) != e

    # d = configs.dist.dist_i(configs[:,not_e,...],epos)
    # d_swapped = np.swapaxes(d,0,1)
    # d_up_swapped = d_swapped[:nup]
    # d_down_swapped = d_swapped[nup:]
    # r_swapped= np.linalg.norm(d_swapped, axis=-1)
    # r_up_swapped = r_swapped[:nup]
    # r_down_swapped = r_swapped[nup:]

    # is_up = e<nup

    # a_up_values = np.zeros((nup,nconf,self._mol.natm,len(self.a_basis)))
    # a_down_values = np.zeros((ndown,nconf,self._mol.natm,len(self.a_basis)))
    # b_upup_values = np.zeros((self.nup*(self.nup-1)/2,nconf,len(self.b_basis)))
    # b_updown_values = np.zeros((self.nup*self.ndown,nconf,len(self.b_basis)))
    # b_downdown_values = np.zeros((self.ndown*(self.ndown-1)/2,nconf,len(self.b_basis)))
    # # a_i_values = np.zeros(nconf,self._mol.natm,len(self.a_basis))

    # # for i, a in enumerate(self.a_basis):
    # #   a_i_values[:,:,i] = a.value(di,ri)
    # #can probably remove this if and replace with integer stuff. but will do that last since it decreases readability for me

    # if is_up:
    #   sep = nup-1
    #   for i, a in enumerate(self.a_basis):
    #     a_up_values[e,:,:,i]=a.value(di,ri)
    #   for j, b in enumerate(self.b_basis):
    #     #unflatten i,j electron axes for this to work. Also think of way such that you dont have to unflatten it.
    #     b_updown_values[e,:e,:,j] = b.value(d_down_swapped,r_down_swapped)[:e]
    #     b_updown_values[e,e+1:,:,j] = b.value(d_down_swapped,r_down_swapped)[e:]
    #     b_upup_values[e,:,:,j]= b.value(d_up_swapped,r_up_swapped)
    # else:
    #   sep=nup
    #   for i, a in enumerate(self.a_basis):
    #     a_down_values[e-nup,:,:,i]=a.value(di,ri)
    #   for j, b in enumerate(self.b_basis):
    #     b_updown_values[e,:,:,j] = b.value(d_down_swapped,r_down_swapped)
    #     b_upup_values[e,:,:,j]= b.value(d_up_swapped,r_up_swapped)

    # old_sum_e = self.sum_j[e]
    # self.sum_j[e,:,:,:,:,:,0] = np.einsum('',a_values[e],a_up_values,b_values)
    # old_sum_ij = self.sum_ij
    # self.sum_ij =  old_sum_ij - old_sum_e + self.sum_j[e]

    def value(self):
        # sum over i
        # val = np.dot(self.parameters["coeff"], self.sumij)
        val =self.val


        return np.ones(len(val)), val

    def testvalue(self, e, epos, mask=None):
        configs = self.configs
        nconf, nelec = configs.configs.shape[:2]
        edown = int(e >= self._mol.nelec[0])
        epos_old= configs[:,e,:]
        rpos_old = np.linalg.norm(epos_old,axis=-1)

        nup = int(self._mol.nelec[0])
        ndown = int(self._mol.nelec[1])
        sep = nup - int(e < nup)
        not_e = np.arange(self._nelec) != e

        de_old = np.zeros((nconf,self._mol.nelec,3))
        de_old = configs.dist.dist_i(configs.configs[:, not_e, :], configs.configs[:, e, :])
        re_old = np.linalg.norm(de_old,axis=-1)

        de_new = np.zeros((nconf,self._mol.nelec,3))
        de_new = configs.dist.dist_i(configs.configs[:, not_e, :], epos)
        re_new = np.linalg.norm(de_new,axis=-1)

        di = np.zeros((nelec, nconf, self._mol.natm, 3))
        #di_e = np.zeros((nconf, self._mol.natm, 3))
        di_e = configs.dist.dist_i(self._mol.atom_coords(), epos)
        for e in range(nelec):
            di[e] = np.asarray(
                configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, e, :])
            )
            
        ri_e = np.linalg.norm(di_e,axis=-1)

        ri = np.linalg.norm(di, axis=-1)
        rpos = np.linalg.norm(epos, axis=-1)

        a_values = np.zeros((self._nelec, nconf, self._mol.natm, len(self.a_basis)))
        a_up_values = np.zeros((nup, nconf, self._mol.natm, len(self.a_basis)))
        a_down_values = np.zeros((ndown, nconf, self._mol.natm, len(self.a_basis)))
        ae_new = np.zeros((nconf, self._mol.natm, len(self.a_basis)))
        ae_old = np.zeros((nconf, self._mol.natm, len(self.a_basis)))

        for i, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_values[:, :, :, i] = a.value(di, ri)
            a_up_values[:, :, :, i] = a.value(di[:nup], ri[:nup])
            a_down_values[:, :, :, i] = a.value(di[nup:], ri[nup:])
            ae_new[:, :, i] = a.value(di_e,ri_e)
        
        
        ae_old= a_values[e]
        a_values_up = a_values[not_e[:sep,...]]

        b_values_old = np.zeros(self._nelec, nconf, len(self.b_basis))
        b_values_new = np.zeros(self._nelec, nconf, len(self.b_basis))

        for i, b in enumerate(self.b_basis):
            # swap axes: nconf and nelec. for now doing it here. check if this swap works.
          b_values_old[:, :, i] = b.value(de_old,re_old)
          b_values_new[:, :, i] = b.value(de_new,re_new)

        b_up_old = b_values_old[..., :sep]
        b_down_old = b_values_old[..., sep:]
        b_up_new = b_values_new[..., :sep]
        b_down_new = b_values_new[..., sep:]

        e_partial_old= np.einsum('kil,ljI,mij->iklmI',ae_old,a_values_up,b_up_old)

        e_partial_new = np.einsum('kil,ljI,mij->iklmI',ae_new,a_values_up,b_up_new)

        sum_ij_old = self.sum_ij
        sum_ij_new = sum_ij_old - 2 * e_partial_old + 2 * e_partial_new
        old_value = self.value()
        new_value = np.dot(sum_ij_new, self.parameters["coeff"])
        val = np.exp(new_value - old_value)
        return val, None
