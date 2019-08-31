import numpy as np
from pyqmc.slateruhf import sherman_morrison_row

def binary_to_occ(S, ncore):
  """
  Converts the binary cistring for a given determinant
  to occupation values for molecular orbitals within
  the determinant.
  """
  occup = [ int(i) for i in range(ncore)]
  occup += [ int(i+ncore)  for i, c in enumerate(reversed(S))
          if c=='1']
  max_orb = max(occup) 
  return (occup, max_orb)

class MultiSlater: 
    """
    A multi-determinant wave function object initialized
    via an SCF calculation. Methods and structure are very similar
    to the PySCFSlaterUHF class.
    """
    def __init__(self, mol, mf, mc):
        self.parameters = {}
        self._mol = mol
        self._nelec = tuple(mol.nelec)
        self._copy_ci(mc)

        if len(mf.mo_occ.shape) == 2:
            self.parameters["mo_coeff_alpha"] = mf.mo_coeff[0][:, :mc.ncas + mc.ncore]
            self.parameters["mo_coeff_beta"] = mf.mo_coeff[1][:, :mc.ncas + mc.ncore]
        else:
            self.parameters["mo_coeff_alpha"] = mf.mo_coeff[:, :mc.ncas + mc.ncore]
            self.parameters["mo_coeff_beta"] = mf.mo_coeff[:, :mc.ncas + mc.ncore]
        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")

    def _copy_ci(self, mc):
        """     
        Copies over determinant coefficients and MO occupations
        for a multi-configuration calculation mc.
        """
        from pyscf import fci 
        norb  = mc.ncas 
        nelec = mc.nelecas
        ncore = mc.ncore 
        orb_cutoff = 0   
        
        # find multi slater determinant occupation
        detwt = []
        occup = []
        deters = fci.addons.large_ci(mc.ci, norb, nelec, tol=0)

        for x in deters:
          detwt.append(x[0])
          alpha_occ, __ = binary_to_occ(x[1], ncore)
          beta_occ, __ =  binary_to_occ(x[2], ncore)
          
          alpha_mask = np.zeros((ncore + norb, ncore + nelec[0]))
          beta_mask = np.zeros((ncore + norb, ncore + nelec[1]))
          alpha_mask[alpha_occ,np.arange(ncore + nelec[0])] = 1
          beta_mask[beta_occ,  np.arange(ncore + nelec[1])] = 1
          occup.append([alpha_mask,beta_mask])
        
        self.parameters['det_coeff'] = np.array(detwt) 
        self._det_occup = np.array(occup)

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        #May want to vectorize later if hanging a lot
        
        mycoords = configs.reshape(
            (configs.shape[0] * configs.shape[1], configs.shape[2])
        )
        ao = self._mol.eval_gto("GTOval_sph", mycoords).reshape(
            (configs.shape[0], configs.shape[1], -1)
        )

        self._aovals = ao
        self._dets = []
        self._inverse = []
        for s in [0,1]:
            det_val = []
            det_inv = []
            for occup in self._det_occup:
                mo = ao[:, self._nelec[0]*s: self._nelec[0] + self._nelec[1]*s, :].dot(
                    self.parameters[self._coefflookup[s]]
                )
                mo = np.dot(mo, occup[s]) #mo[:,:,occup[s]]
                det_val.append(np.linalg.slogdet(mo))
                det_inv.append(np.linalg.inv(mo))
            self._inverse.append(np.array(det_inv)) #(spin, [ndet, nconfig, nelec, nelec])
            self._dets.append(det_val)              #[spin, ndet, (sign, val), nconfig]
        self._dets = np.asarray(self._dets)
        return self.value()
    
    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        #MAY want to vectorize later if it really hangs here, shouldn't!
        
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.shape[0]
        eeff = e - s * self._nelec[0]
        ao = self._mol.eval_gto("GTOval_sph", epos)
        mo = ao.dot(self.parameters[self._coefflookup[s]])
       
        ratio = 0
        for det in range(len(self._det_occup)):
            mo_det = np.dot(mo, self._det_occup[det][s]) #mo[:,self._det_occup[det][s]]
            det_ratio, self._inverse[s][det,mask, :, :] = sherman_morrison_row(
                eeff, self._inverse[s][det,mask, :, :], mo_det[mask, :]
            )
            self._updateval(det_ratio, det, s, mask)

    def value(self):
         """Return logarithm of the wave function as noted in recompute()"""
         wf_val = 0
         wf_sign = 0
        
         wf_val = np.sum(self.parameters['det_coeff'][:,np.newaxis]*\
                         self._dets[0,:,0,:]*self._dets[1,:,0,:]*\
                         np.exp(self._dets[0,:,1,:] + self._dets[1,:,1,:]),axis=0)

         wf_sign = np.sign(wf_val)
         wf_val = np.log(np.abs(wf_val))
         return wf_sign, wf_val  
    
    def _updateval(self, ratio, det, s, mask):
        self._dets[s,det,0,mask] *= np.sign(ratio)  # will not work for complex!
        self._dets[s,det,1,mask] += np.log(np.abs(ratio))
        
    def _testrow(self, e, vec):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0])
        
        ratios = np.einsum('dij,dij->di',vec, self._inverse[s][:,:,:,e-s*self._nelec[0]])
        numer = np.sum(ratios*self.parameters['det_coeff'][:,np.newaxis]*\
                         self._dets[0,:,0,:]*self._dets[1,:,0,:]*\
                         np.exp(self._dets[0,:,1,:] + self._dets[1,:,1,:]),axis=0)

        curr_val = self.value()
        denom = (curr_val[0]*np.exp(curr_val[1]))
        return numer/denom

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self._mol.eval_gto("GTOval_ip_sph", epos)
        mograd_vals = np.einsum('ajm,imk->aijk',
            aograd.dot(self.parameters[self._coefflookup[s]]),
            self._det_occup[:,s,:,:])
        
        grad = []
        for i in range(3):
          grad.append(self._testrow(e, mograd_vals[i]))

        return np.asarray(grad)/self.testvalue(e, epos)[np.newaxis, :]

    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        aolap = np.sum(self._mol.eval_gto("GTOval_sph_deriv2", epos)[[4, 7, 9]], axis=0)

        molap_vals = np.einsum('jm,imk->ijk',
            aolap.dot(self.parameters[self._coefflookup[s]]),
            self._det_occup[:,s,:,:])
        
        return self._testrow(e, molap_vals)/self.testvalue(e, epos)
    
    def testvalue(self, e, epos):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self._mol.eval_gto("GTOval_sph", epos)
        mo_vals = np.einsum('jm,imk->ijk',
            ao.dot(self.parameters[self._coefflookup[s]]),
            self._det_occup[:,s,:,:])
        
        return self._testrow(e, mo_vals)
    
    def pgradient(self):
        """Compute the parameter gradient of Psi. 
        Returns d_p \Psi/\Psi as a dictionary of numpy arrays,
        which correspond to the parameter dictionary."""
        d = {}
        det_coeff_grad = np.zeros((self._aovals.shape[0],len(self._det_occup)))
        det_coeff_grad = (self._dets[0,:,0,:]*self._dets[1,:,0,:]*\
                          np.exp(self._dets[0,:,1,:] + self._dets[1,:,1,:])).T
        
        curr_val = self.value()
        d["det_coeff"] = det_coeff_grad/(curr_val[0] * np.exp(curr_val[1]))[:,np.newaxis]
        #Mo_coeff not implemented yet
        return d
