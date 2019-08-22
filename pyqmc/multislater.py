import numpy as np
from slateruhf import sherman_morrison_row

#Taken from autogenv2/pyscf2qwalk
def binary_to_occ(S, ncore):
  occup = [ int(i) for i in range(ncore)]
  occup += [ int(i+ncore)  for i, c in enumerate(reversed(S))
          if c=='1']
  max_orb = max(occup) 
  return (occup, max_orb)

class MultiSlater: 
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
        norb  = mc.ncas 
        nelec = mc.nelecas
        ncore = mc.ncore 
        orb_cutoff = 0   
        
        # find multi slater determinant occupation
        detwt = []
        occup = []

        from pyscf import fci #SHIFTY, probably don't want this import 
        deters = fci.addons.large_ci(mc.ci, norb, nelec, tol=0)
          
        for x in deters:
          detwt.append(x[0])
          alpha_occ, __ = binary_to_occ(x[1], ncore)
          beta_occ, __ =  binary_to_occ(x[2], ncore)
          occup.append((alpha_occ,beta_occ))
        
        self.parameters['det_coeff'] = np.array(detwt) #Ndet, just coefficients
        self._det_occup = occup #Ndet, nested tuple of two arrays - spin up and spin down

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        mycoords = configs.reshape(
            (configs.shape[0] * configs.shape[1], configs.shape[2])
        )
        ao = self._mol.eval_gto("GTOval_sph", mycoords).reshape(
            (configs.shape[0], configs.shape[1], -1)
        )

        self._aovals = ao
        self._dets = []
        self._inverse = []
        for occup in self._det_occup:
            det_val = []
            det_inv = []
            for s in [0, 1]:
                mo = ao[:, self._nelec[0]*s: self._nelec[0] + self._nelec[1]*s, :].dot(
                    self.parameters[self._coefflookup[s]]
                )
                mo = mo[:,:,occup[s]]
                det_val.append(np.linalg.slogdet(mo))
                det_inv.append(np.linalg.inv(mo))
            self._dets.append(det_val)
            self._inverse.append(det_inv)
        
        return self.value()
       
    def value(self):
         """Return logarithm of the wave function as noted in recompute()"""
         wf_val = 0
         wf_sign = 0
         for det in range(len(self._dets)):
           wf_val += self.parameters['det_coeff'][det]*\
                self._dets[det][0][0]*self._dets[det][1][0]*\
                np.exp(self._dets[det][0][1] + self._dets[det][1][1])
         wf_sign = np.sign(wf_val)
         wf_val = np.log(np.abs(wf_val))
         
         return wf_sign, wf_val
    
    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.shape[0]
        eeff = e - s * self._nelec[0]
        ao = self._mol.eval_gto("GTOval_sph", epos)
        mo = ao.dot(self.parameters[self._coefflookup[s]])
       
        ratio = 0
        for det in range(len(self._det_occup)):
            mo_det = mo[:,self._det_occup[det][s]]
            det_ratio, self._inverse[det][s][mask, :, :] = sherman_morrison_row(
                eeff, self._inverse[det][s][mask, :, :], mo_det[mask, :]
            )
            self._updateval(det_ratio, det, s, mask)

    def _updateval(self, ratio, det, s, mask):
        self._dets[det][s][0][mask] *= np.sign(ratio)  # will not work for complex!
        self._dets[det][s][1][mask] += np.log(np.abs(ratio))

    def testvalue(self, e, epos):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self._mol.eval_gto("GTOval_sph", epos)
        mo = ao.dot(self.parameters[self._coefflookup[s]])

        ratio = 0
        for det in range(len(self._det_occup)):
            mo_det = mo[:,self._det_occup[det][s]]
            det_ratio = self._testrow(e,det,mo_det) 
            ratio += (det_ratio*self.parameters['det_coeff'][det]*\
                self._dets[det][0][0]*self._dets[det][1][0]*\
                np.exp(self._dets[det][0][1] + self._dets[det][1][1]))

        curr_val = self.value()
        ratio /= (curr_val[0]*np.exp(curr_val[1]))
        return ratio
        
    def _testrow(self, e, det, vec):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0])
        ratio = np.einsum(
            "ij,ij->i", vec, self._inverse[det][s][:, :, e - s * self._nelec[0]]
        )
        return ratio

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self._mol.eval_gto("GTOval_ip_sph", epos)
       
        grad = None
        for det in range(len(self._det_occup)):
            mograd = aograd.dot(self.parameters[self._coefflookup[s]])[:,:,self._det_occup[det][s]]
            det_ratio = [self._testrow(e,det,x) for x in mograd]
            det_grad = self.parameters['det_coeff'][det]*np.asarray(det_ratio)*\
                self._dets[det][0][0]*self._dets[det][1][0]*\
                np.exp(self._dets[det][0][1] + self._dets[det][1][1])
            if(grad is None): grad = det_grad
            else: grad += det_grad  

        curr_val = self.value()
        return grad / (self.testvalue(e, epos)[np.newaxis, :] * curr_val[0] * np.exp(curr_val[1]))

    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        aolap = np.sum(self._mol.eval_gto("GTOval_sph_deriv2", epos)[[4, 7, 9]], axis=0)
        
        lap = None
        for det in range(len(self._det_occup)):
            molap = aolap.dot(self.parameters[self._coefflookup[s]])[:,self._det_occup[det][s]]
            det_ratio = self._testrow(e,det,molap)
            det_lap = self.parameters['det_coeff'][det]*det_ratio*\
                self._dets[det][0][0]*self._dets[det][1][0]*\
                np.exp(self._dets[det][0][1] + self._dets[det][1][1])
            if(lap is None): lap = det_lap
            else: lap += det_lap

        curr_val = self.value()
        return lap / (self.testvalue(e, epos) * curr_val[0] * np.exp(curr_val[1]))

    def pgradient(self):
        """Compute the parameter gradient of Psi. 
        Returns d_p \Psi/\Psi as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.
        """
        d = {}
        for parm in self.parameters:
            if(parm == 'det_coeff'): 
                det_coeff_grad = np.zeros((self._aovals.shape[0],len(self._det_occup)))
                for det in range(det_coeff_grad.shape[1]):
                    det_coeff_grad[:,det] = self._dets[det][0][0]*self._dets[det][1][0]*\
                        np.exp(self._dets[det][0][1] + self._dets[det][1][1])
                curr_val = self.value()
                d[parm] = det_coeff_grad/(curr_val[0] * np.exp(curr_val[1]))[:,np.newaxis]
            else: pass #Mo_coeff not implemented yet
        return d
