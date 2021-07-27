import numpy as np
from pyqmc.multiplywf import Parameters

class SuperposWF:
    """
    A general representation of a wavefunction as a superposition of multiple wfs
    """

    def __init__(self, coeffs, wf_components):
        self.coeffs = coeffs 
        self.wf_components = wf_components
        self.parameters = Parameters([wf.parameters for wf in wf_components])
        self.iscomplex = bool(sum(wf.iscomplex for wf in wf_components))
        self.dtype = complex if self.iscomplex else float

    def recompute(self, configs):
        vals, signs = [],[]
        for iw,wf in enumerate(self.wf_components):
            isign, ival = wf.recompute(configs)
            vals.append(ival)
            signs.append(isign)
        signs = np.array(signs)
        vals = np.array(vals)
        ref = np.amax(vals).real
        wf_val = np.einsum('i,ij,ij->j',self.coeffs,signs,np.exp(vals - ref))
        wf_sign = wf_val / np.abs(wf_val)
        wf_val = np.log(np.abs(wf_val)) + ref
        return wf_sign, wf_val

    def updateinternals(self, e, epos, mask=None):
        for wf in self.wf_components:
            wf.updateinternals(e, epos, mask=mask)

    def value(self):
        results_components = np.array([wf.value() for wf in self.wf_components])
        ref = np.amax(results_components[:,1,:]).real
        wf_val = np.einsum('i,ij,ij->j',self.coeffs,results_components[:,0,:], np.exp(results_components[:,1,:]-ref))
        wf_sign = wf_val/np.abs(wf_val)
        wf_val = np.log(np.abs(wf_val))+ref
        return wf_sign, wf_val

    def gradient(self, e, epos):
        grads_components = np.array([wf.gradient(e, epos) for wf in self.wf_components])
        vals = np.array([wf.value() for wf in self.wf_components])
        wf_sign, wf_val = self.value()
        ratio = np.einsum('i,ij,ij->ij',self.coeffs, vals[:,0,:]/wf_sign, np.exp(vals[:,1,:]-wf_val))
        return np.einsum('ijk,ik->jk',grads_components,ratio)

    def testvalue(self, e, epos, mask=None):
        testvalue_components = np.array([wf.testvalue(e, epos, mask=mask) for wf in self.wf_components])
        vals_old = np.array([wf.value() for wf in self.wf_components])
        wf_sign_old, wf_val_old = self.value()
        ratio_old = np.einsum('i,i...j,i...j->i...j',self.coeffs, vals_old[:,0,mask]/wf_sign_old[mask], np.exp(vals_old[:,1,mask]-wf_val_old[mask]))
        return np.einsum('ij...,ij->j...',testvalue_components,np.squeeze(ratio_old))
       
    def testvalue_many(self, e, epos, mask=None):
        testvalue_components = np.array([wf.testvalue_many(e, epos, mask=mask) for wf in self.wf_components])
        vals_old = np.array([wf.value() for wf in self.wf_components])
        wf_sign_old, wf_val_old = self.value()
        ratio_old = np.einsum('i,i...j,i...j->i...j',self.coeffs,vals_old[:,0,mask]/wf_sign_old[mask], np.exp(vals_old[:,1,mask]-wf_val_old[mask]))
        return np.einsum('ijk,ij->jk',testvalue_components,np.squeeze(ratio_old))

    def gradient_value(self, e, epos):
        grad_vals = [wf.gradient_value(e, epos) for wf in self.wf_components]
        grads, vals = list(zip(*grad_vals))
        wf_vals = [wf.value() for wf in self.wf_components]
        wf_sign, wf_val = self.value()
        ratio = np.einsum('i,ij,ij->ij',self.coeffs, np.array(wf_vals)[:,0,:]/wf_sign,\
                np.exp(np.array(wf_vals)[:,1,:] - wf_val))
        return np.einsum('ijk,ik->jk',grads,ratio), np.einsum('ij,ij->j', vals, ratio)

    def gradient_laplacian(self, e, epos):
        grad_laps = [wf.gradient_laplacian(e, epos) for wf in self.wf_components]
        grads, laps = list(zip(*grad_laps))
        vals = np.array([wf.value() for wf in self.wf_components])
        wf_sign, wf_val = self.value()
        ratio = np.einsum('i,ij,ij->ij',self.coeffs, vals[:,0,:]/wf_sign,\
                np.exp(vals[:,1,:]-wf_val))
        grad = np.einsum('ijk,ik->jk', grads, ratio)
        lap = np.einsum('ij,ij->j', laps, ratio)
        return grad, lap
        

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def pgradient(self):
        return Parameters([wf.pgradient() for wf in self.wf_components])

