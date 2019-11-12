from pyscf.gto import eval_gto
from pyscf import gto
import numpy as np
import pyqmc

class J3:
    def __init__(self, mol):
        self.mol = mol
        # determine num of parameters
        randpos = np.random.random((1,3))
        dim = mol.eval_gto('GTOval_cart', randpos).shape[-1]
        self.parameters={}
        self.parameters["gcoeff"] = np.zeros((dim, dim))
    
    def recompute(self, configs):
        # self._configscurrent = configs.copy()
        self.nelec = configs.configs.shape[1]
        # shape of arrays:
        # ao_val: (nconf, nelec, nbasis)
        # ao_grad: (3, nconf, nelec, nbasis)
        # ao_lap: (3, nconf, nelec, nbasis)
        self.ao_val, self.ao_grad, self.ao_lap = self._get_val_grad_lap(configs)
        return (1, self.value()) #(sign, value)
    
    def updateinternals(self, e, epos, mask=None):
        nconfig = epos.configs.shape[0]
        if mask is None:
            mask = [True]*nconfig
        e_val, e_grad, e_lap = self._get_val_grad_lap(epos)
        self.ao_val[mask, e, :] = e_val[mask,0, :]
        self.ao_grad[:, mask, e, :] = e_grad[:, mask, e, :]
        self.ao_lap[:, mask, e, :] = e_lap[:, mask, e, :]

    def value(self):
        mask = np.tril(np.ones((self.nelec, self.nelec)), -1)
        return np.einsum('mn,cim, cjn, ij-> c', self.parameters["gcoeff"], self.ao_val, self.ao_val, mask)

    def gradient(self, e, epos):
        _, e_grad = self._get_val_grad_lap(epos, mode = 'grad')
        grad1 = np.einsum('mn, dcm, cjn -> d', self.parameters["gcoeff"], e_grad[:,:,0,:], self.ao_val[:,e+1:,:])
        grad2 = np.einsum('mn, cim, dcn -> d', self.parameters["gcoeff"], self.ao_val[:,:e,:], e_grad[:,:,0,:])
        return grad1 + grad2

    def laplacian(self, e,epos):
        """
        Return lap(psi)/ psi = lap(J) when psi = exp(J)
        """
        _, _, e_lap = self._get_val_grad_lap(epos)
        lap1 = np.einsum('mn, dcm, cjn-> d', self.parameters["gcoeff"], e_lap[:,:,0,:], self.ao_val[:,e+1:,:])
        lap2 = np.einsum('mn, cim, dcn -> d', self.parameters["gcoeff"], self.ao_val[:,:e,:], e_lap[:,:,0,:])
        # lap3 = np.einsum('mn, dm, dn-> d'), self.parameters["gcoeff"], e_grad[:,0,:], e_grad[] # No cross term due to the i<j constraint on electrons
        return lap1 + lap2

    def gradient_laplacian(self, e, epos):
        return self.gradient(e, epos), self.laplacian(e, epos)
    
    def pgradient(self):
        mask = np.tril(np.ones((self.nelec, self.nelec)), -1)
        coeff_grad = np.einsum('cim, cjn, ij-> cmn', self.parameters["gcoeff"], self.ao_val, self.ao_val, mask)
        return {"gcoeff":coeff_grad}

    def _get_val_grad_lap(self, configs, mode='lap'):
        nconf, nelec = np.shape(configs.configs)[:2]
        coords = np.reshape(configs.configs, (-1,3))
        if mode == "val":
            ao = np.real_if_close(self.mol.eval_gto("GTOval_cart", coords), tol = 1e4)
            return ao.reshape((nconf, nelec, -1))
        elif mode == "grad":
            ao = np.real_if_close(self.mol.eval_gto("GTOval_cart_deriv1",coords), tol = 1e4)
            val = ao[0].reshape((nconf, nelec, -1))
            grad = ao[1:4].reshape((3, nconf, nelec, -1))
            return (val, grad)
        elif mode == "lap":
            ao = np.real_if_close(self.mol.eval_gto("GTOval_cart_deriv2", coords), tol = 1e4)
            val = ao[0].reshape((nconf, nelec, -1))
            grad = ao[1:4].reshape((3, nconf, nelec, -1))
            lap = ao[4,7,9].reshape((3, nconf, nelec, -1))
            return (val, grad, lap)
# end J3 class
