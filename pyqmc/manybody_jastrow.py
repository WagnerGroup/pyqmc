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
        self._configscurrent = configs.copy()
        self.nelec = self._configscurrent.configs.shape[1]

        nconf, nelec = configs.configs.shape[:2]
        coords = np.reshape(configs.configs, (-1, 3))
        # No need to recompute anything...?
        val_grad_lap = self.mol.eval_gto('GTOval_sph_deriv2', coords)
        self.ao_value = val_grad_lap[0].reshape((nconf, nelec, -1))
        self.ao_gradient = val_grad_lap[1:4].reshape((3, nconf, nelec, -1))
        self.ao_hessian = val_grad_lap[4:10].reshape((6, nconf, nelec, -1))
        return (1, self.value())
    
    def updateinternals(self, e, epos, mask=None):
        pass
    
    def value(self):
        mask = np.tril(np.ones((self.nelec, self.nelec)), -1)
        return np.einsum('mn,cim, cjn, ij-> c', self.parameters["gcoeff"], self.ao_value, self.ao_value, mask)

    def gradient(self, e, epos):
        ao = np.real_if_close(self.mol.eval_gto("GTOcal_cart_deriv1", epos.configs), tol = 1e4)
        ao_val = ao[0]
        ao_grad = ao[1:]
        mask = np.zeros(self.nelec)
        mask[e:] = 1
        grad1 = np.einsum('pq, dcp, jq, j -> dc', self.parameters["gcoeff"], ao_grad, ao_val, mask)
        mask[:e-1] = 1
        mask[e-1:] = 0
        grad2 = np.einsum('pq, ip, dcq, i -> dc', self.parameters["gcoeff"], ao_grad, ao_val, mask)
        return grad1+grad2

    def laplacian(self, e,epos):
        ao = np.real_if_close(self.mol_eval_gto("GTOval_cart_deriv2", epos.configs), tol = 1e4)
        ao_val = ao[0]
        ao_grad = ao[1:4] # x, y, z
        ao_hess = ao[4, 7, 9] #xx, yy, zz
        
    def gradient_laplacian(self, e, epos):
        pass
    
    def pgradient(self):
        pass

# end J3 class
