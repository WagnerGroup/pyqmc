from pyscf.gto import eval_gto
from pyscf import gto
import numpy as np
import pyqmc

class J3:
    def __init__(self, mol):
        self.mol = mol
        randpos = np.random.random((1,3))
        dim = mol.eval_gto('GTOval_cart', randpos).shape[-1]
        self.parameters={}
        self.parameters["gcoeff"] = np.zeros((dim, dim))
    
    def recompute(self, configs):
        self._configscurrent = configs.copy()
        self.nelec = configs.configs.shape[1]
        # shape of arrays:
        # ao_val: (nconf, nelec, nbasis)
        # ao_grad: (3, nconf, nelec, nbasis)
        # ao_lap: (3, nconf, nelec, nbasis)
        self.ao_val, self.ao_grad, self.ao_lap = self._get_val_grad_lap(configs)
        return self.value() 
    
    def updateinternals(self, e, epos, mask=None):
        nconfig = epos.configs.shape[0]
        if mask is None:
            mask = [True]*nconfig
        e_val, e_grad, e_lap = self._get_val_grad_lap(epos)
        self.ao_val[mask, e, :] = e_val[mask,0, :]
        self.ao_grad[:, mask, e, :] = e_grad[:, mask, 0, :]
        self.ao_lap[:, mask, e, :] = e_lap[:, mask, 0, :]
        self._configscurrent.configs[:, e, :] = epos.configs

    def value(self):
        mask = np.tril(np.ones((self.nelec, self.nelec)), -1)
        vals = np.einsum('mn,cim, cjn, ij-> c', self.parameters["gcoeff"], self.ao_val, self.ao_val, mask)
        signs = np.ones(len(vals))
        return (signs, vals)

    def gradient(self, e, epos):
        _, e_grad = self._get_val_grad_lap(epos, mode = 'grad')
        grad1 = np.einsum('mn, dcm, cjn -> dc', self.parameters["gcoeff"], e_grad[:,:,0,:], self.ao_val[:,:e,:])
        grad2 = np.einsum('mn, cim, dcn -> dc', self.parameters["gcoeff"], self.ao_val[:,e+1:,:], e_grad[:,:,0,:])
        return grad1 + grad2

    def laplacian(self, e,epos):
        """
        Return lap(psi)/ psi = lap(J) when psi = exp(J)
        """
        _, e_grad, e_lap = self._get_val_grad_lap(epos)
        lap1 = np.einsum('mn, dcm, cjn-> c', self.parameters["gcoeff"], e_lap[:,:,0,:], self.ao_val[:,:e,:])
        lap2 = np.einsum('mn, cim, dcn -> c', self.parameters["gcoeff"], self.ao_val[:,e+1:,:], e_lap[:,:,0,:])
       
        grad1 = np.einsum('mn, dcm, cjn -> dc', self.parameters["gcoeff"], e_grad[:,:,0,:], self.ao_val[:,:e,:])
        grad2 = np.einsum('mn, cim, dcn -> dc', self.parameters["gcoeff"], self.ao_val[:,e+1:,:], e_grad[:,:,0,:])
        grad = grad1 + grad2
        
        lap3 = np.einsum('dc,dc->c', grad, grad)
        return lap1 + lap2 + lap3

    def gradient_laplacian(self, e, epos):
        _, e_grad, e_lap = self._get_val_grad_lap(epos)
        lap1 = np.einsum('mn, dcm, cjn-> c', self.parameters["gcoeff"], e_lap[:,:,0,:], self.ao_val[:,:e,:])
        lap2 = np.einsum('mn, cim, dcn -> c', self.parameters["gcoeff"], self.ao_val[:,e+1:,:], e_lap[:,:,0,:])
       
        grad1 = np.einsum('mn, dcm, cjn -> dc', self.parameters["gcoeff"], e_grad[:,:,0,:], self.ao_val[:,:e,:])
        grad2 = np.einsum('mn, cim, dcn -> dc', self.parameters["gcoeff"], self.ao_val[:,e+1:,:], e_grad[:,:,0,:])
        grad = grad1 + grad2
        
        lap3 = np.einsum('dc,dc->c', grad, grad)
        return grad, lap1 + lap2 + lap3

    def pgradient(self):
        mask = np.tril(np.ones((self.nelec, self.nelec)), -1) # to prevent double counting of electron pairs
        coeff_grad = np.einsum('cim, cjn, ij-> cmn', self.ao_val, self.ao_val, mask)
        return {"gcoeff":coeff_grad}

    def _get_val_grad_lap(self, configs, mode='lap'):
        shape = configs.configs.shape
        if len(shape) == 3:
            nconf, nelec= configs.configs.shape[:2]
            coords = np.reshape(configs.configs, (nconf*nelec, 3))
        else: 
            nconf = shape[0]
            nelec = 1
            coords = configs.configs
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
            lap = ao[[4,7,9]].reshape((3, nconf, nelec, -1))
            return (val, grad, lap)

    def testvalue(self, e, epos, mask=None):
        curr_epos = self._configscurrent.configs[:, e, :].copy()
        curr_val = self.value()
        self._configscurrent.configs[:, e, :] = epos.configs
        self.recompute(self._configscurrent)
        after_val = self.value()
        self._configscurrent.configs[:, e, :] = curr_epos
        self.recompute(self._configscurrent)
        return np.exp(after_val[0]*after_val[1]-curr_val[0]*curr_val[1])
