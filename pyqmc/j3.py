import numpy as np
import pyqmc.gpu as gpu
import pyqmc.orbitals


class J3:
    def __init__(self, mol, orbitals=None):
        if hasattr(mol, "lattice_vectors"):
            raise NotImplementedError("J3 is not implemented for periodic systems")
        self.mol = mol
        if orbitals is None:
            self.orbitals = pyqmc.orbitals.MoleculeOrbitalEvaluator(mol, [0, 0])
        else:
            self.orbitals = orbitals
        randpos = np.random.random((1, 3))
        dim = mol.eval_gto("GTOval_cart", randpos).shape[-1]
        self.parameters = {"gcoeff": gpu.cp.zeros((dim, dim))}
        self.iscomplex = False
        self.optimize = "greedy"

    def recompute(self, configs):
        nconf, self.nelec = configs.configs.shape[:2]
        # shape of arrays:
        # ao_val: (nconf, nelec, nbasis)
        aos = self.orbitals.aos("GTOval_cart", configs)
        self.ao_val = aos.reshape(nconf, self.nelec, aos.shape[-1])
        return self.value()

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        if saved_values is None:
            aoval = self.orbitals.aos("GTOval_cart", epos, mask=mask)[0]
        else:
            aoval = saved_values[mask]
        self.ao_val[mask, e, :] = aoval

    def value(self):
        mask = gpu.cp.tril(gpu.cp.ones((self.nelec, self.nelec)), -1)
        vals = gpu.cp.einsum(
            "mn,cim, cjn, ij-> c",
            self.parameters["gcoeff"],
            self.ao_val,
            self.ao_val,
            mask,
            optimize=self.optimize,
        )
        signs = np.ones(len(vals))
        return (signs, gpu.asnumpy(vals))

    def compute_value(self, ao_e, ao, e):
        # `...` is for derivatives
        curr_val = gpu.cp.einsum(
            "mn, ...cm, cjn -> ...c",
            self.parameters["gcoeff"],
            ao_e,
            ao[:, :e, :],
            optimize=self.optimize,
        )
        curr_val += gpu.cp.einsum(
            "mn, ...cn, cim -> ...c",
            self.parameters["gcoeff"],
            ao_e,
            ao[:, e + 1 :, :],
            optimize=self.optimize,
        )
        return curr_val

    def gradient_value(self, e, epos):
        ao = self.orbitals.aos("GTOval_cart_deriv1", epos)[0]
        deriv = self.compute_value(ao, self.ao_val, e)
        curr_val = self.compute_value(self.ao_val[:, e], self.ao_val, e)
        val_ratio = gpu.cp.exp(deriv[0] - curr_val)
        return gpu.asnumpy(deriv[1:]), gpu.asnumpy(val_ratio), ao[0]

    def gradient(self, e, epos):
        ao = self.orbitals.aos("GTOval_cart_deriv1", epos)[0]
        grad = self.compute_value(ao[1:], self.ao_val, e)
        return gpu.asnumpy(grad)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def gradient_laplacian(self, e, epos):
        ao = self.orbitals.aos("GTOval_cart_deriv2", epos)[0]
        ao = gpu.cp.concatenate(
            [ao[1:4, ...], ao[[4, 7, 9], ...].sum(axis=0, keepdims=True)], axis=0
        )
        deriv = self.compute_value(ao, self.ao_val, e)
        grad = deriv[:3]
        lap3 = gpu.cp.einsum("dc,dc->c", grad, grad)
        return gpu.asnumpy(grad), gpu.asnumpy(deriv[3] + lap3)

    def pgradient(self):
        mask = gpu.cp.tril(
            gpu.cp.ones((self.nelec, self.nelec)), -1
        )  # to prevent double counting of electron pairs
        coeff_grad = gpu.cp.einsum(
            "cim, cjn, ij-> cmn", self.ao_val, self.ao_val, mask, optimize=self.optimize
        )
        return {"gcoeff": coeff_grad}

    def testvalue(self, e, epos, mask=None):
        if mask is None:
            mask = [True] * self.ao_val.shape[0]
        masked_ao_val = self.ao_val[mask]
        curr_val = self.compute_value(masked_ao_val[:, e], masked_ao_val, e)
        aos = self.orbitals.aos("GTOval_cart", epos, mask=mask)[0]
        new_ao_val = aos.reshape(
            len(curr_val), *epos.configs.shape[1:-1], aos.shape[-1]
        )
        # `...` is for extra dimension for ECP aux coordinates
        new_val = gpu.cp.einsum(
            "mn, c...m, cjn -> c...",
            self.parameters["gcoeff"],
            new_ao_val,
            masked_ao_val[:, :e, :],
            optimize=self.optimize,
        )
        new_val += gpu.cp.einsum(
            "mn, c...n, cim -> c...",
            self.parameters["gcoeff"],
            new_ao_val,
            masked_ao_val[:, e + 1 :, :],
            optimize=self.optimize,
        )
        return gpu.asnumpy(gpu.cp.exp((new_val.T - curr_val).T)), new_ao_val
