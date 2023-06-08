import numpy as np
import pyqmc.gpu as gpu
import pyqmc.orbitals
import pyqmc.supercell


class GeminalJastrow:
    r"""
    Jastrow factor defined by Casula, Attaccalite, and Sorella, J. Chem. Phys. 121, 7110 (2004); https://doi.org/10.1063/1.1794632

    ..math:: \widetilde{J}_G(\mathbf{R}) = \sum_{i\ne j} \sum_{mn} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    :math:`\chi_m(\mathbf{r})` is a set of basis orbitals

    There is a redundancy since :math:`\chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j) = \chi_n(\mathbf{r}_j) \chi_m(\mathbf{r}_i)`, so the effective coefficient is :math:`g_{mn} + g_{nm}`.
    It is important to remove the extra degree of freedom in :math:`g_{mn}` to be able to optimize the function.
    This can be done by summing only over indices :math:`i<j` or :math:`m\le n`.
    Under the second constraint, the case of :math:`m=n` will count the pair :math:`(i,j)` twice, whereas it will only be conuted once under the first constraint.
    This factor of two can be absorbed into the parameter :math:`g_{mm}` and doesn't give rise to an extra degree of freedom, so it's not a problem.
    
    Jastrow factors are required to be symmetric under particle exchange.
    So the term :math:`\chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)` must appear together with its exchanged version :math:`\chi_m(\mathbf{r}_j) \chi_n(\mathbf{r}_i)`.
    Either of the above index restrictions :math:`i<j` or :math:`m\le n` satisfies this symmetry requirement.
    It is obvious that the restriction :math:`m\le n` satisfies this symmetry requirement, since we are explicitly including both of these terms in the sum.
    The restriction :math:`i<j` is equivalent. Starting from the :math:`m\le n` form, we can take all terms having :math:`j<i` and exchange the two orbitals in the product so that :math:`m\ge n` and :math:`i<j:. 
    
    
    ..math:: J_G(\mathbf{R}) = \sum_{m\le n} \sum_{i\ne j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    ..math:: J_G(\mathbf{R}) = \sum_{m\le n} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)
                             + \sum_{m\le n} \sum_{i> j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    ..math:: J_G(\mathbf{R}) = \sum_{m\le n} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)
                             + \sum_{m\le n} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_j) \chi_n(\mathbf{r}_i)

    ..math:: J_G(\mathbf{R}) = \sum_{m\le n} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)
                             + \sum_{m\le n} \sum_{i< j} g_{mn} \chi_n(\mathbf{r}_i) \chi_m(\mathbf{r}_j)

    ..math:: J_G(\mathbf{R}) = \sum_{m\le n} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)
                             + \sum_{m\ge n} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    ..math:: J_G(\mathbf{R}) = \sum_{mn} \sum_{i< j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)
                             + \sum_{m} \sum_{i< j} g_{mm} \chi_m(\mathbf{r}_i) \chi_m(\mathbf{r}_j)

    ..math:: J_G(\mathbf{R}) = \sum_{i<j} \sum_{mn} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    In both cases, we need to evaluate every orbital at the position of all electrons, so neither form appears preferable for efficiency.
    However, the case :math:`m\le n` contains about half the parameters, meaning that the :math:`S_{ij}` matrix formed during optimization is four times smaller, so the :math:`m\le n` choice is preferable.
    """

    def __init__(self, mol, orbitals=None):
        if orbitals is None:
            if hasattr(mol, "lattice_vectors"):
                if not hasattr(mol, "original_cell"):
                    mol = pyqmc.supercell.get_supercell(mol, np.eye(3))
                else:
                    mol = make_pbc_supercell_for_gamma_aos(mol)
                kpts = [[0, 0, 0]]
                self.orbitals = pyqmc.orbitals.PBCOrbitalEvaluatorKpoints(
                    mol, kpts=kpts
                )
            else:
                self.orbitals = pyqmc.orbitals.MoleculeOrbitalEvaluator(mol, [0, 0])
        else:
            self.orbitals = orbitals
        randpos = np.random.random((1, 3))
        dim = mol.eval_gto("GTOval_cart", randpos).shape[-1]
        self.parameters = {"gcoeff": gpu.cp.zeros((dim, dim))}
        self.dtype = float
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


def make_pbc_supercell_for_gamma_aos(scell, S=None, **kwargs):
    import pyscf.pbc.gto as gto

    cell = scell.original_cell
    if S is None:
        S = scell.S
    scale = np.abs(int(np.round(np.linalg.det(S))))
    superlattice = np.dot(S, cell.lattice_vectors())
    Rpts = pyqmc.supercell.get_supercell_copies(cell.lattice_vectors(), S)
    atom = []
    for name, xyz in cell._atom:
        atom.extend([(name, xyz + R) for R in Rpts])

    newcell = gto.Cell(
        atom=atom,
        a=superlattice,
        unit="Bohr",
        spin=cell.spin * scale,
    )
    for k in ["basis", "ecp", "exp_to_discard"]:
        if k not in kwargs.keys() and k in cell.__dict__.keys():
            kwargs[k] = cell.__dict__[k]
    for k, v in kwargs.items():
        newcell.__dict__[k] = v
    newcell.build()
    newsupercell = pyqmc.supercell.get_supercell(newcell, np.eye(3))
    return newsupercell
