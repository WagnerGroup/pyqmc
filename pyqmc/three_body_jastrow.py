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
        """returns phase, log value"""
        self._configscurrent = configs.copy()
        nconf, nelec = configs.configs.shape[:2]
        na, nb = len(self.a_basis), len(self.b_basis)
        # n_elec in first axis to match di format
        self.sum_j = np.zeros((nelec, nconf, self._mol.natm, na, na, nb, 2))
        # order of spin channel: upup,updown,downdown

        # electron-electron distances
        # d_upup dim is  nconf, nup(nup-1)/2,3
        # d_downdown dim is nconf, ndown(ndown-1)/2,3
        # d_updown dim is nconf, nup*ndown,3
        nup = int(self._mol.nelec[0])
        ndown = int(self._mol.nelec[1])
        d_upup, ij_upup = configs.dist.dist_matrix(configs.configs[:, :nup])
        d_updown, ij_updown = configs.dist.pairwise(
            configs.configs[:, :nup], configs.configs[:, nup:]
        )
        d_downdown, ij_downdown = configs.dist.dist_matrix(configs.configs[:, nup:])
        d_all = [d_upup, d_updown, d_downdown]
        ij_all = [ij_upup, ij_updown, ij_downdown]
        r_all = [np.linalg.norm(d, axis=-1) for d in d_all]

        # electron-ion distances
        di = np.zeros((nelec, nconf, self._mol.natm, 3))
        for e in range(nelec):
            di[e] = np.asarray(
                configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, e, :])
            )
        ri = np.linalg.norm(di, axis=-1)

        # bvalues are the evaluations of b bases. bm(rij)
        b_2d_values = []
        for s, shape in enumerate([(nup, nup), (nup, ndown), (ndown, ndown)]):
            bvalues = np.stack(
                [b.value(d_all[s], r_all[s]) for i, b in enumerate(self.b_basis)],
                axis=-1,
            )
            b_2d_values_s = np.zeros((*shape, nconf, len(self.b_basis)))
            inds = tuple(zip(*ij_all[s]))
            b_2d_values_s[inds] = bvalues.swapaxes(0, 1)
            b_2d_values.append(b_2d_values_s)

        # evaluate a_values ak(rIi)
        # might not need all of these, but have them defined here for now. idealy use as few as possible
        a_values = np.zeros((self._nelec, nconf, self._mol.natm, len(self.a_basis)))
        for i, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_values[:, :, :, i] = a.value(di, ri)

        self.C = self.parameters["ccoeff"] + self.parameters["ccoeff"].swapaxes(1, 2)

        self.sum_jC = np.zeros((nelec, nconf))
        updown = np.einsum(
            "inIk,jnIl,ijnm,Iklm->ijn",
            a_values[:nup],
            a_values[nup:],
            b_2d_values[1],
            self.C[..., 1],
        )
        self.sum_jC[:nup] += updown.sum(axis=1)
        self.sum_jC[nup:] += updown.sum(axis=0)
        self.sum_jC[:nup] += np.einsum(
            "inIk,jnIl,ijnm,Iklm->in",
            a_values[:nup],
            a_values[:nup],
            b_2d_values[0],
            self.C[..., 0],
        )
        self.sum_jC[nup:] += np.einsum(
            "inIk,jnIl,ijnm,Iklm->in",
            a_values[nup:],
            a_values[nup:],
            b_2d_values[2],
            self.C[..., 2],
        )

        val = self.sum_jC.sum(axis=0)
        self.val = val

        return (np.ones(len(val)), val)

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        newconfigs = configs.copy()
        newconfigs.move(e, epos, np.ones(len(epos.configs), dtype=bool))
        self.recompute(newconfigs)

    def value(self):
        val = self.val

        return (np.ones(len(val)), val)

    def single_e_partial(self, configs, e, epos, a_values):
        nup = int(self._mol.nelec[0])
        ndown = int(self._mol.nelec[1])
        sep = nup - int(e < nup)
        not_e = np.arange(self._nelec) != e

        nconf = configs.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])

        de = configs.dist.dist_i(configs.configs[:, not_e], epos)
        re = np.linalg.norm(de, axis=-1)

        di_e = configs.dist.dist_i(self._mol.atom_coords(), epos)
        ri_e = np.linalg.norm(di_e, axis=-1)

        ae = np.zeros((nconf, self._mol.natm, len(self.a_basis)))
        for i, a in enumerate(self.a_basis):
            ae[:, :, i] = a.value(di_e, ri_e)

        b_values = np.zeros((self._nelec - 1, nconf, len(self.b_basis)))

        for i, b in enumerate(self.b_basis):
            # swap axes: nconf and nelec. for now doing it here.
            b_values[:, :, i] = b.value(de, re).swapaxes(0, 1)

        na, nb = len(self.a_basis), len(self.b_basis)
        e_partial = np.zeros(nconf)
        e_partial += np.einsum(
            "nIk,jnIl,jnm,Iklm->n",
            ae,
            a_values[:sep],
            b_values[:sep],
            self.C[..., edown],
        )
        e_partial += np.einsum(
            "nIk,jnIl,jnm,Iklm->n",
            ae,
            a_values[sep:],
            b_values[sep:],
            self.C[..., edown + 1],
        )
        return e_partial

    def testvalue(self, e, epos, mask=None):
        configs = self._configscurrent
        nconf, nelec = configs.configs.shape[:2]
        edown = int(e >= self._mol.nelec[0])

        nup = int(self._mol.nelec[0])
        ndown = int(self._mol.nelec[1])
        sep = nup - int(e < nup)
        not_e = np.arange(self._nelec) != e

        di = np.stack(
            [
                configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, e_, :])
                for e_ in np.arange(nelec)[not_e]
            ],
            axis=0,
        )

        ri = np.linalg.norm(di, axis=-1)

        a_values = np.zeros((self._nelec - 1, nconf, self._mol.natm, len(self.a_basis)))

        for i, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_values[:, :, :, i] = a.value(di, ri)

        e_partial_old = self.single_e_partial(configs, e, configs.configs[:, e], a_values)
        e_partial_new = self.single_e_partial(configs, e, epos.configs, a_values)

        self.val = np.exp(2 * e_partial_new - 2 * e_partial_old)
        return self.val, None
