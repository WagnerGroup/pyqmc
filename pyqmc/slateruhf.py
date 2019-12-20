import numpy as np


def sherman_morrison_row(e, inv, vec):
    ratio = np.einsum("ij,ij->i", vec, inv[:, :, e])
    tmp = np.einsum("ek,ekj->ej", vec, inv)
    invnew = (
        inv
        - np.einsum("ki,kj->kij", inv[:, :, e], tmp) / ratio[:, np.newaxis, np.newaxis]
    )
    invnew[:, :, e] = inv[:, :, e] / ratio[:, np.newaxis]
    return ratio, invnew


class PySCFSlaterUHF:
    """A wave function object has a state defined by a reference configuration of electrons.
    The functions recompute() and updateinternals() change the state of the object, and 
    the rest compute and return values from that state. """

    def __init__(self, mol, mf, twist=[0, 0, 0]):
        """
        Inputs:
          mol:
          mf:
          twist: (3,) array-like. k=pi*twist, real-valued twists are integer
        """
        self.occ = np.asarray(mf.mo_occ) > 0.9
        self.parameters = {}
        self.real_tol = 1e4
        if np.linalg.norm(twist) == 0:
            self.single_twist = lambda e, c: 1
            self.single_twist_mask = lambda e, c, m: 1
            self.all_twist = lambda c, i0, i1: 1
        else:
            assert hasattr(mol, "a"), "twist can only be nonzero for a periodic system"
            if (np.abs(twist - np.rint(twist)) < 1e-14).all():  # real kpt
                get_twist = lambda wrap: (-1) ** np.dot(wrap, twist)
                print("real kpt", twist)
            else:
                get_twist = lambda wrap: np.exp(1j * np.pi * np.dot(wrap, twist))
                print("cplx kpt", twist - np.array([1, 0, 1]), np.mod(twist, 1.0))

            def get_full_twist(c, i0, i1):
                self.wrap[:, i0:i1] = c.wrap[:, i0:i1]
                return np.prod(get_twist(c.wrap[:, i0:i1]), axis=1)

            def get_single_twist(e, c, mask=None):
                twist = get_twist(c.wrap - self.wrap[:, e])
                if mask is not None:
                    twist = twist[mask]
                return twist

            self.single_twist = lambda e, c: get_twist(c.wrap - self.wrap[:, e])
            self.single_twist_mask = lambda e, c, m: get_twist(
                c.wrap[m] - self.wrap[m, e]
            )
            self.all_twist = lambda c, i0, i1: np.prod(
                get_twist(c.wrap[:, i0:i1, :]), axis=1
            )

        # Determine if we're initializing from an RHF or UHF object.
        if hasattr(mf, "kpts"):
            kind = np.where(
                np.linalg.norm(
                    np.dot(mf.kpts, mol.a.T) - np.array(twist) * np.pi, axis=1
                )
                < 1e-12
            )[0][0]
            if len(np.asarray(mf.mo_occ).shape) == 3:
                self.parameters["mo_coeff_alpha"] = np.real_if_close(
                    mf.mo_coeff[0][kind][:, self.occ[0, kind]], tol=self.real_tol
                )
                self.parameters["mo_coeff_beta"] = np.real_if_close(
                    mf.mo_coeff[1][kind][:, self.occ[1, kind]], tol=self.real_tol
                )
            else:
                self.parameters["mo_coeff_alpha"] = np.real_if_close(
                    mf.mo_coeff[kind][:, np.asarray(mf.mo_occ[kind] > 0.9)],
                    tol=self.real_tol,
                )
                self.parameters["mo_coeff_beta"] = np.real_if_close(
                    mf.mo_coeff[kind][:, np.asarray(mf.mo_occ[kind] > 1.1)],
                    tol=self.real_tol,
                )

        else:
            if len(mf.mo_occ.shape) == 2:
                self.parameters["mo_coeff_alpha"] = mf.mo_coeff[0][:, self.occ[0]]
                self.parameters["mo_coeff_beta"] = mf.mo_coeff[1][:, self.occ[1]]
            else:
                self.parameters["mo_coeff_alpha"] = mf.mo_coeff[
                    :, np.asarray(mf.mo_occ > 0.9)
                ]
                self.parameters["mo_coeff_beta"] = mf.mo_coeff[
                    :, np.asarray(mf.mo_occ > 1.1)
                ]

        if (
            np.iscomplexobj(
                np.concatenate([p.ravel() for p in self.parameters.values()])
            )
            or np.linalg.norm(twist) != 0
        ):
            self.get_phase = lambda x: np.exp(2j * np.pi * np.angle(x))
        else:
            self.get_phase = np.sign
        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")
        self._mol = mol
        self._nelec = tuple(mol.nelec)
        self.pbc_str = "PBC" if hasattr(mol, "a") else ""

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        nconf, nelec, ndim = configs.configs.shape
        self.wrap = np.zeros((configs.configs.shape))  # only needed for PBC
        mycoords = configs.configs.reshape((nconf * nelec, ndim))
        ao = self._mol.eval_gto(self.pbc_str + "GTOval_sph", mycoords).reshape(
            (nconf, nelec, -1)
        )

        self._aovals = ao
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            if s == 0:
                mo = ao[:, 0 : self._nelec[0], :].dot(
                    self.parameters[self._coefflookup[s]]
                )
            else:
                mo = ao[:, self._nelec[0] : self._nelec[0] + self._nelec[1], :].dot(
                    self.parameters[self._coefflookup[s]]
                )
            # This could be done faster; we are doubling our effort here.
            phase, mag = np.linalg.slogdet(mo)
            phase *= self.all_twist(
                configs, s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            )
            self._dets.append((phase, mag))
            self._inverse.append(np.linalg.inv(mo))
            # Apply twist to phase

        return self.value()

    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        ao = self._mol.eval_gto(self.pbc_str + "GTOval_sph", epos.configs)
        self._aovals[:, e, :] = ao
        mo = ao.dot(self.parameters[self._coefflookup[s]])
        ratio, self._inverse[s][mask, :, :] = sherman_morrison_row(
            eeff, self._inverse[s][mask, :, :], mo[mask, :]
        )
        ratio *= self.single_twist_mask(e, epos, mask)
        self._updateval(ratio, s, mask)

    ### not state-changing functions

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        return self._dets[0][0] * self._dets[1][0], self._dets[0][1] + self._dets[1][1]

    def _updateval(self, ratio, s, mask):
        self._dets[s][0][mask] *= self.get_phase(ratio)  # will not work for complex!
        self._dets[s][1][mask] += np.log(np.abs(ratio))

    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        if spin is None:
            s = int(e >= self._nelec[0])
        else:
            s = spin

        if mask is None:
            return np.einsum(
                "i...j,ij...->i...", vec, self._inverse[s][:, :, e - s * self._nelec[0]]
            )

        return np.einsum(
            "i...j,ij...->i...",
            vec,
            self._inverse[s][mask][:, :, e - s * self._nelec[0]],
        )

    def _testcol(self, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i"""
        ratio = np.einsum("ij,ij->i", vec, self._inverse[s][:, i, :])
        return ratio

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self._mol.eval_gto("GTOval_sph_deriv1", epos.configs)
        mograd = aograd.dot(self.parameters[self._coefflookup[s]])
        ratios = np.asarray([self._testrow(e, x) for x in mograd])
        return ratios[1:] / ratios[:1]

    def laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self._mol.eval_gto(self.pbc_str + "GTOval_sph_deriv2", epos.configs)[
            [0, 4, 7, 9]
        ]
        mo = np.dot([ao[0], ao[1:].sum(axis=0)], self.parameters[self._coefflookup[s]])
        ratios = self._testrow(e, mo[1])
        testvalue = self._testrow(e, mo[0])
        return ratios / testvalue

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self._mol.eval_gto(self.pbc_str + "GTOval_sph_deriv2", epos.configs)[
            [0, 1, 2, 3, 4, 7, 9]
        ]
        ao = np.concatenate([ao[0:4], ao[4:].sum(axis=0, keepdims=True)])
        mo = np.dot(ao, self.parameters[self._coefflookup[s]])
        ratios = np.asarray([self._testrow(e, x) for x in mo])
        return ratios[1:-1] / ratios[:1], ratios[-1] / ratios[0]

    def testvalue(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eposmask = epos.configs[mask]
        if len(eposmask) == 0:
            return np.zeros(eposmask.shape[:2])
        ao = self._mol.eval_gto(
            self.pbc_str + "GTOval_sph", eposmask.reshape((-1, 3))
        ).reshape((*eposmask.shape[:-1], -1))
        mo = ao.dot(self.parameters[self._coefflookup[s]])
        a = self._testrow(e, mo, mask)
        b = self.single_twist_mask(e, epos, mask)
        return a * b

    def testvalue_many(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        an electron's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eposmask = epos.configs[mask]
        if len(eposmask) == 0:
            return np.zeros(eposmask.shape[:2])
        ao = self._mol.eval_gto(
            self.pbc_str + "GTOval_sph", eposmask.reshape((-1, 3))
        ).reshape((*eposmask.shape[:-1], -1))

        ratios = np.zeros((epos.configs.shape[0], e.shape[0]))
        for spin in [0, 1]:
            ind = s == spin
            mo = ao.dot(self.parameters[self._coefflookup[spin]])
            ratios[:, ind] = self._testrow(e[ind], mo, spin=spin)
        return ratios

    def pgradient(self):
        """Compute the parameter gradient of Psi. 
        Returns d_p \Psi/\Psi as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.
        """
        d = {}

        for parm in self.parameters:
            s = 0
            if "beta" in parm:
                s = 1
            # Get AOs for our spin channel only
            ao = self._aovals[
                :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]  # (config, electron, ao)

            pgrad_shape = (ao.shape[0],) + self.parameters[parm].shape
            pgrad = np.zeros(pgrad_shape)
            # Compute derivatives w.r.t MO coefficients
            for i in range(self._nelec[s]):  # MO loop
                for j in range(ao.shape[2]):  # AO loop
                    vec = ao[:, :, j]
                    pgrad[:, j, i] = self._testcol(i, s, vec)  # nconfig
            d[parm] = np.array(pgrad)  # Returns config, coeff
        return d
