import numpy as np
from pyqmc.slater import sherman_morrison_row, get_complex_phase


def sherman_morrison_ms(e, inv, vec):
    ratio = np.einsum("idj,idj->id", vec, inv[:, :, :, e])
    tmp = np.einsum("edk,edkj->edj", vec, inv)
    invnew = (
        inv
        - np.einsum("kdi,kdj->kdij", inv[:, :, :, e], tmp)
        / ratio[:, :, np.newaxis, np.newaxis]
    )
    invnew[:, :, :, e] = inv[:, :, :, e] / ratio[:, :, np.newaxis]
    return ratio, invnew


def binary_to_occ(S, ncore):
    """
  Converts the binary cistring for a given determinant
  to occupation values for molecular orbitals within
  the determinant.
  """
    occup = [int(i) for i in range(ncore)]
    occup += [int(i + ncore) for i, c in enumerate(reversed(S)) if c == "1"]
    max_orb = max(occup)
    return (occup, max_orb)


class MultiSlater:
    """
    A multi-determinant wave function object initialized
    via an SCF calculation. Methods and structure are very similar
    to the PySCFSlaterUHF class.

    How to use with hci

    .. code-block:: python

        cisolver = pyscf.hci.SCI(mol)
        cisolver.select_cutoff=0.1
        nmo = mf.mo_coeff.shape[1]
        nelec = mol.nelec
        h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
        h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
        e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
        cisolver.ci = civec[0]
        wf = pyqmc.multislater.MultiSlater(mol, mf, cisolver, tol=0.1)


    """

    def __init__(self, mol, mf, mc, tol=None, freeze_orb=None):
        self.tol = -1 if tol is None else tol
        self.parameters = {}
        self._mol = mol
        if hasattr(mc, "nelecas"):
            # In case nelecas overrode the information from the molecule object.
            self._nelec = (mc.nelecas[0] + mc.ncore, mc.nelecas[1] + mc.ncore)
        else:
            self._nelec = mol.nelec
        self._copy_ci(mc)
        mo_coeff = mc.mo_coeff if hasattr(mc, "mo_coeff") else mf.mo_coeff
        mo_cutoff_alpha = np.max(self._det_occup[0]) + 1
        mo_cutoff_beta = np.max(self._det_occup[1]) + 1

        if len(mo_coeff.shape) == 3:
            self.parameters["mo_coeff_alpha"] = mo_coeff[0][:, :mo_cutoff_alpha]
            self.parameters["mo_coeff_beta"] = mo_coeff[1][:, :mo_cutoff_beta]
        else:
            self.parameters["mo_coeff_alpha"] = mo_coeff[:, :mo_cutoff_alpha]
            self.parameters["mo_coeff_beta"] = mo_coeff[:, :mo_cutoff_beta]
        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")
        self.pbc_str = "PBC" if hasattr(mol, "a") else ""
        self.iscomplex = bool(sum(map(np.iscomplexobj, self.parameters.values())))
        self.get_phase = get_complex_phase if self.iscomplex else np.sign
        self.freeze_orb = [[], []] if freeze_orb is None else freeze_orb

    def _copy_ci(self, mc):
        """       
        Copies over determinant coefficients and MO occupations
        for a multi-configuration calculation mc.
        """
        from pyscf import fci

        ncore = mc.ncore if hasattr(mc, "ncore") else 0

        # find multi slater determinant occupation
        if hasattr(mc, "_strs"):
            # if this is a HCI object, it will have _strs
            bigcis = np.abs(mc.ci > self.tol)
            nstrs = int(mc._strs.shape[1] / 2)
            # old code for single strings.
            # deters = [(c,bin(s[0]), bin(s[1])) for c, s in zip(mc.ci[bigcis],mc._strs[bigcis,:])]
            deters = []
            # In pyscf, the first n/2 strings represent the up determinant and the second
            # represent the down determinant.
            for c, s in zip(mc.ci[bigcis], mc._strs[bigcis, :]):
                s1 = "".join([str(bin(p)).replace("0b", "") for p in s[0:nstrs]])
                s2 = "".join([str(bin(p)).replace("0b", "") for p in s[nstrs:]])
                deters.append((c, s1, s2))
        else:
            deters = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)

        # Create map and occupation objects
        detwt = []
        map_dets = [[], []]
        occup = [[], []]
        for x in deters:
            if np.abs(x[0]) > self.tol:
                detwt.append(x[0])
                alpha_occ, __ = binary_to_occ(x[1], ncore)
                beta_occ, __ = binary_to_occ(x[2], ncore)
                if alpha_occ not in occup[0]:
                    map_dets[0].append(len(occup[0]))
                    occup[0].append(alpha_occ)
                else:
                    map_dets[0].append(occup[0].index(alpha_occ))

                if beta_occ not in occup[1]:
                    map_dets[1].append(len(occup[1]))
                    occup[1].append(beta_occ)
                else:
                    map_dets[1].append(occup[1].index(beta_occ))

        self.parameters["det_coeff"] = np.array(detwt)
        self._det_occup = occup  # Spin, [Ndet_up_unique, Ndet_dn_unique]
        self._det_map = np.array(map_dets)  # Spin, N_det

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""

        nconf, nelec, ndim = configs.configs.shape
        mycoords = configs.configs.reshape((nconf * nelec, ndim))
        ao = np.real_if_close(
            self._mol.eval_gto(self.pbc_str + "GTOval_sph", mycoords).reshape(
                (nconf, nelec, -1)
            ),
            tol=1e4,
        )

        self._aovals = ao
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            mo = ao[:, self._nelec[0] * s : self._nelec[0] + self._nelec[1] * s, :].dot(
                self.parameters[self._coefflookup[s]]
            )
            mo_vals = np.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            self._dets.append(
                np.array(np.linalg.slogdet(mo_vals))
            )  # Spin, (sign, val), nconf, [ndet_up, ndet_dn]
            self._inverse.append(
                np.linalg.inv(mo_vals)
            )  # spin, Nconf, [ndet_up, ndet_dn], nelec, nelec
        return self.value()

    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        # MAY want to vectorize later if it really hangs here, shouldn't!

        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        ao = np.real_if_close(
            self._mol.eval_gto(self.pbc_str + "GTOval_sph", epos.configs), tol=1e4
        )
        self._aovals[:, e, :] = ao
        mo = ao.dot(self.parameters[self._coefflookup[s]])

        mo_vals = mo[:, self._det_occup[s]]
        det_ratio, self._inverse[s][mask, :, :, :] = sherman_morrison_ms(
            eeff, self._inverse[s][mask, :, :, :], mo_vals[mask, :]
        )

        self._updateval(det_ratio, s, mask)

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        wf_val = 0
        wf_sign = 0

        wf_val = np.einsum(
            "id,di->i",
            self.parameters["det_coeff"][np.newaxis, :],
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * np.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
            ),
        )

        wf_sign = self.get_phase(wf_val)
        wf_val = np.log(np.abs(wf_val))
        return wf_sign, wf_val

    def _updateval(self, ratio, s, mask):
        self._dets[s][0, mask, :] *= self.get_phase(ratio)
        self._dets[s][1, mask, :] += np.log(np.abs(ratio))

    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin
        if mask is None:
            mask = [True] * vec.shape[0]

        ratios = np.einsum(
            "i...dj,idj...->i...d",
            vec,
            self._inverse[s][mask][..., e - s * self._nelec[0]],
        )
        det_array = (
            self._dets[0][0, :, self._det_map[0]][:, mask]
            * self._dets[1][0, :, self._det_map[1]][:, mask]
            * np.exp(
                self._dets[0][1, :, self._det_map[0]][:, mask]
                + self._dets[1][1, :, self._det_map[1]][:, mask]
            )
        )
        numer = np.einsum(
            "i...d,d,di->i...",
            ratios[..., self._det_map[s]],
            self.parameters["det_coeff"],
            det_array,
        )

        curr_val = self.value()
        denom = curr_val[0][mask] * np.exp(curr_val[1][mask])
        if len(numer.shape) == 2:
            denom = denom[:, np.newaxis]
        return numer / denom

    def _testcol(self, det, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i 
        of spin s in determinant det"""

        return np.einsum(
            "ij...,ij->i...", vec, self._inverse[s][:, det, i, :], optimize="greedy"
        )

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = np.real_if_close(
            self._mol.eval_gto("GTOval_sph_deriv1", epos.configs), tol=1e4
        )
        mograd = aograd.dot(self.parameters[self._coefflookup[s]])
        mograd_vals = mograd[:, :, self._det_occup[s]]

        ratios = np.asarray([self._testrow(e, x) for x in mograd_vals])
        return ratios[1:] / ratios[:1]

    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        ao = np.real_if_close(
            self._mol.eval_gto(self.pbc_str + "GTOval_sph_deriv2", epos.configs)[
                [0, 4, 7, 9]
            ],
            tol=1e4,
        )
        molap = np.dot(
            [ao[0], ao[1:].sum(axis=0)], self.parameters[self._coefflookup[s]]
        )
        molap_vals = self._testrow(e, molap[1][:, self._det_occup[s]])
        testvalue = self._testrow(e, molap[0][:, self._det_occup[s]])

        return molap_vals / testvalue

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = np.real_if_close(
            self._mol.eval_gto(self.pbc_str + "GTOval_sph_deriv2", epos.configs)[
                [0, 1, 2, 3, 4, 7, 9]
            ],
            tol=1e4,
        )
        ao = np.concatenate([ao[0:4], ao[4:].sum(axis=0, keepdims=True)])
        mo = np.dot(ao, self.parameters[self._coefflookup[s]])
        mo_vals = mo[:, :, self._det_occup[s]]
        ratios = np.asarray([self._testrow(e, x) for x in mo_vals])
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
        mo_vals = mo[..., self._det_occup[s]]
        return self._testrow(e, mo_vals, mask)

    def testvalue_many(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos for each electron"""
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
            mo_vals = mo[..., self._det_occup[spin]]
            ratios[:, ind] = self._testrow(e[ind], mo_vals, mask, spin=spin)

        return ratios

    def pgradient(self):
        r"""Compute the parameter gradient of Psi. 
        Returns $$d_p \Psi/\Psi$$ as a dictionary of numpy arrays,
        which correspond to the parameter dictionary."""
        d = {}

        # Det coeff
        det_coeff_grad = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * np.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
            )
        )

        curr_val = self.value()
        d["det_coeff"] = (
            det_coeff_grad.T / (curr_val[0] * np.exp(curr_val[1]))[:, np.newaxis]
        )

        # Mo_coeff, adapted from SlaterUHF
        for parm in ["mo_coeff_alpha", "mo_coeff_beta"]:
            s = 0
            if "beta" in parm:
                s = 1

            ao = self._aovals[
                :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]
            pgrad_shape = (ao.shape[0],) + self.parameters[parm].shape
            pgrad = np.zeros(pgrad_shape)

            largest_mo = np.max(np.ravel(self._det_occup[s]))
            for i in range(largest_mo + 1):  # MO loop
                if i not in self.freeze_orb[s]:
                    for det in range(self.parameters["det_coeff"].shape[0]):  # Det loop
                        if (
                            i in self._det_occup[s][self._det_map[s][det]]
                        ):  # Check if MO in det
                            col = self._det_occup[s][self._det_map[s][det]].index(i)
                            pgrad[:, :, i] += (
                                self.parameters["det_coeff"][det]
                                * d["det_coeff"][:, det, np.newaxis]
                                * self._testcol(self._det_map[s][det], col, s, ao)
                            )
            d[parm] = np.array(pgrad)
        return d
