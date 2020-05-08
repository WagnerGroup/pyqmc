import numpy as np
from pyqmc.multislater import sherman_morrison_ms
from pyqmc.slaterpbc import get_supercell_kpts
from pyqmc import pbc


class MultiSlaterPBC:
    """
    A multi-determinant wave function object initialized
    via an SCF calculation. Methods and structure are very similar
    to the PySCFSlaterUHF class.
    """

    def __init__(
        self, supercell, mf, tol=-1, detwt=(1,), occup=None, map_dets=None, twist=None
    ):
        """
        detwt: list of determinant weights
        occup: list (spin, det, dict{kind: occupation list})
        map_dets: list (spin, ndet) to identify which determinant of each spin to use (e.g. may use the same up-determinant in multiple products)
        """
        for attribute in ["original_cell", "S"]:
            if not hasattr(supercell, attribute):
                print('Warning: supercell is missing attribute "%s"' % attribute)
                print("setting original_cell=supercell and S=np.eye(3)")
                supercell.original_cell = supercell
                supercell.S = np.eye(3)

        assert occup is not None
        assert len(map_dets[0]) == len(detwt) and len(map_dets[1]) == len(detwt)
        self.parameters = {"det_coeff": np.array(detwt)}

        self.tol = tol
        self.real_tol = 1e4
        self._mol = supercell.original_cell
        self._nelec = tuple(int(sum(len(v) for v in o[0].values())) for o in occup)

        self.supercell = supercell
        if twist is None:
            twist = np.zeros(3)
        else:
            twist = np.dot(np.linalg.inv(supercell.a), np.mod(twist, 1.0)) * 2 * np.pi
        self._kpts = get_supercell_kpts(supercell) + twist
        kdiffs = mf.kpts[np.newaxis] - self._kpts[:, np.newaxis]
        self.kinds = np.nonzero(np.linalg.norm(kdiffs, axis=-1) < 1e-12)[1]
        self.nk = len(self._kpts)
        print("nk", self.nk, self.kinds)

        maxorb = [
            {
                k: np.amax([np.amax(list(d[k]) + [-1]) for d in o]) + 1
                for k in self.kinds
            }
            for o in occup
        ]

        self.param_split = {}
        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")
        for s, lookup in enumerate(self._coefflookup):
            mclist = []
            for kind in self.kinds:
                if len(mf.mo_coeff[0][0].shape) == 2:
                    mca = mf.mo_coeff[s][kind][:, : maxorb[s][kind]]
                else:
                    mca = mf.mo_coeff[kind][:, : maxorb[s][kind]]
                mca = np.real_if_close(mca, tol=self.real_tol)
                mclist.append(mca / np.sqrt(self.nk))
            self.param_split[lookup] = np.cumsum([m.shape[1] for m in mclist])
            self.parameters[lookup] = np.concatenate(mclist, axis=-1)

        # _det_occup: Spin, [(Ndet_up_unique, nup), (Ndet_dn_unique, ndn)]
        self._det_occup = [
            np.zeros((len(o), n), dtype=int) for o, n in zip(occup, self._nelec)
        ]
        for s, o in enumerate(occup):
            orb_inds = np.cumsum([0] + [maxorb[s][k] for k in self.kinds[:-1]])
            for d, det in enumerate(o):
                self._det_occup[s][d] = np.concatenate(
                    [np.array(det[k]) + ind for k, ind in zip(self.kinds, orb_inds)]
                )
        self._det_map = np.asarray(map_dets)  # Spin, N_det
        print("det_coeff", self.parameters["det_coeff"])
        print("_det_occup\n", self._det_occup)
        print("_det_map\n", self._det_map)

        self.iscomplex = bool(sum(map(np.iscomplexobj, self.parameters.values())))
        self.iscomplex = self.iscomplex or np.linalg.norm(self._kpts) > 1e-12
        self.dtype = complex if self.iscomplex else float
        if self.iscomplex:
            self.get_phase = lambda x: x / np.abs(x)
            self.get_wrapphase = lambda x: np.exp(1j * x)
        else:
            self.get_phase = np.sign
            self.get_wrapphase = lambda x: (-1) ** np.round(x / np.pi)

    def evaluate_orbitals(self, configs, mask=None, eval_str="PBCGTOval_sph"):
        mycoords = configs.configs
        configswrap = configs.wrap
        if mask is not None:
            mycoords = mycoords[mask]
            configswrap = configswrap[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        # wrap supercell positions into primitive cell
        prim_coords, prim_wrap = pbc.enforce_pbc(self._mol.lattice_vectors(), mycoords)
        configswrap = configswrap.reshape(prim_wrap.shape)
        wrap = prim_wrap + np.dot(configswrap, self.supercell.S)
        kdotR = np.linalg.multi_dot((self._kpts, self._mol.lattice_vectors().T, wrap.T))
        wrap_phase = self.get_wrapphase(kdotR)
        # evaluate AOs for all electron positions
        ao = self._mol.eval_gto(eval_str, prim_coords, kpts=self._kpts)
        ao = [ao[k] * wrap_phase[k][:, np.newaxis] for k in range(self.nk)]
        return ao

    def evaluate_mos(self, aos, s):
        l = self._coefflookup[s]
        p = np.split(self.parameters[l], self.param_split[l], axis=-1)
        mo = [np.dot(ao, p[k]) for k, ao in enumerate(aos)]
        mo = np.concatenate(mo, axis=-1)
        return mo[..., self._det_occup[s]]

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        nconf, nelec, ndim = configs.configs.shape
        aos = self.evaluate_orbitals(configs)
        aos = np.reshape(aos, (self.nk, nconf, nelec, -1))
        self._aovals = aos
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            i0, i1 = s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            mo_vals = self.evaluate_mos(aos[:, :, i0:i1], s).swapaxes(1, 2)
            self._dets.append(np.array(np.linalg.slogdet(mo_vals)))
            self._inverse.append(np.linalg.inv(mo_vals))
        return self.value()

    def updateinternals(self, e, epos, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        aos = self.evaluate_orbitals(epos)
        self._aovals[:, :, e, :] = np.asarray(aos)
        mo = self.evaluate_mos(aos, s)
        ratio, self._inverse[s][mask] = sherman_morrison_ms(
            eeff, self._inverse[s][mask], mo[mask, :]
        )
        self._updateval(ratio, s, mask)

    # identical to multislater.py
    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        wf_val = 0
        wf_sign = 0

        wf_val = np.einsum(
            "d,di->i",
            self.parameters["det_coeff"],
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

    # identical to slateruhf
    def _updateval(self, ratio, s, mask):
        self._dets[s][0, mask, :] *= self.get_phase(ratio)
        self._dets[s][1, mask, :] += np.log(np.abs(ratio))

    # identical to multislater.py
    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        if spin is None:
            s = int(e >= self._nelec[0])
        else:
            s = spin

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

    # identical to multislater.py
    def _testcol(self, det, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i 
        of spin s in determinant det"""

        ratio = np.einsum("ij...,ij->i...", vec, self._inverse[s][:, det, i, :])
        return ratio

    # identical to slaterpbc
    def testvalue(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        nmask = np.sum(mask)
        if nmask == 0:
            return np.zeros((0, epos.configs.shape[1]))
        ao = self.evaluate_orbitals(epos, mask=mask)
        mo = self.evaluate_mos(ao, s)
        mo = mo.reshape((nmask, *epos.configs.shape[1:-1], -1, self._nelec[s]))
        return self._testrow(e, mo, mask)

    # identical to slaterpbc
    def testvalue_many(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        nmask = np.sum(mask)
        if nmask == 0:
            return np.zeros((0, epos.configs.shape[1]))

        ao = self.evaluate_orbitals(epos, mask=mask)
        ratios = np.zeros((epos.configs.shape[0], e.shape[0]), dtype=self.dtype)
        for spin in [0, 1]:
            ind = s == spin
            mo = self.evaluate_mos(aos, spin)
            mo = mo.reshape(nmask, *epos.configs.shape[1:-1], -1, self._nelec[spin])
            ratios[:, ind] = self._testrow(e[ind], mo, mask=mask, spin=spin)
        return ratios

    # identical to slaterpbc
    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.evaluate_orbitals(epos, eval_str="PBCGTOval_sph_deriv1")
        mograd = self.evaluate_mos(aograd, s)
        ratios = np.asarray([self._testrow(e, x) for x in mograd])
        return ratios[1:] / ratios[:1]

    # identical to slaterpbc
    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        ao = self.evaluate_orbitals(epos, eval_str="PBCGTOval_sph_deriv2")
        aostack = [np.stack([ak[0], ak[[4, 7, 9]].sum(axis=0)], axis=0) for ak in ao]
        molap = self.evaluate_mos(aostack, s)
        ratios = np.asarray([self._testrow(e, x) for x in molap])
        return ratios[1] / ratios[0]

    # identical to slaterpbc
    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.evaluate_orbitals(epos, eval_str="PBCGTOval_sph_deriv2")
        aostack = [
            np.concatenate([ak[0:4], ak[[4, 7, 9]].sum(axis=0, keepdims=True)], axis=0)
            for ak in ao
        ]
        mo_vals = self.evaluate_mos(aostack, s)
        ratios = np.asarray([self._testrow(e, x) for x in mo_vals])
        return ratios[1:-1] / ratios[:1], ratios[-1] / ratios[0]

    def pgradient(self):
        r"""Compute the parameter gradient of Psi. 
        Returns :math:`\frac{\partial_p \Psi}{\Psi}` as a dictionary of numpy arrays,
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

        for s, parm in enumerate(["mo_coeff_alpha", "mo_coeff_beta"]):
            i0, i1 = s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            ao = self._aovals[:, :, i0:i1, :]  # nk, nconf, nelec, nao
            pgrad_shape = (ao.shape[1],) + self.parameters[parm].shape
            pgrad = np.zeros(pgrad_shape, dtype=self.dtype)
            split_sizes = np.diff([0] + list(self.param_split[parm]))
            k = np.repeat(np.arange(self.nk), split_sizes)

            largest_mo = np.max(np.ravel(self._det_occup[s]))
            for det, detwt in enumerate(self.parameters["det_coeff"]):  # Det loop
                mapdet = self._det_map[s][det]
                for col, i in enumerate(self._det_occup[s][mapdet]):
                    pgrad[:, :, i] += (
                        detwt
                        * d["det_coeff"][:, det, np.newaxis]
                        * self._testcol(mapdet, col, s, ao[k[i]])
                    )
            d[parm] = np.array(pgrad)
        return d
