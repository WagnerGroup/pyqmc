import numpy as np
import pyqmc.gpu as gpu
import pyqmc.determinant_tools as determinant_tools
import pyqmc.orbitals
import warnings


def sherman_morrison_row(e, inv, vec):
    tmp = np.einsum("ek,ekj->ej", vec, inv)
    ratio = tmp[:, e]
    inv_ratio = inv[:, :, e] / ratio[:, np.newaxis]
    invnew = inv - np.einsum("ki,kj->kij", inv_ratio, tmp)
    invnew[:, :, e] = inv_ratio
    return ratio, invnew


def get_complex_phase(x):
    return x / np.abs(x)


class JoinParameters:
    """
    This class provides a dict-like interface that actually references
    other dictionaries in the background.
    If keys collide, then the first dictionary that matches the key will be returned.
    However, some bad things may happen if you have colliding keys.
    """

    def __init__(self, dicts):
        self.data = {}
        self.data = dicts

    def find_i(self, idx):
        for i, d in enumerate(self.data):
            if idx in d:
                return i

    def __setitem__(self, idx, value):
        i = self.find_i(idx)
        self.data[i][idx] = value

    def __getitem__(self, idx):
        i = self.find_i(idx)
        return self.data[i][idx]

    def __delitem__(self, idx):
        i = self.find_i(idx)
        del self.data[i][idx]

    def __iter__(self):
        for d in self.data:
            yield from d.keys()

    def __len__(self):
        return sum(len(i) for i in self.data)

    def items(self):
        for d in self.data:
            yield from d.items()

    def __repr__(self):
        return self.data.__repr__()

    def keys(self):
        for d in self.data:
            yield from d.keys()

    def values(self):
        for d in self.data:
            yield from d.values()


def sherman_morrison_ms(e, inv, vec):
    tmp = np.einsum("edk,edkj->edj", vec, inv)
    ratio = tmp[:, :, e]
    inv_ratio = inv[:, :, :, e] / ratio[:, :, np.newaxis]
    invnew = inv - np.einsum("kdi,kdj->kdij", inv_ratio, tmp)
    invnew[:, :, :, e] = inv_ratio
    return ratio, invnew


class Slater:
    """
    A multi-determinant wave function object initialized
    via an SCF calculation.

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

    def __init__(self, mol, mf, mc=None, tol=None, twist=None, determinants=None):
        """
        determinants should be a list of tuples, for example
        [ (1.0, [0,1],[0,1]),
          (-0.2, [0,2],[0,2]) ]
        would be a two-determinant wave function with a doubles excitation in the second one.

        determinants overrides any information in mc, if passed.
        """
        self.tol = -1 if tol is None else tol
        self._mol = mol
        if hasattr(mc, "nelecas"):
            # In case nelecas overrode the information from the molecule object.
            ncore = mc.ncore 
            if not hasattr(ncore, "__len__"):
                ncore = [ncore, ncore]
            self._nelec = (mc.nelecas[0] + ncore[0], mc.nelecas[1] + ncore[1])
        else:
            self._nelec = mol.nelec

        self.myparameters = {}
        (
            self.myparameters["det_coeff"],
            self._det_occup,
            self._det_map,
            self.orbitals,
        ) = pyqmc.orbitals.choose_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=determinants, tol=self.tol
        )

        self.parameters = JoinParameters([self.myparameters, self.orbitals.parameters])

        self.iscomplex = self.orbitals.iscomplex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        self.dtype = complex if self.iscomplex else float
        self.get_phase = get_complex_phase if self.iscomplex else gpu.cp.sign

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""

        nconf, nelec, ndim = configs.configs.shape
        aos = self.orbitals.aos("GTOval_sph", configs)
        self._aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(self._aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            self._dets.append(
                gpu.cp.asarray(np.linalg.slogdet(mo_vals))
            )  # Spin, (sign, val), nconf, [ndet_up, ndet_dn]

            is_zero = np.sum(np.abs(self._dets[s][0]) < 1e-16)
            compute = np.isfinite(self._dets[s][1])
            if is_zero > 0:
                warnings.warn(
                    f"A wave function is zero. Found this proportion: {is_zero/nconf}"
                )
                # print(configs.configs[])
                print(f"zero {is_zero/np.prod(compute.shape)}")
            self._inverse.append(gpu.cp.zeros(mo_vals.shape, dtype=mo_vals.dtype))
            for d in range(compute.shape[1]):
                self._inverse[s][compute[:, d], d, :, :] = gpu.cp.linalg.inv(
                    mo_vals[compute[:, d], d, :, :]
                )
            # spin, Nconf, [ndet_up, ndet_dn], nelec, nelec
        return self.value()

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array
        which allows us to update only certain walkers"""

        s = int(e >= self._nelec[0])
        if mask is None:
            mask = np.ones(epos.configs.shape[0], dtype=bool)
        is_zero = np.sum(np.isinf(self._dets[s][1]))
        if is_zero:
            warnings.warn(
                "Found a zero in the wave function. Recomputing everything. This should not happen often."
            )
            self.recompute(configs)
            return

        eeff = e - s * self._nelec[0]
        if saved_values is None:
            ao = self.orbitals.aos("GTOval_sph", epos, mask)
            self._aovals[:, mask, e, :] = ao
            mo = self.orbitals.mos(ao, s)
        else:
            ao, mo = saved_values
            self._aovals[:, mask, e, :] = ao[:, mask]
            mo = mo[mask]
        mo_vals = mo[:, self._det_occup[s]]
        det_ratio, self._inverse[s][mask, :, :, :] = sherman_morrison_ms(
            eeff, self._inverse[s][mask, :, :, :], mo_vals
        )
        self._dets[s][0, mask, :] *= self.get_phase(det_ratio)
        self._dets[s][1, mask, :] += gpu.cp.log(gpu.cp.abs(det_ratio))

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]
        return determinant_tools.compute_value(
            updets, dndets, self.parameters["det_coeff"]
        )

    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin
        if mask is None:
            mask = [True] * vec.shape[0]

        ratios = gpu.cp.einsum(
            "i...dj,idj...->i...d",
            vec,
            self._inverse[s][mask][..., e - s * self._nelec[0]],
        )

        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, mask][:, self._det_map[0]]
            * self._dets[1][0, mask][:, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, mask][:, self._det_map[0]]
                + self._dets[1][1, mask][:, self._det_map[1]]
                - upref
                - dnref
            )
        ).T
        numer = gpu.cp.einsum(
            "i...d,d,di->i...",
            ratios[..., self._det_map[s]],
            self.parameters["det_coeff"],
            det_array,
        )
        denom = gpu.cp.einsum(
            "d,di->i...",
            self.parameters["det_coeff"],
            det_array,
        )

        if len(numer.shape) == 2:
            denom = denom[:, gpu.cp.newaxis]
        return numer / denom

    def _testrowderiv(self, e, vec, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin

        ratios = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            vec,
            self._inverse[s][..., e - s * self._nelec[0]],
        )

        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )
        numer = gpu.cp.einsum(
            "ei...d,d,di->ei...",
            ratios[..., self._det_map[s]],
            self.parameters["det_coeff"],
            det_array,
        )
        denom = gpu.cp.einsum(
            "d,di->i...",
            self.parameters["det_coeff"],
            det_array,
        )
        # curr_val = self.value()

        if len(numer.shape) == 3:
            denom = denom[gpu.cp.newaxis, :, gpu.cp.newaxis]
        return numer / denom

    def _testcol(self, det, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i
        of spin s in determinant det"""

        return gpu.cp.einsum(
            "ij...,ij->i...", vec, self._inverse[s][:, det, i, :], optimize="greedy"
        )

    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)

        mograd_vals = mograd[:, :, self._det_occup[s]]

        ratios = self._testrowderiv(e, mograd_vals)
        return gpu.asnumpy(ratios[1:] / ratios[0])

    def gradient_value(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)

        mograd_vals = mograd[:, :, self._det_occup[s]]

        ratios = gpu.asnumpy(self._testrowderiv(e, mograd_vals))
        derivatives = ratios[1:] / ratios[0]
        derivatives[~np.isfinite(derivatives)] = 0.0
        values = ratios[0]
        values[~np.isfinite(values)] = 1.0
        return derivatives, values, (aograd[:, 0], mograd[0])

    def laplacian(self, e, epos):
        """Compute the laplacian Psi/ Psi."""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao_val = ao[:, 0, :, :]
        ao_lap = gpu.cp.sum(ao[:, [4, 7, 9], :, :], axis=1)
        mos = gpu.cp.stack(
            [self.orbitals.mos(x, s)[..., self._det_occup[s]] for x in [ao_val, ao_lap]]
        )
        ratios = self._testrowderiv(e, mos)
        return gpu.asnumpy(ratios[1] / ratios[0])

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao = gpu.cp.concatenate(
            [ao[:, 0:4, ...], ao[:, [4, 7, 9], ...].sum(axis=1, keepdims=True)], axis=1
        )
        mo = self.orbitals.mos(ao, s)
        mo_vals = mo[:, :, self._det_occup[s]]
        ratios = self._testrowderiv(e, mo_vals)
        ratios = gpu.asnumpy(ratios / ratios[:1])
        return ratios[1:-1], ratios[-1]

    def testvalue(self, e, epos, mask=None):
        """return the ratio between the current wave function and the wave function if
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        mo = self.orbitals.mos(ao, s)
        mo_vals = mo[..., self._det_occup[s]]
        if len(epos.configs.shape) > 2:
            mo_vals = mo_vals.reshape(
                -1, epos.configs.shape[1], mo_vals.shape[1], mo_vals.shape[2]
            )
        return gpu.asnumpy(self._testrow(e, mo_vals, mask)), (ao, mo)

    def testvalue_many(self, e, epos, mask=None):
        """return the ratio between the current wave function and the wave function if
        electron e's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        ratios = gpu.cp.zeros((epos.configs.shape[0], e.shape[0]), dtype=self.dtype)
        for spin in [0, 1]:
            ind = s == spin
            mo = self.orbitals.mos(ao, spin)
            mo_vals = mo[..., self._det_occup[spin]]
            ratios[:, ind] = self._testrow(e[ind], mo_vals, mask, spin=spin)

        return gpu.asnumpy(ratios)

    def pgradient(self):
        """Compute the parameter gradient of Psi.
        Returns :math:`\partial_p \Psi/\Psi` as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.

        The wave function is given by ci Di, with an implicit sum

        We have two sets of parameters:

        Determinant coefficients:
        di psi/psi = Dui Ddi/psi

        Orbital coefficients:
        dj psi/psi = ci dj (Dui Ddi)/psi

        Let's suppose that j corresponds to an up orbital coefficient. Then
        dj (Dui Ddi) = (dj Dui)/Dui Dui Ddi/psi = (dj Dui)/Dui di psi/psi
        where di psi/psi is the derivative defined above.
        """
        d = {}

        # Det coeff
        curr_val = self.value()
        nonzero = curr_val[0] != 0.0

        # dets[spin][ (phase,log), configuration, determinant]
        dets = (
            self._dets[0][:, :, self._det_map[0]],
            self._dets[1][:, :, self._det_map[1]],
        )

        d["det_coeff"] = gpu.cp.zeros(dets[0].shape[1:], dtype=dets[0].dtype)
        d["det_coeff"][nonzero, :] = (
            dets[0][0, nonzero, :]
            * dets[1][0, nonzero, :]
            * gpu.cp.exp(
                dets[0][1, nonzero, :]
                + dets[1][1, nonzero, :]
                - gpu.cp.array(curr_val[1][nonzero, np.newaxis])
            )
            / gpu.cp.array(curr_val[0][nonzero, np.newaxis])
        )

        for s, parm in zip([0, 1], ["mo_coeff_alpha", "mo_coeff_beta"]):
            ao = self._aovals[
                :, :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]

            split, aos = self.orbitals.pgradient(ao, s)
            mos = gpu.cp.split(gpu.cp.arange(split[-1]), gpu.asnumpy(split).astype(int))
            # Compute dj Diu/Diu
            nao = aos[0].shape[-1]
            nconf = aos[0].shape[0]
            nmo = int(split[-1])
            deriv = gpu.cp.zeros(
                (len(self._det_occup[s]), nconf, nao, nmo), dtype=curr_val[0].dtype
            )
            for det, occ in enumerate(self._det_occup[s]):
                for ao, mo in zip(aos, mos):
                    for i in mo:
                        if i in occ:
                            col = occ.index(i)
                            deriv[det, :, :, i] = self._testcol(det, col, s, ao)

            # now we reduce over determinants
            d[parm] = gpu.cp.zeros(deriv.shape[1:], dtype=curr_val[0].dtype)
            for di, coeff in enumerate(self.parameters["det_coeff"]):
                whichdet = self._det_map[s][di]
                d[parm] += (
                    deriv[whichdet]
                    * coeff
                    * d["det_coeff"][:, di, np.newaxis, np.newaxis]
                )

        for k, v in d.items():
            d[k] = gpu.asnumpy(v)

        for k in list(d.keys()):
            if np.prod(d[k].shape) == 0:
                del d[k]
        return d
