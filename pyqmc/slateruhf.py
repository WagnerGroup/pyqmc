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

    def __init__(self, mol, mf, twist=[0,0,0]):
        """
        Inputs:
          mol:
          mf:
          twist: (3,) array-like. k=pi*twist, real-valued twists are integer
        """
        self.occ = np.asarray(mf.mo_occ > 0.9)
        self.parameters = {}
        #self.twist = np.array(twist)
        if np.linalg.norm(twist)==0:
            self.get_twist_phase = lambda wrap : 1
            self.get_full_phase = lambda wrap, i0, i1 : 1
        else:
            assert hasattr(mol, "a"), "twist can only be nonzero for a periodic system"
            if (np.mod(twist, 1)==0).all(): # real kpt
                get_twist = lambda wrap : np.prod((-1)**np.dot(wrap, twist), axis=1)
            else: 
                get_twist = lambda wrap : np.prod(
                        np.exp(1j*np.pi*np.dot(wrap, twist)), axis=1
                    )
            def get_full_twist(configs, ind0, ind1):
                return np.prod(get_twist(configs.wrap[:, ind0 : ind1, :]), axis=1)
            self.get_twist_phase = get_twist
            self.get_full_phase = get_full_twist
            

        # Determine if we're initializing from an RHF or UHF object...
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

        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")
        self._mol = mol
        self._nelec = tuple(mol.nelec)
        self.pbc_str = "PBC" if hasattr(mol, "a") else ""

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        nconf, nelec, ndim = configs.configs.shape
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
            phase *= self.get_full_phase(
                configs, s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            ) 
            self._dets.append((phase,mag))
            self._inverse.append(np.linalg.inv(mo))
            # Apply twist to phase
        

        return self.value()

    def updateinternals(self, e, epos, wrap=None, mask=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array 
        which allows us to update only certain walkers"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.shape[0]
        eeff = e - s * self._nelec[0]
        ao = self._mol.eval_gto(self.pbc_str + "GTOval_sph", epos)
        mo = ao.dot(self.parameters[self._coefflookup[s]])
        ratio, self._inverse[s][mask, :, :] = sherman_morrison_row(
            eeff, self._inverse[s][mask, :, :], mo[mask, :]
        )
        ratio *= self.get_twist_phase(wrap)
        self._updateval(ratio, s, mask)

    ### not state-changing functions

    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        return self._dets[0][0] * self._dets[1][0], self._dets[0][1] + self._dets[1][1]

    def _updateval(self, ratio, s, mask):
        self._dets[s][0][mask] *= np.sign(ratio)  # will not work for complex!
        self._dets[s][1][mask] += np.log(np.abs(ratio))

    def _testrow(self, e, vec):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0])
        ratio = np.einsum(
            "ij,ij->i", vec, self._inverse[s][:, :, e - s * self._nelec[0]]
        )
        return ratio

    def _testcol(self, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i"""
        ratio = np.einsum("ij,ij->i", vec, self._inverse[s][:, i, :])
        return ratio

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self._mol.eval_gto(self.pbc_str + "GTOval_ip_sph", epos)
        mograd = aograd.dot(self.parameters[self._coefflookup[s]])
        ratios = [self._testrow(e, x) for x in mograd]
        return np.asarray(ratios) / self.testvalue(e, epos)[np.newaxis, :]

    def laplacian(self, e, epos):
        """ Compute the laplacian Psi/ Psi. """
        s = int(e >= self._nelec[0])
        # aograd=self._mol.eval_gto(self.pbc_str+'GTOval_sph_deriv2',epos)
        aolap = np.sum(
            self._mol.eval_gto(self.pbc_str + "GTOval_sph_deriv2", epos)[[4, 7, 9]],
            axis=0,
        )
        molap = aolap.dot(self.parameters[self._coefflookup[s]])
        ratios = self._testrow(e, molap)
        return ratios / self.testvalue(e, epos)

    def testvalue(self, e, epos, wrap=[0,0,0]):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self._mol.eval_gto(self.pbc_str + "GTOval_sph", epos)
        mo = ao.dot(self.parameters[self._coefflookup[s]])
        return self._testrow(e, mo) * self.get_twist_phase(wrap)

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


def test():
    from pyscf import lib, gto, scf
    import pyqmc.testwf as testwf

    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
    for mf in [scf.RHF(mol).run(), scf.ROHF(mol).run(), scf.UHF(mol).run()]:
        print("")
        nconf = 10
        nelec = np.sum(mol.nelec)
        slater = PySCFSlaterUHF(mol, mf)
        configs = np.random.randn(nconf, nelec, 3)
        print("testing internals:", testwf.test_updateinternals(slater, configs))
        for delta in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            print(
                "delta",
                delta,
                "Testing gradient",
                testwf.test_wf_gradient(slater, configs, delta=delta),
            )
            print(
                "delta",
                delta,
                "Testing laplacian",
                testwf.test_wf_laplacian(slater, configs, delta=delta),
            )
            print(
                "delta",
                delta,
                "Testing pgradient",
                testwf.test_wf_pgradient(slater, configs, delta=delta),
            )


if __name__ == "__main__":
    test()
