import numpy as np
from pyqmc import pbc, slateruhf
from pyscf.pbc import scf


def get_supercell_kpts(supercell):
    Sinv = np.linalg.inv(supercell.S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, supercell.S.T)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    kptmesh = np.meshgrid(*[np.arange(*r) for r in xyz_range], indexing="ij")
    possible_kpts = np.dot(np.stack([x.ravel() for x in kptmesh]).T, Sinv)
    in_unit_box = (possible_kpts >= 0) * (possible_kpts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    reclatvec = np.linalg.inv(supercell.original_cell.lattice_vectors()).T * 2 * np.pi
    kpts = np.dot(possible_kpts[select], reclatvec)
    return kpts


def get_supercell(cell, S):
    """
    Inputs:
        cell: pyscf Cell object
        S: (3, 3) supercell matrix for QMC from cell defined by cell.a. In other words, the QMC calculation cell is qmc_cell = np.dot(S, cell.lattice_vectors()). For a 2x2x2 supercell, S is [[2, 0, 0], [0, 2, 0], [0, 0, 2]].
    """
    from pyscf.pbc import gto

    def get_supercell_copies(latvec, S):
        Sinv = np.linalg.inv(S).T
        u = [0, 1]
        unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
        unit_box_ = np.dot(unit_box, S)
        xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
        mesh = np.meshgrid(*[np.arange(*r) for r in xyz_range], indexing="ij")
        possible_pts = np.dot(np.stack([x.ravel() for x in mesh]).T, Sinv.T)
        in_unit_box = (possible_pts >= 0) * (possible_pts < 1 - 1e-12)
        select = np.where(np.all(in_unit_box, axis=1))[0]
        pts = np.linalg.multi_dot((possible_pts[select], S, latvec))
        return pts

    scale = np.abs(int(np.round(np.linalg.det(S))))
    superlattice = np.dot(S, cell.lattice_vectors())
    Rpts = get_supercell_copies(cell.lattice_vectors(), S)
    atom = []
    for (name, xyz) in cell._atom:
        atom.extend([(name, xyz + R) for R in Rpts])
    supercell = gto.Cell()
    supercell.a = superlattice
    supercell.atom = atom
    supercell.ecp = cell.ecp
    supercell.basis = cell.basis
    supercell.exp_to_discard = cell.exp_to_discard
    supercell.unit = "Bohr"
    supercell.spin = cell.spin * scale
    supercell.build()
    supercell.original_cell = cell
    supercell.S = S
    supercell.scale = scale
    return supercell


class PySCFSlaterPBC:
    """A wave function object has a state defined by a reference configuration of electrons.
    The functions recompute() and updateinternals() change the state of the object, and 
    the rest compute and return values from that state. """

    def __init__(self, supercell, mf, twist=None):
        """
        Inputs:
          supercell: object returned by get_supercell(cell, S)
          mf: scf object of primitive cell calculation. scf calculation must include k points that fold onto the gamma point of the supercell
          twist: (3,) array, twisted boundary condition in fractional coordinates, i.e. as coefficients of the reciprocal lattice vectors of the supercell. Integer values are equivalent to zero.
        """
        for attribute in ["original_cell", "S"]:
            if not hasattr(supercell, attribute):
                print('Warning: supercell is missing attribute "%s"' % attribute)
                print("setting original_cell=supercell and S=np.eye(3)")
                supercell.original_cell = supercell
                supercell.S = np.eye(3)

        self.parameters = {}
        self.real_tol = 1e4

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

        self._cell = supercell.original_cell

        self.parameters["mo_coeff_alpha"] = []
        self.parameters["mo_coeff_beta"] = []
        for kind in self.kinds:
            if len(mf.mo_coeff[0][0].shape) == 2:
                mca = mf.mo_coeff[0][kind][:, np.asarray(mf.mo_occ[0][kind] > 0.9)]
                mcb = mf.mo_coeff[1][kind][:, np.asarray(mf.mo_occ[1][kind] > 0.9)]
            else:
                mca = mf.mo_coeff[kind][:, np.asarray(mf.mo_occ[kind] > 0.9)]
                mcb = mf.mo_coeff[kind][:, np.asarray(mf.mo_occ[kind] > 1.1)]
            mca = np.real_if_close(mca, tol=self.real_tol)
            mcb = np.real_if_close(mcb, tol=self.real_tol)
            self.parameters["mo_coeff_alpha"].append(mca / np.sqrt(self.nk))
            self.parameters["mo_coeff_beta"].append(mcb / np.sqrt(self.nk))
        self._coefflookup = ("mo_coeff_alpha", "mo_coeff_beta")

        print("scf object is type", type(mf))
        if isinstance(mf, scf.kuhf.KUHF):
            # Then indices are (spin, kpt, basis, mo)
            self._nelec = [int(np.sum([o[k] for k in self.kinds])) for o in mf.mo_occ]
        elif isinstance(mf, scf.khf.KRHF):
            # Then indices are (kpt, basis, mo)
            self._nelec = [
                int(np.sum([mf.mo_occ[k] > t for k in self.kinds])) for t in (0.9, 1.1)
            ]
        else:
            print("Warning: not expecting scf object of type", type(mf))
            scale = self.supercell.scale
            self._nelec = [int(np.round(n * scale)) for n in self._cell.nelec]
        self._nelec = tuple(self._nelec)

        self.iscomplex = np.linalg.norm(self._kpts) > 1e-12
        for v in self.parameters.values():
            self.iscomplex = self.iscomplex or bool(sum(map(np.iscomplexobj, v)))
        print("iscomplex:", self.iscomplex)
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
        prim_coords, prim_wrap = pbc.enforce_pbc(self._cell.lattice_vectors(), mycoords)
        configswrap = configswrap.reshape(prim_wrap.shape)
        wrap = prim_wrap + np.dot(configswrap, self.supercell.S)
        kdotR = np.linalg.multi_dot(
            (self._kpts, self._cell.lattice_vectors().T, wrap.T)
        )
        wrap_phase = self.get_wrapphase(kdotR)
        # evaluate AOs for all electron positions
        ao = self._cell.eval_gto(eval_str, prim_coords, kpts=self._kpts)
        ao = [ao[k] * wrap_phase[k][:, np.newaxis] for k in range(self.nk)]
        return ao

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
            mo = []
            i0, i1 = s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
            for k in range(self.nk):
                mo_coeff = self.parameters[self._coefflookup[s]][k]
                mo.append(np.dot(aos[k, :, i0:i1], mo_coeff))
            ne = self._nelec[s]
            mo = np.concatenate(mo, axis=-1).reshape(nconf, ne, ne)
            phase, mag = np.linalg.slogdet(mo)
            self._dets.append((phase, mag))
            self._inverse.append(np.linalg.inv(mo))

        return self.value()

    def updateinternals(self, e, epos, mask=None):
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        eeff = e - s * self._nelec[0]
        aos = self.evaluate_orbitals(epos)
        self._aovals[:, :, e, :] = np.asarray(aos)
        mo = []
        for k in range(self.nk):
            mo_coeff = self.parameters[self._coefflookup[s]][k]
            mo.append(np.dot(aos[k], mo_coeff))
        ne = self._nelec[s]
        mo = np.concatenate(mo, axis=-1).reshape(len(mask), ne)
        ratio, self._inverse[s][mask, :, :] = slateruhf.sherman_morrison_row(
            eeff, self._inverse[s][mask, :, :], mo[mask, :]
        )
        self._updateval(ratio, s, mask)

    # identical to slateruhf
    def _updateval(self, ratio, s, mask):
        self._dets[s][0][mask] *= self.get_phase(ratio)
        self._dets[s][1][mask] += np.log(np.abs(ratio))

    ### not state-changing functions

    # identical to slateruhf
    def value(self):
        """Return logarithm of the wave function as noted in recompute()"""
        return self._dets[0][0] * self._dets[1][0], self._dets[0][1] + self._dets[1][1]

    # identical to slateruhf
    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin
        elec = e - s * self._nelec[0]
        if mask is None:
            return np.einsum("i...j,ij...->i...", vec, self._inverse[s][:, :, elec])

        return np.einsum("i...j,ij...->i...", vec, self._inverse[s][mask, :, elec])

    # identical to slateruhf
    def _testcol(self, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i"""
        ratio = np.einsum("ij,ij->i", vec, self._inverse[s][:, i, :])
        return ratio

    def testvalue(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        nmask = np.sum(mask)
        if nmask == 0:
            return np.zeros((0, epos.configs.shape[1]))
        aos = self.evaluate_orbitals(epos, mask)
        mo_coeff = self.parameters[self._coefflookup[s]]
        mo = [np.dot(aos[k], mo_coeff[k]) for k in range(self.nk)]
        mo = np.concatenate(mo, axis=-1)
        mo = mo.reshape(nmask, *epos.configs.shape[1:-1], self._nelec[s])
        return self._testrow(e, mo, mask)

    def testvalue_many(self, e, epos, mask=None):
        """ return the ratio between the current wave function and the wave function if 
        an electron's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        nmask = np.sum(mask)
        if nmask == 0:
            return np.zeros((0, epos.configs.shape[1]))
        aos = self.evaluate_orbitals(epos, mask)

        ratios = np.zeros(
            (epos.configs.shape[0], e.shape[0]),
            dtype=complex if self.iscomplex else float,
        )
        for spin in [0, 1]:
            ind = s == spin
            mo_coeff = self.parameters[self._coefflookup[spin]]
            mo = [np.dot(aos[k], mo_coeff[k]) for k in range(self.nk)]
            mo = np.concatenate(mo, axis=-1)
            mo = mo.reshape(nmask, *epos.configs.shape[1:-1], self._nelec[spin])
            ratios[:, ind] = self._testrow(e[ind], mo, spin=spin)
        return ratios

    def gradient(self, e, epos):
        """ Compute the gradient of the log wave function 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.evaluate_orbitals(epos, eval_str="PBCGTOval_sph_deriv1")
        mo_coeff = self.parameters[self._coefflookup[s]]
        mograd = [ak.dot(mo_coeff[k]) for k, ak in enumerate(aograd)]
        mograd = np.concatenate(mograd, axis=-1)
        ratios = np.asarray([self._testrow(e, x) for x in mograd])
        return ratios[1:] / ratios[:1]

    def laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.evaluate_orbitals(epos, eval_str="PBCGTOval_sph_deriv2")
        mo_coeff = self.parameters[self._coefflookup[s]]
        mo = [
            np.dot([ak[0], ak[[4, 7, 9]].sum(axis=0)], mo_coeff[k])
            for k, ak in enumerate(ao)
        ]
        mo = np.concatenate(mo, axis=-1)
        ratios = self._testrow(e, mo[1])
        testvalue = self._testrow(e, mo[0])
        return ratios / testvalue

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.evaluate_orbitals(epos, eval_str="PBCGTOval_sph_deriv2")
        mo = [
            np.dot(
                np.concatenate([ak[0:4], ak[[4, 7, 9]].sum(axis=0, keepdims=True)]),
                self.parameters[self._coefflookup[s]][k],
            )
            for k, ak in enumerate(ao)
        ]
        mo = np.concatenate(mo, axis=-1)
        ratios = np.asarray([self._testrow(e, x) for x in mo])
        return ratios[1:-1] / ratios[:1], ratios[-1] / ratios[0]

    def pgradient(self):
        d = {}
        # for parm in self.parameters:
        #    s = int("beta" in parm)
        #    # Get AOs for our spin channel only
        #    i0, i1 = s * self._nelec[0], self._nelec[0] + s * self._nelec[1]
        #    ao = self._aovals[:, :, i0:i1]  # (kpt, config, electron, ao)
        #    pgrad_shape = (ao.shape[1],) + self.parameters[parm].shape
        #    pgrad = np.zeros(pgrad_shape)
        #    # Compute derivatives w.r.t. MO coefficients
        #    for k in range(self.nk):
        #        for i in range(self._nelec[s]):
        #            for j in range(ao.shape[2]):
        #                pgrad[:, k, j, i] = self._testcol(i, s, ao[k, :, :, j])
        #    d[parm] = np.array(pgrad)
        return d

    def plot_orbitals(self, mf, norb, spin_channel=0, basename="", nx=80, ny=80, nz=80):
        from pyqmc.coord import PeriodicConfigs

        grid = np.meshgrid(*[np.arange(n) / n for n in [nx, ny, nz]], indexing="ij")
        grid = np.stack([g.ravel() for g in grid]).T
        grid = np.linalg.dot(grid, self.supercell.lattice_vectors())
        configs = PeriodicConfigs(
            grid.reshape((-1, 16, 3)), self._cell.lattice_vectors()
        )
        nconf, nelec, ndim = configs.configs.shape
        ao = self.evaluate_orbitals(configs)

        mo_coeff = np.asarray(mf.mo_coeff)
        coeff = []
        for kind in self.kinds:
            if len(mf.mo_coeff[0][0].shape) == 2:
                mca = mo_coeff[spin_channel][kind][:, :norb]
            else:
                mca = mf.mo_coeff[kind][:, :norb]
            mca = np.real_if_close(mca, tol=self.real_tol)
            coeff.append(mca)

        mo = []
        nsorb = int(np.round(np.linalg.det(self.S) * norb))
        for k in range(self.nk):
            mo.append(np.dot(ao[k], coeff[k]))
        mo = np.concatenate(mo, axis=-1).reshape(-1, nsorb)

        for i in range(nsorb):
            fname = basename + "mo{0}.cube".format(i)
            print("writing", fname, mo[..., i].shape)
            self.generate_cube(fname, mo[..., i], nx, ny, nz)

    def generate_cube(self, fname, vals, nx, ny, nz, comment="HEADER LINE\n"):
        import cubetools

        cube = {}
        cube["comment"] = comment
        cube["type"] = "\n"
        cube["natoms"] = self.supercell.natm
        cube["origin"] = np.zeros(3)
        cube["ints"] = np.array([nx, ny, nz])
        cube["latvec"] = self.supercell.lattice_vectors()
        cube["latvec"] = cube["latvec"] / cube["ints"][:, np.newaxis]
        cube["atomname"] = self.supercell.atom_charges()
        cube["atomxyz"] = self.supercell.atom_coords()
        cube["data"] = np.reshape(vals, (nx, ny, nz))
        with open(fname, "w") as f:
            cubetools.write_cube(cube, f)


def generate_test_inputs():
    import pyqmc
    from pyqmc.coord import PeriodicConfigs
    from pyscf.pbc import gto, scf
    from pyscf.pbc.dft.multigrid import multigrid
    from pyscf.pbc import tools
    from pyscf import lib

    from_chkfile = True

    if from_chkfile:

        def loadchkfile(chkfile):
            cell = gto.cell.loads(lib.chkfile.load(chkfile, "mol"))
            kpts = cell.make_kpts([1, 1, 1])
            mf = scf.KRKS(cell, kpts)
            mf.__dict__.update(lib.chkfile.load(chkfile, "scf"))
            return cell, mf

        cell1, mf1 = loadchkfile("mf1.chkfile")
        cell2, mf2 = loadchkfile("mf2.chkfile")
    else:
        L = 4
        cell2 = gto.M(
            atom="""H     {0}      {0}      {0}                
                      H     {1}      {1}      {1}""".format(
                0.0, L * 0.25
            ),
            basis="sto-3g",
            a=np.eye(3) * L,
            spin=0,
            unit="bohr",
        )

        print("Primitive cell")
        kpts = cell2.make_kpts((2, 2, 2))
        mf2 = scf.KRKS(cell2, kpts)
        mf2.xc = "pbe"
        mf2.chkfile = "mf2.chkfile"
        mf2 = mf2.run()

        print("Supercell")
        cell1 = tools.super_cell(cell2, [2, 2, 2])
        kpts = [[0, 0, 0]]
        mf1 = scf.KRKS(cell1, kpts)
        mf1.xc = "pbe"
        mf1.chkfile = "mf1.chkfile"
        mf1 = mf1.run()

    # wf1 = pyqmc.PySCFSlaterUHF(cell1, mf1)
    wf1 = PySCFSlaterPBC(cell1, mf1, supercell=1 * np.eye(3))
    wf2 = PySCFSlaterPBC(cell2, mf2, supercell=2 * np.eye(3))

    configs = pyqmc.initial_guess(cell1, 10, 0.1)

    return wf1, wf2, configs


def test_recompute(wf1, wf2, configs):
    p1, m1 = wf1.recompute(configs)
    p2, m2 = wf2.recompute(configs)

    print("phase")
    print("p1", p1)
    print("p2", p2)
    print("p1/p2", p1 / p2)
    print("log magnitude")
    print("m1", m1)
    print("m2", m2)
    print("m1-m2", m1 - m2)

    p_err = np.linalg.norm(p1 / p2 - p1[0] / p2[0])
    m_err = np.linalg.norm(m1 - m2 - m1[0] + m2[0])
    assert p_err < 1e-10, (p_err, m_err)
    assert m_err < 1e-1, (p_err, m_err)


if __name__ == "__main__":
    from pyqmc.testwf import (
        test_updateinternals,
        test_wf_gradient,
        test_wf_laplacian,
        test_wf_gradient_laplacian,
    )

    wf1, wf2, configs = generate_test_inputs()
    test_recompute(wf1, wf2, configs)
    test_updateinternals(wf1, configs)
    test_updateinternals(wf2, configs)
    test_wf_gradient(wf1, configs)
    test_wf_gradient(wf2, configs)
    test_wf_laplacian(wf1, configs)
    test_wf_laplacian(wf2, configs)
    test_wf_gradient_laplacian(wf1, configs)
    test_wf_gradient_laplacian(wf2, configs)
