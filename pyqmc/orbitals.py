import numpy as np
import pyqmc.gpu as gpu
import pyqmc.pbc as pbc
import pyqmc.supercell as supercell
import pyqmc.determinant_tools
import pyscf.pbc.gto.eval_gto
import pyscf.lib
import pyqmc.pbc
import pyqmc.twists as twists

"""
The evaluators have the concept of a 'set' of atomic orbitals, that may apply to 
different sets of molecular orbitals

For example, for the PBC evaluator, each k-point is a set, since each molecular 
orbital is only a sum over the k-point of its type.

In the future, this could apply to orbitals of a given point group symmetry, for example.
"""


def get_wrapphase_real(x):
    return (-1) ** np.round(x / np.pi)


def get_wrapphase_complex(x):
    return np.exp(1j * x)


def get_complex_phase(x):
    return x / np.abs(x)


def choose_evaluator_from_pyscf(
    mol, mf, mc=None, twist=None, determinants=None, tol=None
):
    """
    mol: A Mole object
    mf: a pyscf mean-field object
    mc: a pyscf multiconfigurational object. Supports HCI and CAS
    twist: the twist of the calculation (units?)
    determinants: A list of determinants suitable to pass into create_packed_objects
    tol: smallest determinant weight to include in the wave function.

    You cannot pass both mc/tol and determinants.

    Returns:
    an orbital evaluator chosen based on the inputs.
    """

    if hasattr(mol, "a"):
        return PBCOrbitalEvaluatorKpoints.from_mean_field(
            mol, mf, mc=mc, twist=twist, determinants=determinants, tol=tol
        )
    if mc is None:
        return MoleculeOrbitalEvaluator.from_pyscf(
            mol, mf, determinants=determinants, tol=tol
        )
    return MoleculeOrbitalEvaluator.from_pyscf(
        mol, mf, mc, determinants=determinants, tol=tol
    )


class MoleculeOrbitalEvaluator:
    def __init__(self, mol, mo_coeff):
        self.parameters = {
            "mo_coeff_alpha": gpu.cp.asarray(mo_coeff[0]),
            "mo_coeff_beta": gpu.cp.asarray(mo_coeff[1]),
        }
        self.parm_names = ["_alpha", "_beta"]
        iscomplex = bool(sum(map(gpu.cp.iscomplexobj, self.parameters.values())))
        self.ao_dtype = True
        self.mo_dtype = complex if iscomplex else float

        self._mol = mol

    @classmethod
    def from_pyscf(self, mol, mf, mc=None, tol=-1, determinants=None):
        """
        mol: A Mole object
        mf: An object with mo_coeff and mo_occ.
        mc: (optional) a CI object from pyscf

        """
        obj = mc if hasattr(mc, "mo_coeff") else mf
        if mc is not None:
            detcoeff, occup, det_map = pyqmc.determinant_tools.interpret_ci(mc, tol)
        elif determinants is not None:
            detcoeff, occup, det_map = pyqmc.determinant_tools.create_packed_objects(
                determinants, tol, format="list"
            )
        else:
            detcoeff = gpu.cp.array([1.0])
            det_map = gpu.cp.array([[0], [0]])
            # occup
            if len(mf.mo_occ.shape) == 2:
                occup = [
                    [list(np.argwhere(mf.mo_occ[spin] > 0.5)[:, 0])] for spin in [0, 1]
                ]
            else:
                occup = [
                    [list(np.argwhere(mf.mo_occ > 0.5 + spin)[:, 0])] for spin in [0, 1]
                ]

        max_orb = [int(np.max(occup[s], initial=0) + 1) for s in [0, 1]]
        if len(obj.mo_coeff[0].shape) == 2:
            mo_coeff = [obj.mo_coeff[spin][:, 0 : max_orb[spin]] for spin in [0, 1]]
        else:
            mo_coeff = [obj.mo_coeff[:, 0 : max_orb[spin]] for spin in [0, 1]]

        return detcoeff, occup, det_map, MoleculeOrbitalEvaluator(mol, mo_coeff)

    def nmo(self):
        return [
            self.parameters["mo_coeff_alpha"].shape[-1],
            self.parameters["mo_coeff_beta"].shape[-1],
        ]

    def aos(self, eval_str, configs, mask=None):
        """"""
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        aos = gpu.cp.asarray([self._mol.eval_gto(eval_str, mycoords)])
        if len(aos.shape) == 4:  # if derivatives are included
            return aos.reshape((1, aos.shape[1], *mycoords.shape[:-1], aos.shape[-1]))
        else:
            return aos.reshape((1, *mycoords.shape[:-1], aos.shape[-1]))

    def mos(self, ao, spin):
        return ao[0].dot(self.parameters[f"mo_coeff{self.parm_names[spin]}"])

    def pgradient(self, ao, spin):
        return (
            gpu.cp.array(
                [self.parameters[f"mo_coeff{self.parm_names[spin]}"].shape[1]]
            ),
            ao,
        )


def pbc_single_determinant(mf, kinds):
    detcoeff = np.array([1.0])
    det_map = np.array([[0], [0]])

    if len(mf.mo_coeff[0][0].shape) == 2:
        occup_k = [
            [[list(np.argwhere(mf.mo_occ[spin][k] > 0.5)[:, 0])] for k in kinds]
            for spin in [0, 1]
        ]
    elif len(mf.mo_coeff[0][0].shape) == 1:
        occup_k = [
            [[list(np.argwhere(mf.mo_occ[k] > 1.5 - spin)[:, 0])] for k in kinds]
            for spin in [0, 1]
        ]
    return detcoeff, det_map, occup_k


def select_orbitals_kpoints(determinants, mo_coeff, kinds):
    """
    Based on the k-point indices in `kinds`, select the MO coefficients that correspond to those k-points,
    and the determinants.
    The determinant indices are flattened so that the indices refer to the concatenated MO coefficients.
    """
    max_orb = [
        [[np.max(orb_k) + 1 if len(orb_k) > 0 else 0 for orb_k in spin] for spin in det]
        for wt, det in determinants
    ]
    max_orb = np.amax(max_orb, axis=0)

    if len(mo_coeff[0][0].shape) == 2:
        mf_mo_coeff = mo_coeff
    elif len(mo_coeff[0][0].shape) == 1:
        mf_mo_coeff = [mo_coeff, mo_coeff]
    else:
        raise ValueError(
            f"mo_coeff[0][0] has unexpected number of array dimensions: {mo_coeff[0][0].shape}"
        )
    mo_coeff = [
        [mf_mo_coeff[s][k][:, 0 : max_orb[s][k]] for ki, k in enumerate(kinds)]
        for s in range(2)
    ]

    # and finally, we remove the k-index from determinants
    determinants_flat = []
    orb_offsets = np.cumsum(max_orb[:, kinds], axis=1)
    orb_offsets = np.pad(orb_offsets[:, :-1], ((0, 0), (1, 0)))
    for wt, det in determinants:
        flattened_det = []
        for det_s, offset_s in zip(det, orb_offsets):
            flattened = (
                np.concatenate([det_s[k] + offset_s[ki] for ki, k in enumerate(kinds)])
                .flatten()
                .astype(int)
            )
            flattened_det.append(list(flattened))
        determinants_flat.append((wt, flattened_det))
    return mo_coeff, determinants_flat


class PBCOrbitalEvaluatorKpoints:
    """
    Evaluate orbitals from a PBC object.
    cell is expected to be one made with make_supercell().
    mo_coeff should be in [spin][k][ao,mo] order
    kpts should be a list of the k-points corresponding to mo_coeff

    """

    def __init__(self, cell, mo_coeff=None, kpts=None):
        """
        :parameter cell: PyQMC supercell object (from get_supercell)
        :parameter mo_coeff: (2, nk, nao, nelec) array. MO coefficients for all kpts of primitive cell. If None, this object can't evaluate mos(), but can still evaluate aos().
        :parameter kpts: list of kpts to evaluate AOs
        """
        self._cell = cell.original_cell
        self.S = cell.S
        self.Lprim = self._cell.lattice_vectors()

        self._kpts = [0, 0, 0] if kpts is None else kpts
        # If gamma-point only, AOs are real-valued
        isgamma = np.abs(self._kpts).sum() < 1e-9
        if mo_coeff is not None:
            nelec_per_kpt = [np.asarray([m.shape[1] for m in mo]) for mo in mo_coeff]
            self.param_split = [np.cumsum(nelec_per_kpt[spin]) for spin in [0, 1]]
            self.parm_names = ["_alpha", "_beta"]
            self.parameters = {
                "mo_coeff_alpha": gpu.cp.asarray(np.concatenate(mo_coeff[0], axis=1)),
                "mo_coeff_beta": gpu.cp.asarray(np.concatenate(mo_coeff[1], axis=1)),
            }
            iscomplex = (not isgamma) or bool(
                sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
            )
        else:
            iscomplex = not isgamma

        self.ao_dtype = float if isgamma else complex
        self.mo_dtype = complex if iscomplex else float
        Ls = self._cell.get_lattice_Ls(dimension=3)
        self.Ls = Ls[np.argsort(pyscf.lib.norm(Ls, axis=1))]
        self.rcut = pyscf.pbc.gto.eval_gto._estimate_rcut(self._cell)

    @classmethod
    def from_mean_field(
        self, cell, mf, mc=None, twist=None, determinants=None, tol=None
    ):
        """
        mf is expected to be a KUHF, KRHF, or equivalent DFT objects.
        Selects occupied orbitals from a given twist
        If cell is a supercell, will automatically choose the folded k-points that correspond to that twist.

        """

        cell = (
            cell
            if hasattr(cell, "original_cell")
            else supercell.get_supercell(
                cell, np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            )
        )
        if twist is None:
            twist = 0
        if not hasattr(mf, "kpts"):
            mf.kpts = np.zeros((1, 3))
            if len(mf.mo_occ.shape) == 1:
                mf.mo_coeff = [mf.mo_coeff]
                mf.mo_occ = [mf.mo_occ]
            elif len(mf.mo_occ.shape) == 2:
                mf.mo_coeff = [[c] for c in mf.mo_coeff]
                mf.mo_occ = [[o] for o in mf.mo_occ]

        kinds = twists.create_supercell_twists(cell, mf)["primitive_ks"][twist]
        if len(kinds) != cell.scale:
            raise ValueError(
                f"Found {len(kinds)} k-points but should have found {cell.scale}."
            )
        kpts = mf.kpts[kinds]

        if determinants is None:
            if mc is not None:
                if not hasattr(mc, "orbitals") or mc.orbitals is None:
                    mc.orbitals = np.arange(mc.ncore, mc.ncore + mc.ncas)
                determinants = pyqmc.determinant_tools.pbc_determinants_from_casci(
                    mc, mc.orbitals
                )
            else:
                determinants = [
                    (1.0, pyqmc.determinant_tools.create_pbc_determinant(cell, mf, []))
                ]

        if mc is not None:
            mo_coeff = [mc.mo_coeff]  # kpt list
        else:
            mo_coeff = mf.mo_coeff
        mo_coeff, determinants_flat = select_orbitals_kpoints(
            determinants, mo_coeff, kinds
        )
        detcoeff, occup, det_map = pyqmc.determinant_tools.create_packed_objects(
            determinants_flat, format="list", tol=tol
        )
        # Check
        for s, (occ_s, nelec_s) in enumerate(zip(occup, cell.nelec)):
            for determinant in occ_s:
                if len(determinant) != nelec_s:
                    raise RuntimeError(
                        f"The number of electrons of spin {s} should be {nelec_s}, but found {len(determinant)} orbital[s]. You may have used a large smearing value.. Please pass your own determinants list. "
                    )

        return (
            detcoeff,
            occup,
            det_map,
            PBCOrbitalEvaluatorKpoints(cell, mo_coeff, kpts),
        )

    def nmo(self):
        return [
            self.parameters["mo_coeff_alpha"].shape[-1],
            self.parameters["mo_coeff_beta"].shape[-1],
        ]

    def aos(self, eval_str, configs, mask=None):
        """
        Returns an ndarray in order [k,..., orbital] of the ao's if value is requested

        if a derivative is requested, will instead return [k,d,...,orbital].

        The ... is the total length of mycoords. You'll need to reshape if you want the original shape
        """
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        primcoords, primwrap = pbc.enforce_pbc(self.Lprim, mycoords)
        ao = gpu.cp.asarray(
            pyscf.pbc.gto.eval_gto.eval_gto(
                self._cell,
                "PBC" + eval_str,
                primcoords,
                kpts=self._kpts,
                rcut=self.rcut,
                Ls=self.Ls,
            )
        )
        if self.ao_dtype == complex:
            wrap = configs.wrap if mask is None else configs.wrap[mask]
            wrap = np.dot(wrap, self.S)
            wrap = wrap.reshape((-1, wrap.shape[-1])) + primwrap
            kdotR = np.linalg.multi_dot(
                (self._kpts, self._cell.lattice_vectors().T, wrap.T)
            )
            # k, coordinate
            wrap_phase = get_wrapphase_complex(kdotR)
            ao = gpu.cp.einsum("k...,k...a->k...a", wrap_phase, ao)
        if len(ao.shape) == 4:  # if derivatives are included
            return ao.reshape(
                (ao.shape[0], ao.shape[1], *mycoords.shape[:-1], ao.shape[-1])
            )
        else:
            return ao.reshape((ao.shape[0], *mycoords.shape[:-1], ao.shape[-1]))

    def mos(self, ao, spin):
        """ao should be [k,[d],...,ao].
        Returns a concatenated list of all molecular orbitals in form [..., mo]

        In the derivative case, returns [d,..., mo]
        """
        p = np.split(
            self.parameters[f"mo_coeff{self.parm_names[spin]}"],
            self.param_split[spin],
            axis=-1,
        )
        ps = [0] + list(self.param_split[spin])
        nelec = self.parameters[f"mo_coeff{self.parm_names[spin]}"].shape[1]
        out = gpu.cp.zeros([nelec, *ao[0].shape[:-1]], dtype=self.mo_dtype)
        for i, ak, mok in zip(range(len(ao)), ao, p[:-1]):
            gpu.cp.einsum("...a,an->n...", ak, mok, out=out[ps[i] : ps[i + 1]])
        return out.transpose([*np.arange(1, len(out.shape)), 0])

    def pgradient(self, ao, spin):
        """
        returns:
        N sets of atomic orbitals
        split: which molecular orbitals correspond to which set

        You can construct the determinant by doing, for example:
        split, aos = pgradient(self.aos)
        mos = np.split(range(nmo),split)
        for ao, mo in zip(aos,mos):
            for i in mo:
                pgrad[:,:,i] = self._testcol(i,spin,ao)

        """
        return self.param_split[spin], ao
