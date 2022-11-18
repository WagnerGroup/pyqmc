import numpy as np
import pyqmc.gpu as gpu
import pyqmc.pbc as pbc
import pyqmc.supercell as supercell
import pyqmc.pbc_eval_gto as pbc_eval_gto
import pyqmc.determinant_tools
import pyscf.pbc.dft.gen_grid


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
        if mc is not None:
            if not hasattr(mc, "orbitals") or mc.orbitals is None:
                mc.orbitals = np.arange(mc.ncore, mc.ncore + mc.ncas)
            determinants = pyqmc.determinant_tools.pbc_determinants_from_casci(
                mc, mc.orbitals
            )
        return PBCOrbitalEvaluatorKpoints.from_mean_field(
            mol, mf, twist, determinants=determinants, tol=tol
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
        self.iscomplex = False
        self.parameters = {
            "mo_coeff_alpha": gpu.cp.asarray(mo_coeff[0]),
            "mo_coeff_beta": gpu.cp.asarray(mo_coeff[1]),
        }
        self.parm_names = ["_alpha", "_beta"]

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


def get_k_indices(cell, mf, kpts, tol=1e-6):
    """Given a list of kpts, return inds such that mf.kpts[inds] is a list of kpts equivalent to the input list"""
    kdiffs = mf.kpts[np.newaxis] - kpts[:, np.newaxis]
    frac_kdiffs = np.dot(kdiffs, cell.lattice_vectors().T) / (2 * np.pi)
    kdiffs = np.mod(frac_kdiffs + 0.5, 1) - 0.5
    return np.nonzero(np.linalg.norm(kdiffs, axis=-1) < tol)[1]


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


def select_orbitals_kpoints(determinants, mf, kinds):
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

    if len(mf.mo_coeff[0][0].shape) == 2:
        mf_mo_coeff = mf.mo_coeff
    elif len(mf.mo_coeff[0][0].shape) == 1:
        mf_mo_coeff = [mf.mo_coeff, mf.mo_coeff]
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

    def __init__(self, cell, mo_coeff, kpts=None):
        self.iscomplex = True
        self._cell = cell.original_cell
        self.S = cell.S
        self.Lprim = self._cell.lattice_vectors()

        self._kpts = [0, 0, 0] if kpts is None else kpts
        self.param_split = [
            np.cumsum(np.asarray([m.shape[1] for m in mo_coeff[spin]]))
            for spin in [0, 1]
        ]
        self.parm_names = ["_alpha", "_beta"]
        self.parameters = {
            "mo_coeff_alpha": gpu.cp.asarray(np.concatenate(mo_coeff[0], axis=1)),
            "mo_coeff_beta": gpu.cp.asarray(np.concatenate(mo_coeff[1], axis=1)),
        }

        self.Ls = pbc_eval_gto.get_lattice_Ls(self._cell)
        self.rcut = pbc_eval_gto._estimate_rcut(self._cell)

    @classmethod
    def from_mean_field(self, cell, mf, twist=None, determinants=None, tol=None):
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
            twist = np.zeros(3)
        else:
            twist = np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
        if not hasattr(mf, "kpts"):
            mf.kpts = np.zeros((1, 3))
            if len(mf.mo_occ.shape) == 1:
                mf.mo_coeff = [mf.mo_coeff]
                mf.mo_occ = [mf.mo_occ]
            elif len(mf.mo_occ.shape) == 2:
                mf.mo_coeff = [[c] for c in mf.mo_coeff]
                mf.mo_occ = [[o] for o in mf.mo_occ]
        kinds = list(
            set(get_k_indices(cell, mf, supercell.get_supercell_kpts(cell) + twist))
        )
        if len(kinds) != cell.scale:
            raise ValueError(
                f"Found {len(kinds)} k-points but should have found {cell.scale}."
            )
        kpts = mf.kpts[kinds]

        if determinants is None:
            determinants = [
                (1.0, pyqmc.determinant_tools.create_pbc_determinant(cell, mf, []))
            ]

        mo_coeff, determinants_flat = select_orbitals_kpoints(determinants, mf, kinds)
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
        # coordinate, dimension
        wrap = configs.wrap if mask is None else configs.wrap[mask]
        wrap = np.dot(wrap, self.S)
        wrap = wrap.reshape((-1, wrap.shape[-1])) + primwrap
        kdotR = np.linalg.multi_dot(
            (self._kpts, self._cell.lattice_vectors().T, wrap.T)
        )
        # k, coordinate
        wrap_phase = get_wrapphase_complex(kdotR)
        # k,coordinate, orbital
        ao = gpu.cp.asarray(
            self._cell.eval_gto(
                "PBC" + eval_str,
                primcoords,
                kpts=self._kpts,
            )
        )
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
        return gpu.cp.concatenate(
            [ak.dot(mok) for ak, mok in zip(ao, p[0:-1])], axis=-1
        )

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
