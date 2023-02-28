import numpy as np
import pyqmc.gpu as gpu
import pyscf.fci as fci
import pyqmc.orbitals


def binary_to_occ(S, ncore):
    """
    Converts the binary cistring for a given determinant
    to occupation values for molecular orbitals within
    the determinant.
    """
    occup = [int(i) for i in range(ncore)]
    occup += [int(i + ncore) for i, c in enumerate(reversed(S)) if c == "1"]
    max_orb = max(occup) if occup else 1
    return (occup, max_orb)


def deters_from_hci(mc, tol):
    bigcis = np.abs(mc.ci) > tol
    nstrs = int(mc._strs.shape[1] / 2)
    deters = []
    # In pyscf, the first n/2 strings represent the up determinant and the second
    # represent the down determinant.
    for c, s in zip(mc.ci[bigcis], mc._strs[bigcis, :]):
        s1 = "".join(str(bin(p)).replace("0b", "") for p in s[0:nstrs])
        s2 = "".join(str(bin(p)).replace("0b", "") for p in s[nstrs:])
        deters.append((c, s1, s2))
    return deters


def interpret_ci(mc, tol):
    """
    Copies over determinant coefficients and MO occupations
    for a multi-configuration calculation mc.

    returns:
    detwt: array of weights for each determinant
    occup: which orbitals go in which determinants
    map_dets: given a determinant in detwt, which determinant in occup it corresponds to
    """
    ncore = mc.ncore if hasattr(mc, "ncore") else 0
    # find multi slater determinant occupation
    if hasattr(mc, "_strs"):  # if this is a HCI object, it will have _strs
        deters = deters_from_hci(mc, tol)
    else:
        deters = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    return create_packed_objects(deters, ncore, tol)


def create_packed_objects(deters, ncore=0, tol=0, format="binary"):
    """
    if format == "binary":
    deters is expected to be an iterable of tuples, each of which is
    (weight, occupation string up, occupation_string down)
    if format == "list"
    (weight, occupation)
    where occupation is a nested list [s][0, 1, 2, 3 ..], for example.

    ncore should be the number of core orbitals not included in the occupation strings.
    tol is the threshold at which to include the determinants

    :returns:
        * detwt: array of weights for each determinant
        * occup: which orbitals go in which determinants
        * map_dets: given a determinant in detwt, which determinant in occup it corresponds to
    """
    # Create map and occupation objects
    detwt = []
    map_dets = [[], []]
    occup = [[], []]
    for x in deters:
        if np.abs(x[0]) > tol:
            detwt.append(x[0])
            if format == "binary":
                alpha_occ, __ = binary_to_occ(x[1], ncore)
                beta_occ, __ = binary_to_occ(x[2], ncore)
            elif format == "list":
                alpha_occ = x[1][0]
                beta_occ = x[1][1]
            else:
                raise ValueError(
                    "create_packed_objects: Options for format are binary or list"
                )
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
    return np.array(detwt), occup, np.array(map_dets)


def compute_value(updets, dndets, det_coeffs):
    """
    Given the up and down determinant values, safely compute the total log wave function.
    """
    upref = gpu.cp.amax(updets[1]).real
    dnref = gpu.cp.amax(dndets[1]).real
    phases = updets[0] * dndets[0]
    logvals = updets[1] - upref + dndets[1] - dnref

    phases = updets[0] * dndets[0]
    wf_val = gpu.cp.einsum("d,id->i", det_coeffs, phases * gpu.cp.exp(logvals))

    wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
    wf_logval = np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + upref + dnref)
    return gpu.asnumpy(wf_sign), gpu.asnumpy(wf_logval)


def translate_occ(x, orbitals, nocc):
    a = binary_to_occ(x, 0)[0]
    orbitals_without_active = list(range(nocc))
    for o in orbitals:
        if o in orbitals_without_active:
            orbitals_without_active.remove(o)

    return orbitals_without_active + [orbitals[i] for i in a]


def create_single_determinant(mf):
    """
    excitations should be a list of tuples with
    (s,ka,a,ki,i),
    s is the spin (0 or 1),
    ka, a is the occupied orbital,
    and ki, i is the unoccupied orbital.
    """
    coeff = mf.mo_coeff[0][0] if hasattr(mf, "kpts") else mf.mo_coeff[0]
    if len(coeff.shape) == 2:
        _occup = [mf.mo_occ[spin] > 0.5 for spin in [0, 1]]
    elif len(coeff.shape) == 1:
        _occup = [mf.mo_occ > 0.5 + spin for spin in [0, 1]]
    occupation = [list(np.argwhere(occupied)[0]) for occupied in _occup]
    return [(1.0, occupation)]


def pbc_determinants_from_casci(mc, orbitals, cutoff=0.05):
    if hasattr(mc.ncore, "__len__"):
        nocc = [c + e for c, e in zip(mc.ncore, mc.nelecas)]
    else:
        nocc = [mc.ncore + e for e in mc.nelecas]
    if not hasattr(orbitals[0], "__len__"):
        orbitals = [orbitals, orbitals]
    deters = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    determinants = []
    for x in deters:
        if abs(x[0]) > cutoff:
            allorbs = [
                [translate_occ(x[1], orbitals[0], nocc[0])],
                [translate_occ(x[2], orbitals[1], nocc[1])],
            ]
            determinants.append((x[0], allorbs))
    return determinants


def create_mol_expansion(mol, mf, mc=None, tol=-1, determinants=None):
    """
    mol: A Mole object
    mf: An object with mo_coeff and mo_occ.
    mc: (optional) a CI object from pyscf

    """
    if mc is not None:
        detcoeff, occup, det_map = interpret_ci(mc, tol)
    else: 
        if determinants is None:
            determinants = create_single_determinant(mf)
        detcoeff, occup, det_map = create_packed_objects(
            determinants, tol=tol, format="list"
        )
         
    max_orb = [int(np.max(occup[s], initial=0) + 1) for s in [0, 1]]
    _mo_coeff = mc.mo_coeff if hasattr(mc, "mo_coeff") else mf.mo_coeff
    if len(_mo_coeff[0].shape) == 2:
        mo_coeff = [_mo_coeff[spin][:, 0 : max_orb[spin]] for spin in [0, 1]]
    else:
        mo_coeff = [_mo_coeff[:, 0 : max_orb[spin]] for spin in [0, 1]]

    evaluator = pyqmc.orbitals.MoleculeOrbitalEvaluator(mol, mo_coeff)
    return detcoeff, occup, det_map, evaluator


def create_pbc_expansion(cell, mf, mc=None, twist=0, determinants=None, tol=-1):
    """
    mf is expected to be a KUHF, KRHF, or equivalent DFT objects.
    Selects occupied orbitals from a given twist
    If cell is a supercell, will automatically choose the folded k-points that correspond to that twist.

    """

    if not hasattr(cell, "original_cell"):
        cell = supercell.get_supercell(cell, np.eye(3))
    if not hasattr(mf, "kpts"):
        mf = pyqmc.pbc.scf.addons.convert_to_khf(mf)
    kinds = twists.create_supercell_twists(cell, mf)['primitive_ks'][twist]
    if len(kinds) != cell.scale:
        raise ValueError(
            f"Found {len(kinds)} k-points but should have found {cell.scale}."
        )
    kpts = mf.kpts[kinds]

    if mc is not None:
        if not hasattr(mc, "orbitals") or mc.orbitals is None:
            mc.orbitals = np.arange(mc.ncore, mc.ncore + mc.ncas)
        if determinants is None:
            determinants = pbc_determinants_from_casci(
                mc, mc.orbitals
            )
    if mc is None and determinants is None:
        determinants = create_single_determinant(mf)

    mo_coeff, determinants_flat = select_orbitals_kpoints(determinants, mf, kinds)
    detcoeff, occup, det_map = create_packed_objects(
        determinants_flat, format="list", tol=tol
    )
    # Check
    for s, (occ_s, nelec_s) in enumerate(zip(occup, cell.nelec)):
        for determinant in occ_s:
            if len(determinant) != nelec_s:
                raise RuntimeError(
                    f"The number of electrons of spin {s} should be {nelec_s}, but found {len(determinant)} orbital[s]. You may have used a large smearing value.. Please pass your own determinants list. "
                    )

    evaluator = pyqmc.orbitals.PBCOrbitalEvaluatorKpoints(cell, mo_coeff, kpts)
    return detcoeff, occup, det_map, evaluator


def select_orbitals_kpoints(determinants, mf, kinds):
    """
    Based on the k-point indices in `kinds`, select the MO coefficients that correspond to those k-points,
    and the determinants.
    The determinant indices are flattened so that the indices refer to the concatenated MO coefficients.
    """
    max_orb = [
        [[np.max(orb_k, initial=0) + 1 for orb_k in orb_s] for orb_s in det]
        for wt, det in determinants
    ]
    max_orb = np.amax(max_orb, axis=0)

    if len(mf.mo_coeff[0][0].shape) == 2:
        _coeff = mf.mo_coeff
    elif len(mf.mo_coeff[0][0].shape) == 1:
        _coeff = [mf.mo_coeff, mf.mo_coeff]
    mo_coeff = [[_coeff[s][k][:, 0 : max_orb[s][k]] for k in kinds] for s in [0, 1]]

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
