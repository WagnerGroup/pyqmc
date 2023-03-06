import numpy as np
import pyqmc.gpu as gpu
import pyscf.fci as fci


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


def determinants_from_mean_field(mf):
    """
    mf can be a few different things:
    RHF on molecule or single k-point:
        mf.mo_coeff is [nao, nmo]
    ROHF on molecule or single k-point:
        mf.mo_coeff is [nao, nmo]
    UHF on molecule or single k-point
        mf.mo_coeff is [spin][nao,nmo]
    KRHF:
        mf.mo_coeff is [k][nao,nmo]
    KUHF:
        mf.mo_coeff is [spin][k][nao,nmo]
    """
    detwt = np.array([1.0])
    occup = [
        [list(np.nonzero(mf.mo_occ[0] > 0.9)[0])],
        [list(np.nonzero(mf.mo_occ[1] > 0.9)[0])],
    ]
    map_dets = np.array([[0], [0]])
    return detwt, occup, map_dets


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

    returns:
    detwt: array of weights for each determinant
    occup: which orbitals go in which determinants
    map_dets: given a determinant in detwt, which determinant in occup it corresponds to
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


def create_pbc_determinant(mol, mf, excitations):
    """
    excitations should be a list of tuples with
    (s,ka,a,ki,i),
    s is the spin (0 or 1),
    ka, a is the occupied orbital,
    and ki, i is the unoccupied orbital.
    """
    if len(mf.mo_coeff[0][0].shape) == 2:
        occupation = [
            [list(np.nonzero(occ > 0.9)[0]) for occ in mf.mo_occ[s]] for s in range(2)
        ]
    elif len(mf.mo_coeff[0][0].shape) == 1:
        occupation = [
            [list(np.nonzero(occ > 1.9 - s)[0]) for occ in mf.mo_occ] for s in range(2)
        ]

    for s, ka, a, ki, i in excitations:
        occupation[s][ka].remove(a)
        occupation[s][ki].append(i)
    return occupation


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
