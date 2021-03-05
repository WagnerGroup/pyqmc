import numpy as np
from pyscf import fci


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

def interpret_ci(mc, tol):
    """       
    Copies over determinant coefficients and MO occupations
    for a multi-configuration calculation mc.

    returns:
    detwt: array of weights for each determinant
    occup: which orbitals go in which determinants
    map_dets: 
    """
    from pyscf import fci

    ncore = mc.ncore if hasattr(mc, "ncore") else 0

    # find multi slater determinant occupation
    if hasattr(mc, "_strs"):
        # if this is a HCI object, it will have _strs
        bigcis = np.abs(mc.ci) > tol
        nstrs = int(mc._strs.shape[1] / 2)
        # old code for single strings.
        # deters = [(c,bin(s[0]), bin(s[1])) for c, s in zip(mc.ci[bigcis],mc._strs[bigcis,:])]
        deters = []
        # In pyscf, the first n/2 strings represent the up determinant and the second
        # represent the down determinant.
        for c, s in zip(mc.ci[bigcis], mc._strs[bigcis, :]):
            s1 = "".join(str(bin(p)).replace("0b", "") for p in s[0:nstrs])
            s2 = "".join(str(bin(p)).replace("0b", "") for p in s[nstrs:])
            deters.append((c, s1, s2))
    else:
        deters = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)

    # Create map and occupation objects
    detwt = []
    map_dets = [[], []]
    occup = [[], []]
    for x in deters:
        if np.abs(x[0]) > tol:
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
    return np.array(detwt), occup, np.array(map_dets)


def compute_value(updets, dndets, det_coeffs):
    """
    Given the up and down determinant values, safely compute the total log wave function.
    """
    upref = np.amax(updets[1])
    dnref = np.amax(dndets[1])
    phases = updets[0] * dndets[0]
    logvals = updets[1] - upref + dndets[1] - dnref

    phases = updets[0] * dndets[0]
    wf_val = np.einsum(
        "d,id->i", det_coeffs, phases * np.exp(logvals)
    )

    wf_sign = wf_val/np.abs(wf_val)
    wf_logval = np.log(np.abs(wf_val)) + upref + dnref
    return wf_sign, wf_logval

