# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
import pyqmc.gpu as gpu


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


def reformat_binary_dets(deters, ncore=0, tol=0):
    #f = lambda x: (binary_to_occ(x[1], ncore)[0], binary_to_occ(x[2], ncore)[0])
    def f(x):
        return (binary_to_occ(x[1], ncore)[0], binary_to_occ(x[2], ncore)[0])
    return [(x[0], f(x)) for x in deters if np.abs(x[0]) > tol]


def create_packed_objects(deters, tol=0):
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
            spin_occ = x[1]
            for s in [0, 1]:
                if spin_occ[s] not in occup[s]:
                    map_dets[s].append(len(occup[s]))
                    occup[s].append(spin_occ[s])
                else:
                    map_dets[s].append(occup[s].index(spin_occ[s]))

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


def flatten_determinants(determinants, max_orb, kinds):
    """
    The determinant indices are flattened so that the indices refer to the concatenated MO coefficients.
    """
    determinants_flat = []
    orb_offsets = np.cumsum(max_orb[:, kinds], axis=1)
    orb_offsets = np.pad(orb_offsets[:, :-1], ((0, 0), (1, 0)))
    for wt, det in determinants:
        flattened_det = []
        for det_s, offset_s in zip(det, orb_offsets):
            detlist = [det_s[k] + offset_s[ki] for ki, k in enumerate(kinds)]
            flattened_det.append(list(np.concatenate(detlist).flatten().astype(int)))
        determinants_flat.append((wt, flattened_det))
    return determinants_flat
