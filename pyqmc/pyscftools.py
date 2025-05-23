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

import pyscf
import pyscf.pbc
import pyscf.mcscf
import pyscf.fci
import h5py
import json
import numpy as np
import pyqmc.wf.determinant_tools as determinant_tools
import pyqmc.wf.orbitals as orbitals
import pyqmc.pbc.supercell as supercell
import pyqmc.pbc.twists as twists


def recover_pyscf(chkfile, ci_checkfile=None, cancel_outputs=True):
    """Generate pyscf objects from a pyscf checkfile, in a way that is easy to use for pyqmc. The chkfile should be saved by setting mf.chkfile in a pyscf SCF object.

    It is recommended to write and recover the objects, rather than trying to use pyscf objects directly when dask parallelization is being used, since by default the pyscf objects contain unserializable objects. (this may be changed in the future)

    `cancel_outputs` will set the outputs of the objects to None. You may need to set `cancel_outputs=False` if you are using this to input to other pyscf functions.

    Typical usage::

        mol, mf = recover_pyscf("dft.hdf5")

    :param chkfile: The filename to read from.
    :type chkfile: string
    :return: mol, mf
    :rtype: pyscf Mole, SCF objects"""

    with h5py.File(chkfile, "r") as f:
        periodic = "a" in json.loads(f["mol"][()]).keys()

    if not periodic:
        mol = pyscf.lib.chkfile.load_mol(chkfile)
        with h5py.File(chkfile, "r") as f:
            mo_occ_shape = f["scf/mo_occ"].shape
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if len(mo_occ_shape) == 2:
            mf = pyscf.scf.UHF(mol)
        elif len(mo_occ_shape) == 1:
            mf = pyscf.scf.ROHF(mol) if mol.spin != 0 else pyscf.scf.RHF(mol)
        else:
            raise Exception("Couldn't determine type from chkfile")
    else:
        mol = pyscf.pbc.lib.chkfile.load_cell(chkfile)
        with h5py.File(chkfile, "r") as f:
            has_kpts = "mo_occ__from_list__" in f["/scf"].keys()
            if has_kpts:
                rhf = "000000" in f["/scf/mo_occ__from_list__/"].keys()
            else:
                rhf = len(f["/scf/mo_occ"].shape) == 1
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if not rhf and has_kpts:
            mf = pyscf.pbc.scf.KUHF(mol)
        elif has_kpts:
            mf = pyscf.pbc.scf.KROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.KRHF(mol)
        elif rhf:
            mf = pyscf.pbc.scf.ROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.RHF(mol)
        else:
            mf = pyscf.pbc.scf.UHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))

    if ci_checkfile is not None:
        casdict = pyscf.lib.chkfile.load(ci_checkfile, "ci")
        if casdict is None:
            casdict = pyscf.lib.chkfile.load(ci_checkfile, "mcscf")
        with h5py.File(ci_checkfile, "r") as f:
            hci = "ci/_strs" in f.keys()
        if hci:
            mc = pyscf.hci.SCI(mol)
        else:
            if len(casdict["mo_coeff"].shape) == 3:
                mc = pyscf.mcscf.UCASCI(mol, casdict["ncas"], casdict["nelecas"])
            else:
                mc = pyscf.mcscf.CASCI(mol, casdict["ncas"], casdict["nelecas"])
        mc.__dict__.update(casdict)

        return mol, mf, mc
    return mol, mf


def orbital_evaluator_from_pyscf(
    mol, mf, mc=None, twist=0, determinants=None, tol=None, eval_gto_precision=None, evaluate_orbitals_with="pyscf",
):
    """
    mol: A Mole object
    mf: a pyscf mean-field object
    mc: a pyscf multiconfigurational object. Supports HCI and CAS
    twist: the twist of the calculation (units?)
    determinants: A list of determinants suitable to pass into create_packed_objects
    tol: smallest determinant weight to include in the wave function.
    eval_gto_precision: desired value of orbital at rcut, used for determining rcut for periodic system. Default value = 0.01

    You cannot pass both mc/tol and determinants.

    :returns:
        * detwt: array of weights for each determinant
        * occup: which orbitals go in which determinants
        * map_dets: given a determinant in detwt, which determinant in occup it corresponds to
        * an orbital evaluator chosen based on the inputs.
    """

    periodic = hasattr(mol, "a")
    f_max_orb = lambda a: int(np.max(a, initial=0)) + 1 if len(a) > 0 else 0

    if periodic:
        mf = pyscf.pbc.scf.addons.convert_to_khf(mf)

    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)

    if determinants is None:
        determinants = determinants_from_pyscf(mol, mf, mc=mc, tol=tol)

    if hasattr(mc, "mo_coeff"):
        # assume no kpts for mc calculation
        _mo_coeff = mc.mo_coeff
        if len(_mo_coeff.shape) == 2:  # restricted spin: create up and down copies
            _mo_coeff = [_mo_coeff, _mo_coeff]
        if periodic:
            _mo_coeff = [m[np.newaxis] for m in _mo_coeff]  # add kpt dimension
    else:
        _mo_coeff = mf.mo_coeff

    if periodic:
        if not hasattr(mol, "original_cell"):
            mol = supercell.get_supercell(mol, np.eye(3))
        kinds = twists.create_supercell_twists(mol, mf)["primitive_ks"][twist]
        if len(kinds) != mol.scale:
            raise ValueError(
                f"Found {len(kinds)} k-points but should have found {mol.scale}."
            )
        kpts = mf.kpts[kinds]

        max_orb = [[[f_max_orb(k) for k in s] for s in det] for wt, det in determinants]
        max_orb = np.amax(max_orb, axis=0)
        mo_coeff = [
            [_mo_coeff[s][k][:, 0 : max_orb[s][k]] for k in kinds] for s in [0, 1]
        ]

        evaluator = orbitals.PBCOrbitalEvaluatorKpoints(
            mol, mo_coeff, kpts, eval_gto_precision, evaluate_orbitals_with
        )
        determinants = determinant_tools.flatten_determinants(
            determinants, max_orb, kinds
        )
    else:
        max_orb = [[f_max_orb(s) for s in det] for wt, det in determinants]
        max_orb = np.amax(max_orb, axis=0)
        mo_coeff = [_mo_coeff[spin][:, 0 : max_orb[spin]] for spin in [0, 1]]
        evaluator = orbitals.MoleculeOrbitalEvaluator(mol, mo_coeff, evaluate_orbitals_with)

    detcoeff, occup, det_map = determinant_tools.create_packed_objects(
        determinants, tol=tol
    )
    return detcoeff, occup, det_map, evaluator


def determinants_from_pyscf(mol, mf, mc=None, tol=-1):
    periodic = hasattr(mol, "a")
    if mc is None:
        determinants = single_determinant_from_mf(mf)
    elif periodic:
        determinants = pbc_determinants_from_casci(mc, cutoff=tol)

    if mc is not None and not periodic:
        determinants = interpret_ci(mc, tol)
    return determinants


def single_determinant_from_mf(mf, weight=1.0):
    """
    Creates a determinant list for a single determinant from SCF object
    """
    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)
    # When KRKS etc is not used, mo_occ is 2 dimensional.
    if hasattr(mf, "kpts") and len(np.asarray(mf.mo_occ).shape) == 3:
        occupation = [[list(np.nonzero(k > 0.5)[0]) for k in s] for s in mf.mo_occ]
    else:
        occupation = [list(np.nonzero(s > 0.5)[0]) for s in mf.mo_occ]
    return [(weight, occupation)]


def pbc_determinants_from_casci(mc, orbitals=None, cutoff=0.05):
    if hasattr(mc.ncore, "__len__"):
        nocc = [c + e for c, e in zip(mc.ncore, mc.nelecas)]
    else:
        nocc = [mc.ncore + e for e in mc.nelecas]
    if orbitals is None:
        orbitals = np.arange(mc.ncore, mc.ncore + mc.ncas)
    if not hasattr(orbitals[0], "__len__"):
        orbitals = [orbitals, orbitals]
    deters = pyscf.fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    determinants = []
    for x in deters:
        if abs(x[0]) > cutoff:
            allorbs = [
                [translate_occ(x[1], orbitals[0], nocc[0])],
                [translate_occ(x[2], orbitals[1], nocc[1])],
            ]
            determinants.append((x[0], allorbs))
    return determinants


def translate_occ(x, orbitals, nocc):
    a = determinant_tools.binary_to_occ(x, 0)[0]
    orbitals_without_active = list(range(nocc))
    for o in orbitals:
        if o in orbitals_without_active:
            orbitals_without_active.remove(o)
    return orbitals_without_active + [orbitals[i] for i in a]


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
        deters = pyscf.fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    return determinant_tools.reformat_binary_dets(deters, ncore=ncore, tol=tol)


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
