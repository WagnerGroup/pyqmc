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
import copy
import scipy.spatial.transform
import functools
import time
import logging
from typing import NamedTuple
from collections import namedtuple
from functools import partial

class ECPAccumulator:
    def __init__(self, mol, threshold=10, naip=6):
        """
        :parameter mol: PySCF molecule object
        :parameter float threshold: threshold for accepting nonlocal moves
        :parameter int naip: number of auxiliary integration points
        """
        self.threshold = threshold
        if naip is None:
            naip = 6
        self.naip = naip
        self._atomic_coordinates = mol.atom_coords()
        self._ecp = mol._ecp
        self._atom_names = [atom[0] for atom in mol._atom]
        self.functors = [generate_ecp_functors(mol, at_name) for at_name in self._atom_names]
        self._vl_evaluator = partial(evaluate_vl, self.functors, self.threshold, self.naip)
        print(self.functors)


    def __call__(self, configs, wf) -> dict:
        """
        :parameter configs: Configs object
        :parameter wf: wave function object
        :returns: dictionary ECP values
        """
        ecp_val = np.zeros(configs.configs.shape[0], dtype=wf.dtype)
        # gather energies and distances

        for e in range(configs.configs.shape[1]):
            atomic_info, v_local = gather_atomic(self._atomic_coordinates, configs, e, self.functors)
            #print("e, atomic_info", e, atomic_info)
            move_info = self._vl_evaluator(atomic_info)
            #print("move_info", move_info)
            # get the electronic coordinates by moving the electron to the atom and then to the integration point
            # this gives us PBCs correctly
            epos_rot = (configs.configs[:, e, :] \
                        - atomic_info.r_ea_vec)[:, np.newaxis] \
                        + move_info.r_ea_i
            epos = configs.make_irreducible(e, epos_rot, move_info.mask)

            ecp_val += v_local
            # important to do the local sum before skipping evaluation 
            if move_info.mask.sum() == 0:
                continue
            # evaluate the wave function ratio
            ratio = wf.testvalue(e, epos, move_info.mask)[0]

            # compute the ECP value
            # n: nconf, a: aip, l: angular momentum channel
            ecp_val[move_info.mask] += np.einsum("na,nal, nl->n", ratio, move_info.P_l[move_info.mask], move_info.v_l[move_info.mask])

        return ecp_val


    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def nonlocal_tmoves(self, configs, wf, e, tau):
        atomic_info = gather_atomic(self._atomic_coordinates, configs, e)
        move_info = self._vl_evaluator(atomic_info)
        epos_rot = (configs.configs[:, e, :] - atomic_info.r_ea_vec)[:, np.newaxis] + move_info.r_ea_i
        epos = configs.make_irreducible(e, epos_rot, move_info.mask)
        # evaluate the wave function ratio
        ratio = np.zeros((configs.configs.shape[0], self.naip))
        ratio[move_info.mask,:] = wf.testvalue(e, epos, move_info.mask)[0]

        weight = np.einsum("ik, ijk -> ij", np.exp(-tau*move_info.v_l)-1, move_info.P_l)

        return {'ratio': ratio, 'weight': weight, 'configs':epos} 

    def has_nonlocal_moves(self):
        return self._ecp != {}

    def keys(self):
        return set(["ecp"])

    def shapes(self):
        return {"ecp": ()}
    


class _AtomicInfo(NamedTuple):
    r_ea: np.ndarray
    r_ea_vec: np.ndarray
    assigned_atom: np.ndarray

def gather_atomic(atomic_coordinates, configs, e, vl_evaluator):
    """
    :parameter atomic_coordinates: atomic coordinates
    :parameter configs: Configs object
    :parameter e: electron index
    :returns: MoveInformation object
    """
    distances = configs.dist.dist_i(atomic_coordinates[np.newaxis,:,:], configs.configs[:, e, :])
    dist2 = np.sum(distances**2, axis=-1)
    dist = np.sqrt(dist2)
    v_local = np.zeros(distances.shape[0])
    for atom, vl in enumerate(vl_evaluator):
        if vl is None or -1 not in vl:
            continue
        v_local += vl[-1](dist[:, atom])  # local part
    # Reduce to (nconf, 3) array for closest atom
    assigned_atom = np.argmin(dist2, axis=1)
    r_ea_vec = distances[np.arange(distances.shape[0]),assigned_atom,:]
    r_ea = np.linalg.norm(r_ea_vec, axis=-1)
    return _AtomicInfo(r_ea, r_ea_vec, assigned_atom), v_local


class _MoveInfo(NamedTuple):
     # 'cheap to compute' quantities
    r_ea_i: np.ndarray  #(nconf, naip, 3)
    probability: np.ndarray # (nconf,)
    mask: np.ndarray  # (nconf,)
    v_l: np.ndarray  # (nconf, nl)
    P_l: np.ndarray # (nconf, naip, nl)
   

def evaluate_vl(vl_evaluator, # list of functors [at][l]
                threshold,
                naip,
                at_info: _AtomicInfo,
                ):
    maxl=max([len(vl) for vl in vl_evaluator])
    nconf = at_info.r_ea.shape[0]
    v_l = np.zeros((nconf,maxl))
    P_l = np.zeros((nconf, naip, maxl))
    r_ea_i = np.zeros((nconf, naip, 3))
    mask = np.zeros((nconf), dtype=bool)
    prob = np.zeros((nconf))
    for atom, vl in enumerate(vl_evaluator):
        m_atom = at_info.assigned_atom == atom
        for l, func in vl.items():  # -1,0,1,...
            # a bit tricky here, we end up with the local part in the last column because
            # vl is a dictionary where -1 is the local part
            v_l[m_atom, l] = func(at_info.r_ea[m_atom]) 
        mask[m_atom], prob[m_atom] = ecp_mask(v_l[m_atom,:], threshold)
        #print(atom, vl.keys())
        nl = len(vl)
        P_l[m_atom,:,:nl], r_ea_i[m_atom,:,:] = get_P_l(at_info.r_ea[m_atom], at_info.r_ea_vec[m_atom], vl.keys(), naip)
        

    #blank = np.arange(nconf)
    return _MoveInfo(r_ea_i,
                     prob,
                     mask,
                     v_l,
                     P_l,
    )



def ecp_mask(v_l, threshold):
    """
    :returns: a mask for configurations sized nconf based on values of v_l. Also returns acceptance probabilities
    """
    l = 2 * np.arange(v_l.shape[1] - 1) + 1
    prob = np.dot(np.abs(v_l[:, :-1]), threshold * (2 * l + 1))
    prob = np.minimum(1, prob)
    accept = prob > np.random.random(size=prob.shape)
    return accept, prob


def get_v_l(mol, at_name, r_ea):
    r"""
    :returns: list of the :math:`l`'s, and a nconf x nl array, v_l values for each :math:`l`: l= 0,1,2,...,-1
    """
    vl = generate_ecp_functors(mol._ecp[at_name][1])
    v_l = np.zeros([r_ea.shape[0], len(vl)])
    for l, func in vl.items():  # -1,0,1,...
        v_l[:, l] = func(r_ea)
    return vl.keys(), v_l


def generate_ecp_functors(mol, at_name):
    """
    :parameter coeffs: `mol._ecp[atom_name][1]` (coefficients of the ECP)
    :returns: a functor v_l, with keys as the angular momenta:
      -1 stands for the local part, 0,1,2,... are the s,p,d channels, etc.
    """
    d = {}
    if at_name not in mol._ecp:
        return d
    coeffs = mol._ecp[at_name][1]
    for c in coeffs:
        el = c[0]
        rn = []
        exponent = []
        coefficient = []
        for n, expand in enumerate(c[1]):
            # print("r",n-2,"coeff",expand)
            for line in expand:
                rn.append(n - 2)
                exponent.append(line[0])
                coefficient.append(line[1])
        d[el] = rnExp(rn, exponent, coefficient)
    return d


class rnExp:
    r"""
    v_l object.

    :math:`cr^{n-2}\cdot\exp(-er^2)`
    """

    def __init__(self, n, e, c):
        self.n = np.asarray(n)
        self.e = np.asarray(e)
        self.c = np.asarray(c)

    def __call__(self, r):
        return np.sum(
            r[:, np.newaxis] ** self.n
            * self.c
            * np.exp(-self.e * r[:, np.newaxis] ** 2),
            axis=1,
        )


def P_l(x, l):
    r"""Legendre functions,

    :parameter  x: distances x=r_ea(i)
    :type x: (nconf,) array
    :parameter int l: angular momentum channel
    :returns: legendre function P_l values for channel :math:`l`.
    :rtype: (nconf, naip) array
    """
    if l == -1:
        return np.zeros(x.shape)
    if l == 0:
        return np.ones(x.shape)
    elif l == 1:
        return x
    elif l == 2:
        return 0.5 * (3 * x * x - 1)
    elif l == 3:
        return 0.5 * (5 * x * x * x - 3 * x)
    elif l == 4:
        return 0.125 * (35 * x * x * x * x - 30 * x * x + 3)
    else:
        raise NotImplementedError(f"Legendre functions for l>4 not implemented {l}")


def get_P_l(r_ea, r_ea_vec, l_list, naip=None):
    r"""The factor :math:`(2l+1)` and the quadrature weights are included.

    :parameter r_ea: distances of electron e and atom a
    :type r_ea: (nconf,)
    :parameter r_ea_vec: displacements of electron e and atom a
    :type r_ea_vec: (nconf, 3)
    :parameter list l_list: [-1,0,1,...] list of given angular momenta
    :returns: legendre function P_l values for each :math:`l` channel.
    :rtype: (nconf, naip, nl) array
    """
    nconf = r_ea.shape[0]
    weights, rot_vec = get_rot(nconf, naip)

    r_ea_i = r_ea[:, np.newaxis, np.newaxis] * rot_vec  # nmask x naip x 3
    rdotR = np.einsum("ik,ijk->ij", r_ea_vec, r_ea_i)
    rdotR /= r_ea[:, np.newaxis] * np.linalg.norm(r_ea_i, axis=-1)

    P_l_val = np.zeros((nconf, naip, len(l_list)))
    # already included the factor (2l+1), and the integration weights here
    for l in l_list:
        P_l_val[:, :, l] = (2 * l + 1) * P_l(rdotR, l) * weights[np.newaxis]
    return P_l_val, r_ea_i


def get_rot(nconf, naip):
    """
    :parameter int nconf: number of configurations
    :parameter int naip: number of auxiliary integration points
    :returns: the integration weights, and the positions of the rotated electron e
    :rtype:  ((naip,) array, (nconf, naip, 3) array)
    """

    #if nconf > 0:  # get around a bug(?) when there are zero configurations.
        #rot = scipy.spatial.transform.Rotation.random(nconf).as_matrix()
        #rot = np.identity(3)[np.newaxis].repeat(nconf, axis=0)
    rot = scipy.spatial.transform.Rotation.random().as_matrix()
    #else:
    #    rot = np.zeros((0, 3, 3))
    quadrature_grid = generate_quadrature_grids()

    if naip not in quadrature_grid.keys():
        raise ValueError(f"Possible AIPs are one of {quadrature_grid.keys()}, got {naip} instead.")
    points, weights = quadrature_grid[naip]
    #rot_vec = np.einsum("jkl,ik->jil", rot, points)
    rot_vec = np.einsum("kl, ik->il", rot, points)[np.newaxis]
    return weights, rot_vec


@functools.lru_cache(maxsize=1)
def generate_quadrature_grids():
    """
    Generate quadrature grids from Mitas, Shirley, and Ceperley J. Chem. Phys. 95, 3467 (1991)
        https://doi.org/10.1063/1.460849
    All the grids in the Mitas paper are hard-coded here.
    Returns a dictionary whose keys are naip (number of auxiliary points) and whose values are tuples of arrays (points, weights)
    """
    # Generate in Cartesian grids for octahedral symmetry
    octpts = np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
    nonzero_count = np.count_nonzero(octpts, axis=1)
    OA = octpts[nonzero_count == 1]
    OB = octpts[nonzero_count == 2] / np.sqrt(2)
    OC = octpts[nonzero_count == 3] / np.sqrt(3)
    d1 = OC * np.sqrt(3 / 11)
    d1[:, 2] *= 3
    OD = np.concatenate([np.roll(d1, i, axis=1) for i in range(3)])
    OAB = np.concatenate([OA, OB], axis=0)
    OABC = np.concatenate([OAB, OC], axis=0)
    OABCD = np.concatenate([OABC, OD], axis=0)

    # Generate in spherical grids for octahedral symmetry
    def sphere(t_, p_):
        s = np.sin(t_)
        return s * np.cos(p_), s * np.sin(p_), np.cos(t_)

    b_1 = np.arctan(2)
    c_1 = np.arccos((2 + 5**0.5) / (15 + 6 * 5**0.5) ** 0.5)
    c_2 = np.arccos(1 / (15 + 6 * 5**0.5) ** 0.5)
    theta, phi = {}, {}
    theta["A"] = np.array([0, np.pi])
    phi["A"] = np.zeros(2)
    k = np.arange(10)
    theta["B"] = np.tile([b_1, np.pi - b_1], 5)
    phi["B"] = k * np.pi / 5
    c_th1 = np.tile([np.pi - c_1, c_1], 5)
    c_th2 = np.tile([np.pi - c_2, c_2], 5)
    theta["C"] = np.concatenate([c_th1, c_th2])
    phi["C"] = np.tile(k * np.pi / 5, 2)
    I = {g: np.transpose(sphere(theta[g], phi[g])) for g in "ABC"}
    IAB = np.concatenate([I["A"], I["B"]], axis=0)
    IABC = np.concatenate([IAB, I["C"]], axis=0)

    lens = {}
    lens["O"] = [len(x) for x in [OA, OB, OC, OD]]
    lens["I"] = [len(I[s]) for s in "ABC"]

    def repeat(s, *args):
        return np.concatenate([np.repeat(w, l) for w, l in zip(args, lens[s])])

    qgrid = {}
    qgrid[6] = (OA, repeat("O", 1 / 6))
    qgrid[18] = (OAB, repeat("O", 1 / 30, 1 / 15))
    qgrid[26] = (OABC, repeat("O", 1 / 21, 4 / 105, 27 / 840))
    qgrid[50] = (OABCD, repeat("O", 4 / 315, 64 / 2835, 27 / 1280, 14641 / 725760))
    qgrid[12] = (IAB, repeat("I", 1 / 12, 1 / 12))
    qgrid[32] = (IABC, repeat("I", 5 / 168, 5 / 168, 27 / 840))

    return qgrid
