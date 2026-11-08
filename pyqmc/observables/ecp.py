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

r"""Nonlocal pseudopotential (ECP) evaluation.

Two sampling strategies share the same interface and the same quadrature:

- :class:`ThresholdECP` loops over ECP atoms and randomly skips
  electron-atom pairs where the nonlocal potential is small. Only the surviving
  configurations evaluate the wave function, so the work varies per
  configuration.
- :class:`FixedBudgetECP` evaluates the same number of quadrature points for
  every configuration: the most important ``nselect_deterministic`` points plus
  ``nselect_random`` points sampled by importance and reweighted. The array
  shapes do not depend on the configurations.

Both are unbiased. Which is faster depends on the system and the wave function
backend.
"""

import copy
import functools
from typing import NamedTuple
import numpy as np
import scipy.spatial.transform
from pyqmc.rng import get_rng


def make_ecp(mol, sampling="threshold", threshold=10, naip=None, **kwargs):
    """Build an ECP evaluator.

    :parameter sampling: 'threshold' for :class:`ThresholdECP` or 'fixed' for
        :class:`FixedBudgetECP`.
    :parameter threshold: used by 'threshold' only.
    :parameter naip: number of quadrature points; see :func:`default_naip`.
    :parameter kwargs: passed to the evaluator.
    """
    if sampling == "threshold":
        return ThresholdECP(mol, threshold=threshold, naip=naip, **kwargs)
    if sampling == "fixed":
        return FixedBudgetECP(mol, naip=naip, **kwargs)
    raise ValueError(f"Unknown ECP sampling {sampling}; use 'threshold' or 'fixed'.")


def default_naip(functors, naip=None):
    """Number of quadrature points for each atom.

    By default, an atom with no nonlocal channels gets 0 points, an atom whose
    only nonlocal channel is s gets 6, and an atom with p or higher nonlocal
    channels gets 12. An integer naip is used for every atom with nonlocal
    channels. An array is used as given.

    :parameter functors: list over atoms of the dictionaries from
        :func:`generate_ecp_functors`.
    :returns: (natom,) integer array
    """
    has_nonlocal = np.array([max(f.keys(), default=-1) >= 0 for f in functors])
    if naip is None:
        maxl = np.array([max(f.keys(), default=-1) for f in functors])
        return np.where(maxl >= 1, 12, np.where(maxl == 0, 6, 0))
    if np.ndim(naip) == 0:
        return np.where(has_nonlocal, int(naip), 0)
    naip = np.asarray(naip, dtype=int)
    if naip.shape != (len(functors),):
        raise ValueError(f"naip has shape {naip.shape}; expected ({len(functors)},).")
    return naip


class ThresholdECP:
    """ECP evaluation that randomly skips electron-atom pairs with a small
    nonlocal potential.

    A pair is evaluated with probability ``min(1, p)``, where p grows with the
    nonlocal potential and with ``threshold``, and its contribution is divided by
    that probability. ``threshold <= 0`` evaluates every pair.
    """

    def __init__(self, mol, threshold=10, naip=None, stochastic_rotation=True):
        self.threshold = threshold
        self.stochastic_rotation = stochastic_rotation
        self._atoms = [
            (i, np.asarray(apos))
            for i, (name, apos) in enumerate(mol._atom)
            if name in mol._ecp
        ]
        self._functors = [
            generate_ecp_functors(mol, name) if name in mol._ecp else {}
            for name, _ in mol._atom
        ]
        self.naip = default_naip(self._functors, naip)
        self._has_ecp = mol._ecp != {}

    def __call__(self, configs, wf):
        """:returns: (nconf,) ECP energy, summed over electrons and atoms."""
        nconf, nelec = configs.configs.shape[0:2]
        ecp_tot = np.zeros(nconf, dtype=wf.dtype)
        for e in range(nelec):
            for atom, apos in self._atoms:
                ecp_tot += self._electron_atom(configs, wf, e, atom, apos)["total"]
        return ecp_tot

    def nonlocal_tmoves(self, configs, wf, e, tau):
        """For electron e, evaluate all possible T-moves.

        :returns: dictionary with
           ratio: psi(R')/psi(R) for each move
           weight: the symmetric part of Eqn 31 in Anderson and Umrigar, so that
               ratio * weight is the amplitude of the T-move
           configs: positions of the moves
        """
        nconf = configs.configs.shape[0]
        data = [
            self._electron_atom(configs, wf, e, atom, apos)
            for atom, apos in self._atoms
            if self.naip[atom] > 0
        ]
        if not data:
            return {"ratio": np.ones((nconf, 0)), "weight": np.zeros((nconf, 0))}

        ratios, weights = [], []
        for d in data:
            npts = d["ratio"].shape[1]
            weight = np.zeros((nconf, npts))
            ratio = np.ones((nconf, npts), dtype=d["ratio"].dtype)
            weight[d["mask"]] = np.einsum(
                "ik, ijk -> ij", np.exp(-tau * d["v_l"]) - 1, d["P_l"]
            )
            ratio[d["mask"]] = d["ratio"]
            ratios.append(ratio)
            weights.append(weight)

        configs = copy.copy(configs)
        configs.join([d["epos"] for d in data], axis=1)
        return {
            "ratio": np.concatenate(ratios, axis=1),
            "weight": np.concatenate(weights, axis=1),
            "configs": configs,
        }

    def has_nonlocal_moves(self):
        return self._has_ecp

    def _electron_atom(self, configs, wf, e, atom, apos):
        """ECP contribution of electron e and one atom, local plus nonlocal."""
        nconf = configs.configs.shape[0]
        ecp_val = np.zeros(nconf, dtype=wf.dtype)

        r_ea_vec = configs.dist.dist_i(apos, configs.configs[:, e, :]).reshape((-1, 3))
        r_ea = np.linalg.norm(r_ea_vec, axis=-1)

        functors = self._functors[atom]
        v_l = evaluate_v_l(functors, r_ea)
        local = v_l[:, -1]
        naip = self.naip[atom]
        if naip == 0:
            return {"total": ecp_val + local}

        mask, prob = self._mask(v_l)
        masked_v_l = v_l[mask]
        masked_v_l[:, :-1] /= prob[mask, np.newaxis]

        r_ea = r_ea[mask]
        r_ea_vec = r_ea_vec[mask]
        P_l, r_ea_i = get_P_l(r_ea, r_ea_vec, functors.keys(), naip, self.stochastic_rotation)

        # Note: epos_rot is not just apos+r_ea_i because of the boundary;
        # positions of the samples are relative to the electron, not atom.
        epos_rot = np.repeat(configs.configs[:, e, :][:, np.newaxis, :], naip, axis=1)
        epos_rot[mask] = (configs.configs[mask, e, :] - r_ea_vec)[:, np.newaxis] + r_ea_i
        epos = configs.make_irreducible(e, epos_rot, mask)

        if np.any(mask):
            ratio = wf.testvalue(e, epos, mask)[0]
        else:
            ratio = np.zeros(P_l.shape[:2])

        ecp_val[mask] = np.einsum("ij,ik,ijk->i", ratio, masked_v_l, P_l)
        return {
            "total": ecp_val + local,
            "v_l": masked_v_l,
            "P_l": P_l,
            "ratio": ratio,
            "epos": epos,
            "mask": mask,
        }

    def _mask(self, v_l):
        """:returns: which configurations to evaluate, and the probability of each."""
        if self.threshold > 0:
            l = 2 * np.arange(v_l.shape[1] - 1) + 1
            prob = np.dot(np.abs(v_l[:, :-1]), self.threshold * (2 * l + 1))
            prob = np.minimum(1, prob)
        else:
            prob = np.ones(v_l.shape[0])
        accept = prob > get_rng().random(size=prob.shape)
        return accept, prob


class FixedBudgetECP:
    """ECP evaluation with the same number of quadrature points for every
    configuration.

    For each electron, the quadrature points of all atoms are ranked by the size
    of the nonlocal potential. The top ``nselect_deterministic`` points (default:
    the largest per-atom naip) are evaluated exactly, and ``nselect_random``
    more (default 1) are sampled from the rest and reweighted.
    """

    def __init__(
        self,
        mol,
        naip=None,
        stochastic_rotation=True,
        nselect_deterministic=None,
        nselect_random=None,
    ):
        self._atomic_coordinates = mol.atom_coords()
        self._has_ecp = mol._ecp != {}
        self.functors = [
            generate_ecp_functors(mol, name) if name in mol._ecp else {}
            for name, _ in mol._atom
        ]
        self.naip = default_naip(self.functors, naip)
        totaip = int(np.sum(self.naip))
        maxaip = int(np.max(self.naip, initial=0))
        self.nselect_deterministic = (
            maxaip if nselect_deterministic is None else nselect_deterministic
        )
        self.nselect_random = (
            min(1, totaip - self.nselect_deterministic)
            if nselect_random is None
            else nselect_random
        )
        self.stochastic_rotation = stochastic_rotation

    def __call__(self, configs, wf):
        """:returns: (nconf,) ECP energy, summed over electrons and atoms."""
        ecp_val = np.zeros(configs.configs.shape[0], dtype=wf.dtype)
        for e in range(configs.configs.shape[1]):
            moves, epos, local_part = self._select_moves(configs, e)
            ecp_val += local_part
            # local part first: there is nothing nonlocal to add if every
            # point has zero weight
            if epos is None or not np.any(moves.probability):
                continue
            ratio = wf.testvalue(e, epos)[0]
            # n: nconf, a: quadrature point, l: angular momentum channel
            ecp_val += np.einsum("na,nal->n", ratio, moves.v_l[:, :, :-1])
        return ecp_val

    def nonlocal_tmoves(self, configs, wf, e, tau):
        """For electron e, evaluate the selected T-moves. See
        :meth:`ThresholdECP.nonlocal_tmoves` for the return value."""
        moves, epos, _ = self._select_moves(configs, e)
        nconf = configs.configs.shape[0]
        if epos is None:
            return {"ratio": np.ones((nconf, 0)), "weight": np.zeros((nconf, 0))}
        ratio = np.asarray(wf.testvalue(e, epos)[0])

        # moves.v_l is v_l(r) P_l divided by the selection probability, so
        # v_l / P_l recovers the reweighted v_l(r).
        expweight = np.zeros_like(moves.P_l)
        mask = np.abs(moves.P_l) > 0
        expweight[mask] = np.exp(-tau * moves.v_l[mask] / moves.P_l[mask]) - 1
        weight = np.einsum("ijk, ijk -> ij", expweight, moves.P_l)
        return {"ratio": ratio, "weight": weight, "configs": epos}

    def has_nonlocal_moves(self):
        return self._has_ecp

    def _select_moves(self, configs, e):
        """:returns: the selected moves, their positions (None if there are no
        nonlocal channels), and the local part of the ECP."""
        move_info, local_part = evaluate_move_info(
            self.functors,
            self.naip,
            self._atomic_coordinates,
            configs,
            e,
            self.stochastic_rotation,
        )
        if move_info.probability.shape[1] == 0:
            return move_info, None, local_part
        moves = downselect_move_info(
            move_info, self.nselect_deterministic, self.nselect_random
        )
        epos_rot = (configs.configs[:, e, :][:, np.newaxis, :] - moves.r_ea_vec) + moves.r_ea_i
        return moves, configs.make_irreducible(e, epos_rot), local_part


class _MoveInfo(NamedTuple):
    # npoints is the total number of quadrature points over all atoms.
    r_ea_vec: np.ndarray  # (nconf, npoints, 3) displacement from the atom to the electron
    r_ea_i: np.ndarray  # (nconf, npoints, 3) quadrature point relative to the atom
    probability: np.ndarray  # (nconf, npoints) importance of each point
    v_l: np.ndarray  # (nconf, npoints, nl) v_l(r) * P_l
    P_l: np.ndarray  # (nconf, npoints, nl) Legendre factors with quadrature weights


def evaluate_move_info(functors, naip, atomic_coordinates, configs, e, stochastic):
    """Quadrature points and weights of electron e for every atom.

    :returns: a _MoveInfo with the points of all atoms concatenated, and the
        (nconf,) local part of the ECP.
    """
    maxl = max([len(vl) for vl in functors])
    distances = configs.dist.dist_i(
        atomic_coordinates[np.newaxis, :, :], configs.configs[:, e, :]
    )  # nconf, natom, 3
    dist = np.linalg.norm(distances, axis=-1)  # nconf, natom

    npoints = np.sum(naip)
    nconf = distances.shape[0]
    v_l = np.zeros((nconf, npoints, maxl))
    p_l = np.zeros((nconf, npoints, maxl))
    r_ea_i = np.zeros((nconf, npoints, 3))
    prob = np.zeros((nconf, npoints))
    r_ea_vec = np.zeros((nconf, npoints, 3))
    local_part = np.zeros(nconf)
    for atom, vl in enumerate(functors):
        if len(vl) == 0:
            continue
        v_tmp = evaluate_v_l(vl, dist[:, atom])  # nconf x nl
        local_part += v_tmp[:, -1]
        if naip[atom] == 0:
            continue
        pl_tmp, r_ea_tmp = get_P_l(
            dist[:, atom], distances[:, atom, :], vl.keys(), naip[atom], stochastic
        )
        nl = pl_tmp.shape[-1]
        beg = np.sum(naip[:atom])
        end = beg + naip[atom]
        v_l[:, beg:end, :nl] = v_tmp[:, np.newaxis, :] * pl_tmp
        p_l[:, beg:end, :nl] = pl_tmp
        r_ea_i[:, beg:end, :] = r_ea_tmp
        prob[:, beg:end] = np.sum(v_tmp[:, :-1] ** 2, axis=-1)[:, np.newaxis]
        r_ea_vec[:, beg:end, :] = distances[:, atom, np.newaxis, :]

    return _MoveInfo(r_ea_vec, r_ea_i, prob, v_l, p_l), local_part


def downselect_move_info(move_info, nselect_deterministic, nselect_random):
    """Keep the nselect_deterministic most important points and sample
    nselect_random more by importance. The sampled points are reweighted by
    their selection probability, so the estimate stays unbiased.
    """
    nconf, npoints, nl = move_info.v_l.shape
    if nselect_random + nselect_deterministic >= npoints:
        return move_info

    normalized_probability = move_info.probability.copy()
    indices_deterministic = np.argsort(normalized_probability, axis=1)[
        :, -nselect_deterministic:
    ]
    np.put_along_axis(normalized_probability, indices_deterministic, 0.0, axis=1)

    prob_norm = np.sum(normalized_probability, axis=1)
    normalized_probability[prob_norm == 0, :] = 1.0 / (npoints - nselect_deterministic)
    prob_norm[prob_norm == 0] = 1.0
    normalized_probability = normalized_probability / prob_norm[:, np.newaxis]

    cdf = np.cumsum(normalized_probability, axis=1)
    r = get_rng().random((nconf, nselect_random))
    indices_random = (r[:, np.newaxis, :] > cdf[:, :, np.newaxis]).sum(axis=1)
    indices = np.concatenate((indices_deterministic, indices_random), axis=1)

    # the deterministic points are not reweighted
    np.put_along_axis(
        normalized_probability, indices_deterministic, 1.0 / nselect_random, axis=1
    )
    prob_selected = nselect_random * np.take_along_axis(
        normalized_probability, indices, axis=1
    )

    def take(a):
        return np.take_along_axis(a, indices[:, :, np.newaxis], axis=1)

    return _MoveInfo(
        r_ea_vec=take(move_info.r_ea_vec),
        r_ea_i=take(move_info.r_ea_i),
        probability=np.take_along_axis(move_info.probability, indices, axis=1),
        v_l=take(move_info.v_l) / prob_selected[:, :, np.newaxis],
        P_l=take(move_info.P_l),
    )


def evaluate_v_l(functors, r_ea):
    """:returns: (nconf, nl) array of v_l(r_ea), in the order l=0,1,...,local."""
    v_l = np.zeros((r_ea.shape[0], len(functors)))
    for l, func in functors.items():  # -1,0,1,...
        v_l[:, l] = func(r_ea)
    return v_l


def generate_ecp_functors(mol, at_name):
    """
    :returns: a dictionary of rnExp functors keyed by angular momentum:
      -1 is the local part, and 0,1,2,... are the s,p,d nonlocal channels.
    """
    d = {}
    for c in mol._ecp[at_name][1]:
        rn, exponent, coefficient = [], [], []
        for n, expand in enumerate(c[1]):
            for line in expand:
                rn.append(n - 2)
                exponent.append(line[0])
                coefficient.append(line[1])
        d[c[0]] = rnExp(rn, exponent, coefficient)
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


def get_P_l(r_ea, r_ea_vec, l_list, naip, stochastic=True):
    r"""The factor :math:`(2l+1)` and the quadrature weights are included.

    :parameter r_ea: distances of electron e and atom a
    :type r_ea: (nconf,)
    :parameter r_ea_vec: displacements of electron e and atom a
    :type r_ea_vec: (nconf, 3)
    :parameter list l_list: [-1,0,1,...] list of given angular momenta
    :parameter int naip: number of quadrature points
    :returns: legendre function P_l values for each :math:`l` channel.
    :rtype: (nconf, naip, nl) array
    """
    nconf = r_ea.shape[0]
    weights, rot_vec = get_rot(nconf, naip, stochastic)

    r_ea_i = r_ea[:, np.newaxis, np.newaxis] * rot_vec  # nmask x naip x 3
    rdotR = np.einsum("ik,ijk->ij", r_ea_vec, r_ea_i)
    rdotR /= r_ea[:, np.newaxis] * np.linalg.norm(r_ea_i, axis=-1)

    P_l_val = np.zeros((nconf, naip, len(l_list)))
    for l in l_list:
        P_l_val[:, :, l] = (2 * l + 1) * P_l(rdotR, l) * weights[np.newaxis]
    return P_l_val, r_ea_i


def get_rot(nconf, naip, stochastic=True):
    """
    :parameter int nconf: number of configurations
    :parameter int naip: number of auxiliary integration points
    :returns: the integration weights, and the positions of the rotated electron e
    :rtype:  ((naip,) array, (nconf, naip, 3) array)
    """

    rot = scipy.spatial.transform.Rotation.random(random_state=get_rng()).as_matrix()
    quadrature_grid = generate_quadrature_grids()

    if naip not in quadrature_grid.keys():
        raise ValueError(f"Possible AIPs are one of {quadrature_grid.keys()}")
    points, weights = quadrature_grid[naip]
    if nconf == 0:
        return weights, np.zeros((0, naip, 3))
    if stochastic:
        rot_vec = np.dot(rot, points.T).T[np.newaxis, :, :]  # nconf x naip x 3
    else:
        rot_vec = points[np.newaxis, :, :]
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
