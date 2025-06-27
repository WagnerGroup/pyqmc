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
from pyqmc.observables.eval_ecp import get_P_l, ecp_mask, rnExp

class ECPAccumulator:
    def __init__(self, mol, threshold=10, naip=6, stochastic_rotation=True, nselect_deterministic = 6, nselect_random=4):
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
        self.nselect_deterministic = nselect_deterministic
        self.nselect_random = nselect_random
        self.stochastic_rotation = stochastic_rotation


    def __call__(self, configs, wf) -> dict:
        """
        :parameter configs: Configs object
        :parameter wf: wave function object
        :returns: dictionary ECP values
        """
        ecp_val = np.zeros(configs.configs.shape[0], dtype=wf.dtype)
        # gather energies and distances

        for e in range(configs.configs.shape[1]):
            move_info, local_part = self._vl_evaluator(self._atomic_coordinates, configs, e, self.stochastic_rotation)
            selected_moves = downselect_move_info(move_info, self.nselect_deterministic, self.nselect_random)

            epos_rot = (configs.configs[:, e, :][:,np.newaxis,:] \
                        - selected_moves.r_ea_vec) \
                        + selected_moves.r_ea_i
            epos = configs.make_irreducible(e, epos_rot)

            ecp_val += local_part # sum the local part
            # important to do the local sum before skipping evaluation 
            if move_info.probability.sum() == 0:
                continue
            # evaluate the wave function ratio
            ratio = wf.testvalue(e, epos)[0]

            # compute the ECP value
            # n: nconf, a: aip, l: angular momentum channel
            ecp_val += np.einsum("na,nal->n", ratio, selected_moves.v_l[:, :, :-1])  # skip the local part

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
    



class _MoveInfo(NamedTuple):
     # 'cheap to compute' quantities
     # npoints is the total number of potential integration points.
     # we will choose a subset of them based on N
    r_ea_vec: np.ndarray  # (nconf, npoints, 3) displacement from the atom to the electron
    r_ea_i: np.ndarray  #(nconf, npoints, 3) distance from the atom to the electron
    probability: np.ndarray # (nconf, npoints) probability we should evaluate this
    v_l: np.ndarray  # (nconf, npoints, nl) total weight (v(r)*P_l)
   

def evaluate_vl(vl_evaluator, # list of functors [at][l]
                threshold, # a number that determines the relative probability of each move
                naip, # number of auxiliary integration points [should make this able to be a list]
                atomic_coordinates, 
                configs, 
                e,
                stochastic
                ):
    maxl=max([len(vl) for vl in vl_evaluator])
    distances = configs.dist.dist_i(atomic_coordinates[np.newaxis,:,:], configs.configs[:, e, :]) #nconf, natom, 3
    dist2 = np.sum(distances**2, axis=-1)
    dist = np.sqrt(dist2) #nconf, natom
    
    npoints = naip * atomic_coordinates.shape[0]  # naip points for each atom (update this if we allow different numbers of aips)
    nconf = distances.shape[0]
    v_l = np.zeros((nconf,npoints, maxl))
    r_ea_i = np.zeros((nconf, npoints, 3))
    prob = np.zeros((nconf, npoints))
    r_ea_vec  = np.zeros((nconf, npoints, 3))  # displacement from the atom to the electron
    local_part = np.zeros(nconf)
    for atom, vl in enumerate(vl_evaluator):
        v_tmp = np.zeros((nconf, len(vl)))  # nconf x nl
        for l, func in vl.items():  # -1,0,1,...
            # a bit tricky here, we end up with the local part in the last column because
            # vl is a dictionary where -1 is the local part
            v_tmp[:,l] = func(dist[:, atom])
        
        local_part += v_tmp[:, -1]  # accumulate the local part
        #pl_tmp is 
        pl_tmp, r_ea_tmp = get_P_l(dist[:,atom], distances[:,atom,:], vl.keys(), naip, stochastic)

        v_l[:, atom * naip:(atom + 1) * naip, :] = v_tmp[:,np.newaxis,:] * pl_tmp

        r_ea_i[:, atom * naip:(atom + 1) * naip, :] = r_ea_tmp
        
        #_ , prob_tmp = ecp_mask(v_tmp, threshold)
        prob_tmp = np.sum(v_tmp[:,:-1]**2, axis=-1)
        #print(prob_tmp.shape)
        prob[:, atom*naip:(atom+1)*naip]  = prob_tmp[:, np.newaxis]  # nconf x naip
        r_ea_vec[:, atom * naip:(atom + 1) * naip, :] = distances[:, atom, :][:, np.newaxis]  # nconf x naip x 3
        
    return _MoveInfo(r_ea_vec, 
                     r_ea_i,
                     prob,
                     v_l,
    ), local_part

def downselect_move_info(move_info: _MoveInfo, nselect_deterministic: int, nselect_random) -> _MoveInfo:
    """
    Downselect the move_info to nselect points.
    :parameter move_info: _MoveInfo object
    :parameter nselect: number of points to select
    :returns: a new _MoveInfo object with only nselect points
    """
    nconf, npoints, nl = move_info.v_l.shape
    if nselect_random+nselect_deterministic >= npoints:
        return move_info  # no downselection needed

    normalized_probability = move_info.probability / np.sum(move_info.probability, axis=1, keepdims=True)

    # Find indices where r is less than the cumulative distribution function

    indices_deterministic = np.argsort(normalized_probability, axis=1)[:,-nselect_deterministic:]


    for i in range(nconf):
        normalized_probability[i,indices_deterministic[i]] = 0.0
    normalized_probability = normalized_probability/np.sum(normalized_probability, axis=1, keepdims=True)
    cdf = np.cumsum(normalized_probability, axis=1)
    r = np.random.random((nconf, nselect_random))    
    indices_random = np.array([ np.searchsorted(cdf[i], r[i]) for i in range(nconf) ])

    indices = np.concatenate((indices_deterministic, indices_random), axis=1)

    for i in range(nconf):
        normalized_probability[i,indices_deterministic[i]] = 1.0/nselect_random
    prob_selected = nselect_random*np.take_along_axis(normalized_probability, indices, axis=1)

    return _MoveInfo(
        r_ea_vec= np.take_along_axis(move_info.r_ea_vec, indices[:, :, np.newaxis], axis=1),
        r_ea_i=np.take_along_axis(move_info.r_ea_i, indices[:, :, np.newaxis], axis=1),
        probability=np.take_along_axis(move_info.probability, indices, axis=1),
        v_l=np.take_along_axis(move_info.v_l, indices[:,:,np.newaxis], axis=1)/prob_selected[:,:,np.newaxis], 
    )



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