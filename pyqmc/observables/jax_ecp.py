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
from typing import NamedTuple
from functools import partial
from pyqmc.observables.eval_ecp import get_P_l, rnExp
import time

class ECPAccumulator:
    def __init__(self, mol, naip=None, stochastic_rotation=True, nselect_deterministic = None, nselect_random=None):
        """
        :parameter mol: PySCF molecule object
        :parameter float threshold: threshold for accepting nonlocal moves
        :parameter int naip: number of auxiliary integration points
        """
        self._atomic_coordinates = mol.atom_coords()
        self._ecp = mol._ecp
        self._atom_names = [atom[0] for atom in mol._atom]
        self.functors = [generate_ecp_functors(mol, at_name) for at_name in self._atom_names]

        if naip is None:
            naip = np.zeros(len(self.functors), dtype=int)
            for i, func in enumerate(self.functors):
                maxL = np.max(list(func.keys()))
                if maxL == -1: # only local
                    naip[i] = 0
                elif maxL == 0:
                    naip[i] = 6
                elif maxL == 1:
                    naip[i] = 6
                elif maxL == 2:
                    naip[i] = 12
        if isinstance(naip, int): # compatibility with old behavior
            naip = naip*np.zeros(len(self.functors), dtype=int)
        totaip = np.sum(naip)
        self.naip = naip

        self._vl_evaluator = partial(evaluate_vl, self.functors, self.naip)

        if nselect_deterministic is None:
            self.nselect_deterministic = np.max(naip)
        else:
            self.nselect_deterministic = nselect_deterministic
        if nselect_random is None:
            self.nselect_random = min(1, totaip - self.nselect_deterministic)
        else:
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
        move_info, local_part = self._vl_evaluator(self._atomic_coordinates, configs, e, self.stochastic_rotation)
        selected_moves = downselect_move_info(move_info, self.nselect_deterministic, self.nselect_random)
        epos_rot = (configs.configs[:, e, :][:,np.newaxis,:] \
                    - selected_moves.r_ea_vec) \
                    + selected_moves.r_ea_i
        epos = configs.make_irreducible(e, epos_rot)        
        # evaluate the wave function ratio
        ratio = np.asarray(wf.testvalue(e, epos)[0])

        weight = np.einsum("ijk, ijk -> ij", np.exp(-tau*move_info.v_l/move_info.P_l)-1, move_info.P_l)

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
    P_l: np.ndarray # (nconf, npoints, nl) auxiliary integration points for each atom and l
   

def evaluate_vl(vl_evaluator, # list of functors [at][l]
                naip, # number of auxiliary integration points [TODO should make this able to be a list]
                atomic_coordinates, 
                configs, 
                e,
                stochastic
                ):
    maxl=max([len(vl) for vl in vl_evaluator])
    distances = configs.dist.dist_i(atomic_coordinates[np.newaxis,:,:], configs.configs[:, e, :]) #nconf, natom, 3
    dist2 = np.sum(distances**2, axis=-1)
    dist = np.sqrt(dist2) #nconf, natom
    
    npoints = np.sum(naip)
    nconf = distances.shape[0]
    v_l = np.zeros((nconf,npoints, maxl))
    p_l = np.zeros((nconf,npoints, maxl))
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
        
        if naip[atom] == 0:
            continue
        pl_tmp, r_ea_tmp = get_P_l(dist[:,atom], distances[:,atom,:], vl.keys(), naip[atom], stochastic)

        nl = pl_tmp.shape[-1] # number of l's for this atom

        beg = np.sum(naip[:atom])
        end = np.sum(naip[:(atom+1)])
        v_l[:, beg:end, :nl] = v_tmp[:,np.newaxis,:] * pl_tmp
        p_l[:, beg:end, :nl] = pl_tmp  # nconf x naip x nl
        r_ea_i[:, beg:end, :] = r_ea_tmp
        prob_tmp = np.sum(v_tmp[:,:-1]**2, axis=-1)
        prob[:, beg:end]  = prob_tmp[:, np.newaxis]  # nconf x naip
        r_ea_vec[:, beg:end, :] = distances[:, atom, :][:, np.newaxis]  # nconf x naip x 3
        
    return _MoveInfo(r_ea_vec, 
                     r_ea_i,
                     prob,
                     v_l,
                     p_l
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

    normalized_probability = move_info.probability.copy()
    # Find the largest terms to evaluate deterministically
    indices_deterministic = np.argsort(normalized_probability, axis=1)[:,-nselect_deterministic:]  # nconf x nselect_deterministic
    np.put_along_axis(normalized_probability, indices_deterministic, 0.0, axis=1)  # set the deterministic indices to zero

    # Normalize the remaining probabilities for random selection    
    prob_norm = np.sum(normalized_probability, axis=1)
    normalized_probability[prob_norm==0,:] = 1.0/(npoints-nselect_deterministic)
    prob_norm[prob_norm==0] = 1.0  
    normalized_probability = normalized_probability/prob_norm[:,np.newaxis]  # renormalize the random evaluation

    # Select the random evaluation points
    cdf = np.cumsum(normalized_probability, axis=1)
    r = np.random.random((nconf, nselect_random))    
    indices_random = (r[:,np.newaxis, :] > cdf[:, :, np.newaxis]).sum(axis=1)  # nconf x npoints x nselect_random
    indices = np.concatenate((indices_deterministic, indices_random), axis=1)

    # the deterministic indices should not be multiplied by 1/probability
    np.put_along_axis(normalized_probability, indices_deterministic, 1.0/nselect_random, axis=1) 
    prob_selected = nselect_random*np.take_along_axis(normalized_probability, indices, axis=1)


    return _MoveInfo(
        r_ea_vec= np.take_along_axis(move_info.r_ea_vec, indices[:, :, np.newaxis], axis=1),
        r_ea_i=np.take_along_axis(move_info.r_ea_i, indices[:, :, np.newaxis], axis=1),
        probability=np.take_along_axis(move_info.probability, indices, axis=1),
        v_l=np.take_along_axis(move_info.v_l, indices[:,:,np.newaxis], axis=1)/prob_selected[:,:,np.newaxis], 
        P_l=np.take_along_axis(move_info.P_l, indices[:,:,np.newaxis], axis=1)
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