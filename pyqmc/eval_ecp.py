import numpy as np
import copy
import scipy.spatial.transform


def ecp(mol, configs, wf, threshold, naip=None):
    """
    :returns: ECP value, summed over all the electrons and atoms.
    """
    nconf, nelec = configs.configs.shape[0:2]
    ecp_tot = np.zeros(nconf, dtype=complex if wf.iscomplex else float)
    if mol._ecp != {}:
        for atom in mol._atom:
            if atom[0] in mol._ecp.keys():
                for e in range(nelec):
                    ecp_tot += ecp_ea(mol, configs, wf, e, atom, threshold, naip)[
                        "total"
                    ]
    return ecp_tot


def compute_tmoves(mol, configs, wf, e, threshold, tau, naip=None):
    """
    For a given electron, evaluate all possible t-moves.

    returns a dictionary: 
       ratio: psi(R')/psi(R) for each move
       weight: The symmetric part of Eqn 31 in Anderson and Umrigar (i.e., ratio * weight gives the amplitude for the t-move)
       configs: positions of the move
    """
    nconfig = configs.configs.shape[0]
    if mol._ecp != {}:
        data = [
            ecp_ea(mol, configs, wf, e, atom, threshold, naip)
            for atom in mol._atom
            if atom[0] in mol._ecp.keys()
        ]
    else:
        return {"ratio": np.ones((nconfig, 0)), "weight": np.zeros((nconfig, 0))}

    # we want to make a data set which is a list of possible positions, the wave function
    # ratio, and the masks for each
    summed_data = []
    nconfig = configs.configs.shape[0]
    for d in data:
        npts = d["ratio"].shape[1]
        weight = np.zeros((nconfig, npts))
        ratio = np.ones((nconfig, npts), dtype=d["ratio"].dtype)
        weight[d["mask"]] = np.einsum(
            "ik, ijk -> ij", np.exp(-tau * d["v_l"]) - 1, d["P_l"]
        )
        ratio[d["mask"]] = d["ratio"]
        summed_data.append({"weight": weight, "ratio": ratio, "epos": d["epos"]})

    ratio = np.concatenate([d["ratio"] for d in summed_data], axis=1)
    weight = np.concatenate([d["weight"] for d in summed_data], axis=1)
    configs = copy.copy(configs)
    configs.join([d["epos"] for d in summed_data], axis=1)
    return {"ratio": ratio, "weight": weight, "configs": configs}


def ecp_ea(mol, configs, wf, e, atom, threshold, naip=None):
    """
    :returns: the ECP value between electron e and atom at, local+nonlocal.
    TODO: update documentation
    """
    nconf = configs.configs.shape[0]
    ecp_val = np.zeros(nconf, dtype=complex if wf.iscomplex else float)

    at_name, apos = atom
    apos = np.asarray(apos)

    r_ea_vec = configs.dist.dist_i(apos, configs.configs[:, e, :]).reshape((-1, 3))
    r_ea = np.linalg.norm(r_ea_vec, axis=-1)

    l_list, v_l = get_v_l(mol, at_name, r_ea)
    mask, prob = ecp_mask(v_l, threshold)
    masked_v_l = v_l[mask]
    masked_v_l[:, :-1] /= prob[mask, np.newaxis]

    # Use masked objects internally
    r_ea = r_ea[mask]
    r_ea_vec = r_ea_vec[mask]
    P_l, r_ea_i = get_P_l(r_ea, r_ea_vec, l_list, naip)

    # Note: epos_rot is not just apos+r_ea_i because of the boundary;
    # positions of the samples are relative to the electron, not atom.
    epos_rot = np.repeat(
        configs.configs[:, e, :][:, np.newaxis, :], P_l.shape[1], axis=1
    )
    epos_rot[mask] = (configs.configs[mask, e, :] - r_ea_vec)[:, np.newaxis] + r_ea_i

    epos = configs.make_irreducible(e, epos_rot, mask)
    ratio, _ = wf.testvalue(e, epos, mask)

    # Compute local and non-local parts
    ecp_val[mask] = np.einsum("ij,ik,ijk->i", ratio, masked_v_l, P_l)
    ecp_val += v_l[:, -1]  # local part
    return {
        "total": ecp_val,
        "v_l": masked_v_l,
        "local": v_l[:, -1],
        "P_l": P_l,
        "ratio": ratio,
        "epos": epos,
        "mask": mask,
    }


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


def generate_ecp_functors(coeffs):
    """
    :parameter coeffs: `mol._ecp[atom_name][1]` (coefficients of the ECP)
    :returns: a functor v_l, with keys as the angular momenta:
      -1 stands for the nonlocal part, 0,1,2,... are the s,p,d channels, etc.
    """
    d = {}
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
    if naip is None:
        naip = 6 if len(l_list) <= 2 else 12
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

    if nconf > 0:  # get around a bug(?) when there are zero configurations.
        rot = scipy.spatial.transform.Rotation.random(nconf).as_matrix()
    else:
        rot = np.zeros((0, 3, 3))

    if naip not in quadrature_grid.keys():
        raise ValueError("Do not support naip!= 6 or 12")
    points, weights = quadrature_grid[naip]
    rot_vec = np.einsum("jkl,ik->jil", rot, points)
    return weights, rot_vec


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


quadrature_grid = generate_quadrature_grids()
