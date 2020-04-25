import numpy as np
import copy


def ecp(mol, configs, wf, threshold):
    """
    Returns the ECP value, summed over all the electrons and atoms.
    """
    nconf, nelec = configs.configs.shape[0:2]
    ecp_tot = np.zeros(nconf, dtype=complex if wf.iscomplex else float)
    if mol._ecp != {}:
        for atom in mol._atom:
            if atom[0] in mol._ecp.keys():
                for e in range(nelec):
                    ecp_tot += ecp_ea(mol, configs, wf, e, atom, threshold)
    return ecp_tot


def ecp_ea(mol, configs, wf, e, atom, threshold):
    """ 
    Returns the ECP value between electron e and atom at, local+nonlocal.
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
    P_l, r_ea_i = get_P_l(r_ea, r_ea_vec, l_list)

    # Note: epos_rot is not just apos+r_ea_i because of the boundary;
    # positions of the samples are relative to the electron, not atom.
    epos_rot = (configs.configs[mask, e, :] - r_ea_vec)[:, np.newaxis] + r_ea_i

    # Expand externally
    expanded_epos_rot = np.zeros((nconf, P_l.shape[1], 3))
    expanded_epos_rot[mask] = epos_rot
    epos = configs.make_irreducible(e, expanded_epos_rot)
    ratio = wf.testvalue(e, epos, mask)

    # Compute local and non-local parts
    ecp_val[mask] = np.einsum("ij,ik,ijk->i", ratio, masked_v_l, P_l)
    ecp_val += v_l[:, -1]  # local part
    return ecp_val


def ecp_mask(v_l, threshold):
    """
    Returns a mask for configurations sized nconf
    based on values of v_l. Also returns acceptance probabilities
    """
    l = 2 * np.arange(v_l.shape[1] - 1) + 1
    prob = np.dot(np.abs(v_l[:, :-1]), threshold * (2 * l + 1))
    prob = np.minimum(1, prob)
    accept = prob > np.random.random(size=prob.shape)
    return accept, prob


def get_v_l(mol, at_name, r_ea):
    """
    Returns list of the l's, and a nconf x nl array, v_l values for each l: l= 0,1,2,...,-1
    """
    vl = generate_ecp_functors(mol._ecp[at_name][1])
    v_l = np.zeros([r_ea.shape[0], len(vl)])
    for l, func in vl.items():  # -1,0,1,...
        v_l[:, l] = func(r_ea)
    return vl.keys(), v_l


def generate_ecp_functors(coeffs):
    """
    Returns a functor, with keys as the angular momenta:
    -1 stands for the nonlocal part, 0,1,2,... are the s,p,d channels, etc.
    Parameters: 
      mol._ecp[atom_name][1] (coefficients of the ECP)
    Returns:
      v_l function, with key = angular momentum
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
    """
    v_l object. :math:`c*r^{n-2}*exp(-e*r^2)`
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
    """
    Legendre functions,
    returns a nconf x naip array for a given l, x=r_ea(i)
    Parameters:
      x: nconf array, l: integer
    Returns:
      P_l values: nconf x naip array
    """
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
        return np.zeros(x.shape)


def get_P_l(r_ea, r_ea_vec, l_list):
    """
    Returns a nconf x naip x nl array, which is the legendre function values for each l channel.
    The factor (2l+1) and the quadrature weights are included.
    Parameters:
      l_list: [-1,0,1,...] list of given angular momenta
      weights: integration weights
    Return:
      P_l values: nconf x naip x nl array  
    """
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
    Returns the integration weights (naip), and the positions of the rotated electron e (nconf x naip x 3)
    Parameters: 
      configs[:,e,:]: epos of the electron e to be rotated
    Returns:
      weights: naip array
      epos_rot: positions of the rotated electron, nconf x naip x 3
      
    """
    # t and p are sampled randomly over a sphere around the atom
    t = np.random.uniform(low=0.0, high=np.pi, size=nconf)
    p = np.random.uniform(low=0.0, high=2 * np.pi, size=nconf)

    def sphere(t_, p_):
        s = np.sin(t_)
        return s * np.cos(p_), s * np.sin(p_), np.cos(t_)

    # rotated unit vectors:
    rot = np.zeros([3, 3, nconf])
    rot[0, :, :] = sphere(np.zeros(nconf) + np.pi / 2.0, p - np.pi / 2.0)
    rot[1, :, :] = sphere(t + np.pi / 2.0, p)
    rot[2, :, :] = sphere(t, p)

    if naip == 6:
        d1 = np.array([0.0, 1.0, 0.5, 0.5, 0.5, 0.5]) * np.pi
        d2 = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.5]) * np.pi
    elif naip == 12:
        tha = np.arccos(1.0 / np.sqrt(5.0))
        d1 = np.array([0, np.pi] + [tha, np.pi - tha] * 5)
        d2 = np.array([0, 0] + list(range(10))) * np.pi / 5

    rot_vec = np.einsum("ilj,ik->jkl", rot, sphere(d1, d2))
    weights = 1.0 / naip * np.ones(naip)

    return weights, rot_vec
