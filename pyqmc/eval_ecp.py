import numpy as np
import copy

"""
v_l object. c*r^{n-2}*exp{-e*r^2} 
"""


class rnExp:
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


#########################################################################
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
        return 0.5 * (3 * x * x - np.ones(x.shape))
    elif l == 3:
        return 0.5 * (5 * x * x * x - 3 * x)
    elif l == 4:
        return 0.125 * (35 * x * x * x * x - 30 * x * x + 3 * np.ones(x.shape))
    else:
        return np.zeros(x.shape)


def get_r_ea(mol, configs, e, at):
    """
    Returns a nconf x 3 array, distances between electron e and atom at
    Parameters:
      e,at: integers, eletron and atom indices
      configs: nconf x nelec x 3 array
    Returns:
      epos-apos, electron-atom distances
    """
    epos = configs.configs[:, e, :]
    nconf = configs.configs.shape[0]
    apos = np.outer(
        np.ones(nconf), np.array(mol._atom[at][1])
    )  # nconf x 3 array, position of atom at
    return epos - apos


def get_r_ea_i(mol, epos_rot, e, at):
    """
    Returns a nconf x naip x 3 array, distances between the rotated electron (e) and the atom at
    Parameters:
      epos_rot: configs object with rotated positions of electron e, nconf x naip x 3
    Returns:
      epos_rot-apos, (rotated) electron-atom distances
    """
    nconf, naip = epos_rot.shape[0:2]
    apos = np.zeros(
        [nconf, naip, 3]
    )  # position of the atom, broadcasted into nconf x naip x 3
    for aip in range(naip):
        apos[:, aip, :] = np.outer(np.ones(nconf), np.array(mol._atom[at][1]))
    return epos_rot- apos


def get_v_l(mol, configs, e, at):
    """
    Returns list of the l's, and a nconf x nl array, v_l values for each l: l= 0,1,2,...,-1
    """
    nconf = configs.configs.shape[0]
    at_name = mol._atom[at][0]
    r_ea = np.linalg.norm(get_r_ea(mol, configs, e, at), axis=1)
    vl = generate_ecp_functors(mol._ecp[at_name][1])
    Lmax = len(vl)
    v_l = np.zeros([nconf, Lmax])
    for l in vl.keys():  # -1,0,1,...
        v_l[:, l] = vl[l](r_ea)
    return vl.keys(), v_l


def get_wf_ratio(wf, configs, epos_rot, e, mask):
    """
    Returns a nconf x naip array, which is the Psi(r_e(i))/Psi(r_e) values
    """
    nconf, naip = epos_rot.shape[0:2]
    wf_ratio = np.zeros([nconf, naip])
    for aip in range(naip):
        epos = configs.make_irreducible(e, epos_rot[:,aip,:])
        wf_ratio[:, aip] = wf.testvalue_mask(e, epos, mask)
    return wf_ratio


def get_P_l(mol, configs, weights, epos_rot, l_list, e, at):
    """
    Returns a nconf x naip x nl array, which is the legendre function values for each l channel.
    The factor (2l+1) and the quadrature weights are included.
    Parameters:
      l_list: [-1,0,1,...] list of given angular momenta
      weights: integration weights
    Return:
      P_l values: nconf x naip x nl array  
    """
    # at_name = mol._atom[at][0]
    nconf, naip = epos_rot.shape[0:2]

    P_l_val = np.zeros([nconf, naip, len(l_list)])
    r_ea = get_r_ea(mol, configs, e, at)  # nconf x 3
    r_ea_i = get_r_ea_i(mol, epos_rot, e, at)  # nconf x naip x 3
    rdotR = np.zeros(r_ea_i.shape[0:2])  # nconf x naip

    # get the cosine values
    for aip in range(naip):
        rdotR[:, aip] = (
            r_ea[:, 0] * r_ea_i[:, aip, 0]
            + r_ea[:, 1] * r_ea_i[:, aip, 1]
            + r_ea[:, 2] * r_ea_i[:, aip, 2]
        )
        rdotR[:, aip] /= np.linalg.norm(r_ea, axis=1) * np.linalg.norm(
            r_ea_i[:, aip, :], axis=1
        )
    # print('cosine values',rdotR)

    # already included the factor (2l+1), and the integration weights here
    for l in l_list:
        P_l_val[:, :, l] = (
            (2 * l + 1) * P_l(rdotR, l) * np.outer(np.ones(nconf), weights)
        )
    return P_l_val


#########################################################################


def ecp_ea(mol, configs, wf, e, at, mask):
    """ 
    Returns the ECP value between electron e and atom at, local+nonlocal.
    """
    l_list, v_l = get_v_l(mol, configs, e, at)
    naip = 6
    if len(l_list) > 2:
        naip = 12

    weights, epos_rot = get_rot(mol, configs, e, at, naip)
    P_l = get_P_l(mol, configs, weights, epos_rot, l_list, e, at)
    ratio = get_wf_ratio(wf, configs, epos_rot, e, mask)
    ecp_val = np.einsum("ij,ik,ijk->i", ratio, v_l, P_l)
    # compute the local part
    local_l = -1
    ecp_val += v_l[:, local_l]
    return ecp_val


def ecp(mol, configs, wf, cutoff):
    """
    Returns the ECP value, summed over all the electrons and atoms.
    """
    nconf, nelec = configs.configs.shape[0:2]
    ecp_tot = np.zeros(nconf)
    if mol._ecp != {}:
        for e in range(nelec):
            for at in range(len(mol._atom)):
                r = get_r_ea(mol,configs,e,at)
                mask = (r[:,0]**2 + r[:,1]**2 + r[:,2]**2) < cutoff**2
                masked_configs = configs.mask(mask)
                ecp_tot[mask] += ecp_ea(mol, masked_configs, wf, e, at, mask)
    return ecp_tot


#################### Quadrature Rules ############################
def get_rot(mol, configs, e, at, naip):
    """
    Returns the integration weights (naip), and the positions of the rotated electron e (nconf x naip x 3)
    Parameters: 
      configs[:,e,:]: epos of the electron e to be rotated
    Returns:
      weights: naip array
      epos_rot: positions of the rotated electron, nconf x naip x 3
      
    """
    nconf = configs.configs.shape[0]
    apos = np.outer(np.ones(nconf), np.array(mol._atom[at][1]))

    r_ea_vec = get_r_ea(mol, configs, e, at)
    r_ea = np.linalg.norm(r_ea_vec, axis=1)

    # t and p are sampled randomly over a sphere around the atom
    t = np.random.uniform(low=0.0, high=np.pi, size=nconf)
    p = np.random.uniform(low=0.0, high=2 * np.pi, size=nconf)

    # rotated unit vectors:
    i_rot, j_rot, k_rot = (
        np.zeros([nconf, 3]),
        np.zeros([nconf, 3]),
        np.zeros([nconf, 3]),
    )
    i_rot[:, 0] = np.cos(p - np.pi / 2.0)
    i_rot[:, 1] = np.sin(p - np.pi / 2.0)
    j_rot[:, 0] = np.sin(t + np.pi / 2.0) * np.cos(p)
    j_rot[:, 1] = np.sin(t + np.pi / 2.0) * np.sin(p)
    j_rot[:, 2] = np.cos(t + np.pi / 2.0)
    k_rot[:, 0] = np.sin(t) * np.cos(p)
    k_rot[:, 1] = np.sin(t) * np.sin(p)
    k_rot[:, 2] = np.cos(t)

    d1, d2 = np.zeros(naip), np.zeros(naip)
    if naip == 6:
        d1[1] = np.pi

        d1[2] = np.pi / 2.0

        d1[3] = np.pi / 2.0
        d2[3] = np.pi

        d1[4] = np.pi / 2.0
        d2[4] = np.pi / 2.0

        d1[5] = np.pi / 2.0
        d2[5] = 3.0 * np.pi / 2.0

    elif naip == 12:
        d1[1] = np.pi

        fi0 = np.pi / 5.0
        tha = np.arccos(1.0 / np.sqrt(5.0))
        for i in range(5):
            rk2 = 2 * i
            d1[i + 2] = tha
            d2[i + 2] = rk2 * fi0

            d1[i + 7] = np.pi - tha
            d2[i + 7] = (rk2 + 1) * fi0

    epos_rot = np.zeros((nconf, naip, 3))
    for aip in range(naip):
        for d in range(3):
            epos_rot[:, aip, d] = apos[:, d] + r_ea * (
                np.sin(d1[aip]) * np.cos(d2[aip]) * i_rot[:, d]
                + np.sin(d1[aip]) * np.sin(d2[aip]) * j_rot[:, d]
                + np.cos(d1[aip]) * k_rot[:, d]
            )
    weights = 1.0 / naip * np.ones(naip)

    return weights, epos_rot
