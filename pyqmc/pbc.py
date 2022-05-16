import numpy as np


def enforce_pbc(lattvecs, epos):
    """Enforces periodic boundary conditions on a set of configs.
    Args:

      lattvecs: orthogonal lattice vectors defining 3D torus: (3,3)

      init_epos: attempted new electron coordinates: (nconfig,3)

    Returns:

      final_epos: final electron coordinates with PBCs imposed: (nconfig,3)

      wraparound: vector used to bring a given electron back to the simulation cell written in terms of lattvecs: (nconfig,3)
    """
    # Writes epos in terms of (lattice vecs) fractional coordinates
    recpvecs = np.linalg.inv(lattvecs)
    epos_lvecs_coord = np.einsum("...ij,jk->...ik", epos, recpvecs)
    # to_wrap = np.any((epos_lvecs_coord < 0) | (epos_lvecs_coord > 1), axis=-1)
    # print(to_wrap.shape)
    # wrapped = np.divmod(epos_lvecs_coord[to_wrap, :], 1)

    # wraparound = np.zeros(epos.shape)
    # wraparound[to_wrap, :] = wrapped[0]

    # final_epos = epos.copy()
    # final_epos[to_wrap, :] = np.einsum("...ij,jk->...ik", wrapped[1], lattvecs)
    # Finds position inside box and wraparound vectors (in lattice vector coordinates)
    tmp = np.divmod(epos_lvecs_coord, 1)
    wraparound = tmp[0]
    final_epos = np.dot(tmp[1], lattvecs)

    return final_epos, wraparound
