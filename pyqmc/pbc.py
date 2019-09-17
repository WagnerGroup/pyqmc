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
    epos_lvecs_coord = np.dot(epos, recpvecs)

    # Finds position inside box and wraparound vectors (in lattice vector coordinates)
    tmp = np.divmod(epos_lvecs_coord, 1)
    wraparound = tmp[0]
    final_epos = np.dot(tmp[1], lattvecs)

    return final_epos, wraparound
