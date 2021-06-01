import numpy as np
import pyqmc


def get_supercell_kpts(supercell):
    Sinv = np.linalg.inv(supercell.S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, supercell.S.T)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    kptmesh = np.meshgrid(*[np.arange(*r) for r in xyz_range], indexing="ij")
    possible_kpts = np.dot(np.stack([x.ravel() for x in kptmesh]).T, Sinv)
    in_unit_box = (possible_kpts >= 0) * (possible_kpts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    reclatvec = np.linalg.inv(supercell.original_cell.lattice_vectors()).T * 2 * np.pi
    return np.dot(possible_kpts[select], reclatvec)


def get_supercell_copies(latvec, S):
    Sinv = np.linalg.inv(S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, S)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    mesh = np.meshgrid(*[np.arange(*r) for r in xyz_range], indexing="ij")
    possible_pts = np.dot(np.stack([x.ravel() for x in mesh]).T, Sinv.T)
    in_unit_box = (possible_pts >= 0) * (possible_pts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    return np.linalg.multi_dot((possible_pts[select], S, latvec))


def get_supercell(cell, S):
    """
    Inputs:
        cell: pyscf Cell object
        S: (3, 3) supercell matrix for QMC from cell defined by cell.a. In other words, the QMC calculation cell is qmc_cell = np.dot(S, cell.lattice_vectors()). For a 2x2x2 supercell, S is [[2, 0, 0], [0, 2, 0], [0, 0, 2]].
    """
    import pyscf.pbc

    scale = np.abs(int(np.round(np.linalg.det(S))))
    superlattice = np.dot(S, cell.lattice_vectors())
    Rpts = get_supercell_copies(cell.lattice_vectors(), S)
    atom = []
    for (name, xyz) in cell._atom:
        atom.extend([(name, xyz + R) for R in Rpts])
    supercell = pyscf.pbc.gto.Cell()
    supercell.a = superlattice
    supercell.atom = atom
    supercell.ecp = cell.ecp
    supercell.basis = cell.basis
    supercell.exp_to_discard = cell.exp_to_discard
    supercell.unit = "Bohr"
    supercell.spin = cell.spin * scale
    supercell.build()
    supercell.original_cell = cell
    supercell.S = S
    supercell.scale = scale
    supercell.output = None
    supercell.stdout = None
    return supercell


def make_supercell_jastrow(jastrow, S):
    from pyqmc.jastrowspin import JastrowSpin

    scale = int(np.round(np.linalg.det(S)))
    supercell = get_supercell(jastrow._mol, S)
    newjast = JastrowSpin(supercell, jastrow.a_basis, jastrow.b_basis)
    newjast.parameters["bcoeff"] = jastrow.parameters["bcoeff"]
    newjast.parameters["acoeff"] = np.repeat(
        jastrow.parameters["acoeff"], scale, axis=0
    )
    return newjast
