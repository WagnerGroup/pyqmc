import numpy as np
import pyqmc
from pyqmc.coord import PeriodicConfigs
from pyqmc.slaterpbc import get_pyscf_supercell
from pyscf.pbc import gto, scf


def test_ewald_NaCl():
    L = 2.0  # this normalizes n.n. separation to 1 for rock salt structure
    mol = gto.Cell(
        atom="""H     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        unit="bohr",
        charge=0,
        spin=1,
    )
    mol.build(a=(np.ones((3, 3)) - np.eye(3)) * L / 2)

    # supercell = np.ones((3,3)) - 2 * np.eye(3)
    S = np.eye(3)
    supercell = get_pyscf_supercell(mol, S)
    ewald = pyqmc.ewald.Ewald(supercell)

    # configs = np.ones((1, 4, 3))
    # configs[:, 1:, :] = np.eye(3)
    configs = np.ones((1, 1, 3)) * L / 2
    configs = PeriodicConfigs(configs, supercell.lattice_vectors())

    print("NaCl ewald energy")
    ee, ei, ii = ewald.energy(configs)
    print(dict(ee=ee, ei=ei, ii=ii))
    print("total energy", ee + ei + ii)


def test_ewald_CaF2():
    L = 4 / np.sqrt(3)  # this normalizes n.n. separation to 1 for fluorite structure
    mol = gto.Cell(
        atom="""He     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        unit="bohr",
        charge=0,
        spin=0,
    )
    mol.build(a=(np.ones((3, 3)) - np.eye(3)) * L / 2)

    # supercell = np.ones((3,3)) - 2 * np.eye(3)
    S = np.eye(3)
    supercell = get_pyscf_supercell(mol, S)
    ewald = pyqmc.ewald.Ewald(supercell)

    # cube = np.stack(np.meshgrid(*[[0, 1]]*3, indexing="ij"), axis=-1).reshape((-1, 3))
    # configs = np.reshape((cube + 0.5) * L / 2, (1, 8, 3))
    configs = np.ones((1, 2, 3)) * L / 4
    configs[0, 1, 1] *= -1
    configs = PeriodicConfigs(configs, supercell.lattice_vectors())

    print("CaF2 ewald energy")
    ee, ei, ii = ewald.energy(configs)
    print(dict(ee=ee, ei=ei, ii=ii))
    print("total energy", ee + ei + ii)


r"""
https://en.wikipedia.org/wiki/Madelung_constant

For a lattice of ions of charge $z_i$, the Madelung constant M_i is defined for an ion at site i of the lattice by

$V_i = \frac{1}{r_0} M_i$,

where $V_i$ is the potential felt by the ion at site i, due to all other ions in the lattice.

$M_i = \sum_j \frac{z_j}{r_{ij} / r_0}$.

So the total electrostatic energy will be the sum of the energies $z_i M_i$, divided by two for double counting.

The test above is for the NaCl lattice, (H for "plus", and electrons at "minus" positions, nearest neighbor distance chosen to be $r_0=1$). ewald.energy returns the energy of the unit cell (not per ion). For NaCl, 

$M_i = \pm 1.748$,

so the total energy of a unit cell of four formula units would be

$E = [-M_i (1 + 1 + 1 + 1) + M_i (-1 - 1 - 1 - 1)] / 2 = -6.99 {\rm Ha}$.

Output:
ewald energy
{'ee': array([-4.58486208]), 'ei': array([2.17946577]), 'ii': -4.5848620786449485}
total energy [-6.99025839]

For CaF2, Madelung constant (per F) is -2.52, so the energy of the unit cell should be -20.16
"""

if __name__ == "__main__":
    test_ewald_NaCl()
    test_ewald_CaF2()
