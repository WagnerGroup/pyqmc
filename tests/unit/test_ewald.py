import numpy as np
import pyqmc
from pyqmc.coord import PeriodicConfigs
from pyqmc.slaterpbc import get_supercell
from pyscf.pbc import gto, scf


def get_ewald_energy(cell, S, configs):
    supercell = get_supercell(cell, S)
    ewald = pyqmc.ewald.Ewald(supercell)
    configs = PeriodicConfigs(configs, supercell.lattice_vectors())
    ee, ei, ii = ewald.energy(configs)
    etot = ee + ei + ii
    print(dict(ee=ee, ei=ei, ii=ii))
    print("total energy", etot)
    return etot


def test_ewald_NaCl():
    print("NaCl ewald energy")
    nacl_answer = 1.74756
    L = 2.0  # this normalizes n.n. separation to 1 for rock salt structure
    cell = gto.Cell(
        atom="""H     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        unit="bohr",
        spin=1,
    )
    cell.spin = 1
    cell.build(a=(np.ones((3, 3)) - np.eye(3)) * L / 2, spin=1)
    cell.spin = 1

    # Primitive cell
    print("primitive cell")
    S = np.eye(3)
    configs = np.ones((1, 1, 3)) * L / 2
    etot = get_ewald_energy(cell, S, configs)
    print("correct answer: ", nacl_answer)
    assert np.abs(etot + nacl_answer) < 1e-4

    # Conventional cell
    print("conventional cell")
    S = np.ones((3, 3)) - 2 * np.eye(3)
    configs = np.ones((1, 4, 3)) * L / 2
    configs[:, 1:, :] = np.eye(3) * L / 2
    etot = get_ewald_energy(cell, S, configs)
    print("correct answer: ", 4 * nacl_answer)
    assert np.abs(etot / 4 + nacl_answer) < 1e-4


def test_ewald_CaF2():
    print("CaF2 ewald energy")
    caf2_answer = 5.03879
    L = 4 / np.sqrt(3)  # this normalizes n.n. separation to 1 for fluorite structure
    cell = gto.Cell(
        atom="""He     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        unit="bohr",
        spin=0,
    )
    cell.build(a=(np.ones((3, 3)) - np.eye(3)) * L / 2)

    # Primitive cell
    print("primitive cell")
    S = np.eye(3)
    configs = np.ones((1, 2, 3)) * L / 4
    configs[0, 1, 1] *= -1
    etot = get_ewald_energy(cell, S, configs)
    print("correct answer: ", caf2_answer)
    assert np.abs(etot + caf2_answer) < 1e-4

    # Conventional cell
    print("conventional cell")
    S = np.ones((3, 3)) - 2 * np.eye(3)
    cube = np.stack(np.meshgrid(*[[0, 1]] * 3, indexing="ij"), axis=-1).reshape((-1, 3))
    configs = np.reshape((cube + 0.5) * L / 2, (1, 8, 3))
    etot = get_ewald_energy(cell, S, configs)
    print("correct answer: ", 4 * caf2_answer)
    assert np.abs(etot / 4 + caf2_answer) < 1e-4


r"""
https://en.wikipedia.org/wiki/Madelung_constant
https://aip.scitation.org/doi/pdf/10.1063/1.1731810
https://chem.libretexts.org/Bookshelves/Inorganic_Chemistry/Map%3A_Inorganic_Chemistry_(Housecroft)/06%3A_Structures_and_energetics_of_metallic_and_ionic_solids/6.13%3A_Lattice_Energy_-_Estimates_from_an_Electrostatic_Model/6.13E%3A_Madelung_Constants

For a lattice of ions of charge $z_i$, the Madelung constant M_i is defined for an ion at site i of the lattice by

$V_i = \frac{1}{r_0} M_i$,

where $V_i$ is the potential felt by the ion at site i, due to all other ions in the lattice.

$M_i = \sum_j \frac{z_j}{r_{ij} / r_0}$.

So the total electrostatic energy will be the sum of the energies $z_i M_i$, divided by two for double counting.

The test above is for the NaCl lattice, (H for "plus", and electrons at "minus" positions, nearest neighbor distance chosen to be $r_0=1$). ewald.energy returns the energy of the unit cell (not per ion). For NaCl, 

$M_i = \pm 1.74756$,

so the total energy of a unit cell of four formula units would be

$E = [-M_i (1 + 1 + 1 + 1) + M_i (-1 - 1 - 1 - 1)] / 2 = -6.99 {\rm Ha}$.

Output:
ewald energy
{'ee': array([-4.58486208]), 'ei': array([2.17946577]), 'ii': -4.5848620786449485}
total energy [-6.99025839]

For CaF2, Madelung constant (per F) is -2.51939, (per Ca is -5.03879)so the energy of the unit cell should be -20.16
"""

if __name__ == "__main__":
    test_ewald_NaCl()
    test_ewald_CaF2()
