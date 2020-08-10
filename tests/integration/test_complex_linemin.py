import numpy as np
import os
import pyqmc.recipes


def run_scf(chkfile):
    from pyscf.pbc import gto, scf
    from pyscf.scf.addons import remove_linear_dep_

    mol = gto.Cell(
        atom="H 0 0 0; H 2 2 2",
        a=np.eye(3) * 5,
        basis="ccecpccpvdz",
        ecp="ccecp",
        unit="bohr",
    )
    mol.exp_to_discard = 0.1
    mol.build()
    mf = scf.KUKS(mol, kpts=mol.make_kpts([3, 3, 3]))
    mf.chkfile = chkfile
    mf = remove_linear_dep_(mf)

    dm = np.asarray(mf.get_init_guess())
    n = int(dm.shape[-1] / 2)
    dm[0, :n, :n] = 0
    dm[1, n:, n:] = 0
    energy = mf.kernel(dm)


def test():
    chkfile = "hepbc.hdf5"
    optfile = "linemin.hdf5"
    run_scf(chkfile)
    pyqmc.recipes.OPTIMIZE(
        chkfile,
        optfile,
        nconfig=50,
        S=np.diag([1, 1, 3]),
        slater_kws={"optimize_orbitals": True},
        linemin_kws={"max_iterations": 5},
    )
    assert os.path.isfile(optfile)
    os.remove(chkfile)
    os.remove(optfile)


if __name__ == "__main__":
    test()
