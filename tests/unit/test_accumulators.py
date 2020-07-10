import numpy as np
from pyqmc.energy import energy
from pyqmc.accumulators import LinearTransform
from pyqmc.obdm import OBDMAccumulator
import pyqmc


def test_transform():
    """ Just prints things out; 
    TODO: figure out a thing to test.
    """
    from pyscf import gto, scf

    r = 1.54 / 0.529177
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. %g" % r,
        ecp="bfd",
        basis="bfd_vtz",
        unit="bohr",
        verbose=1,
    )
    mf = scf.RHF(mol).run()
    wf, to_opt = pyqmc.default_sj(mol, mf)
    enacc = pyqmc.EnergyAccumulator(mol)
    print(list(wf.parameters.keys()))
    transform = LinearTransform(wf.parameters)
    x = transform.serialize_parameters(wf.parameters)

    nconfig = 10
    configs = pyqmc.initial_guess(mol, nconfig)
    wf.recompute(configs)
    pgrad = wf.pgradient()
    gradtrans = transform.serialize_gradients(pgrad)
    assert gradtrans.shape[1] == len(x)
    assert gradtrans.shape[0] == nconfig


def test_info_functions_mol():
    from pyscf import gto, scf
    from pyqmc.tbdm import TBDMAccumulator

    mol = gto.Mole()
    mol.atom = """He 0.00 0.00 0.00 """
    mol.basis = "ccpvdz"
    mol.build()

    mf = scf.RHF(mol)
    ehf = mf.kernel()

    wf, to_opt = pyqmc.default_sj(mol, mf)
    accumulators = {
        "pgrad": pyqmc.gradient_generator(mol, wf, to_opt),
        "obdm": OBDMAccumulator(mol, orb_coeff=mf.mo_coeff),
        "tbdm_updown": TBDMAccumulator(mol, np.asarray([mf.mo_coeff] * 2), (0, 1)),
    }
    info_functions(mol, wf, accumulators)


def test_info_functions_pbc():
    from pyscf.pbc import gto, scf
    from pyqmc.supercell import get_supercell

    mol = gto.Cell(atom="He 0.00 0.00 0.00", basis="ccpvdz", unit="B")
    mol.a = 5.61 * np.eye(3)
    mol.build()

    mf = scf.KRHF(mol, kpts=mol.make_kpts([2, 2, 2])).density_fit()
    ehf = mf.kernel()

    supercell = get_supercell(mol, 2 * np.eye(3))
    kinds = [0, 1]
    dm_orbs = [mf.mo_coeff[i][:, :2] for i in kinds]
    wf, to_opt = pyqmc.default_sj(mol, mf)
    accumulators = {
        "pgrad": pyqmc.gradient_generator(mol, wf, to_opt, ewald_gmax=10),
        "obdm": OBDMAccumulator(mol, dm_orbs, kpts=mf.kpts[kinds]),
        "Sq": pyqmc.accumulators.SqAccumulator(mol.lattice_vectors()),
    }
    info_functions(mol, wf, accumulators)


def info_functions(mol, wf, accumulators):
    accumulators["energy"] = accumulators["pgrad"].enacc
    configs = pyqmc.initial_guess(mol, 100)
    wf.recompute(configs)
    for k, acc in accumulators.items():
        shapes = acc.shapes()
        keys = acc.keys()
        assert shapes.keys() == keys, "keys: {0}\nshapes: {1}".format(keys, shapes)
        avg = acc.avg(configs, wf)
        assert avg.keys() == keys, (k, avg.keys(), keys)
        for ka in keys:
            assert shapes[ka] == avg[ka].shape, "{0} {1}".format(ka, avg[ka].shape)


if __name__ == "__main__":
    test_info_functions_mol()
    test_info_functions_pbc()
