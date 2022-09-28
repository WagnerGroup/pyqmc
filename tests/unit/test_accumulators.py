import numpy as np
from pyqmc.accumulators import EnergyAccumulator, LinearTransform, SqAccumulator
from pyqmc.obdm import OBDMAccumulator
from pyqmc.tbdm import TBDMAccumulator
import pyqmc.api as pyq
import copy


def test_transform(LiH_sto3g_rhf):
    """Tests that the shapes are ok"""
    mol, mf = LiH_sto3g_rhf
    wf, to_opt = pyq.generate_wf(mol, mf)
    transform = LinearTransform(wf.parameters)
    x = transform.serialize_parameters(wf.parameters)
    nconfig = 10
    configs = pyq.initial_guess(mol, nconfig)
    wf.recompute(configs)
    pgrad = wf.pgradient()
    gradtrans = transform.serialize_gradients(pgrad)
    assert gradtrans.shape[1] == len(x)
    assert gradtrans.shape[0] == nconfig


def test_info_functions_mol(LiH_sto3g_rhf):
    mol, mf = LiH_sto3g_rhf
    wf, to_opt = pyq.generate_wf(mol, mf)
    accumulators = {
        "pgrad": pyq.gradient_generator(mol, wf, to_opt),
        "obdm": OBDMAccumulator(mol, orb_coeff=mf.mo_coeff),
        "tbdm_updown": TBDMAccumulator(mol, np.asarray([mf.mo_coeff] * 2), (0, 1)),
    }
    info_functions(mol, wf, accumulators)


def test_info_functions_pbc(H_pbc_sto3g_krks):
    from pyqmc.supercell import get_supercell

    mol, mf = H_pbc_sto3g_krks
    kinds = [0, 1]
    dm_orbs = [mf.mo_coeff[i][:, :2] for i in kinds]
    wf, to_opt = pyq.generate_wf(mol, mf)
    accumulators = {
        "pgrad": pyq.gradient_generator(mol, wf, to_opt, ewald_gmax=10),
        "obdm": OBDMAccumulator(mol, dm_orbs, kpts=mf.kpts[kinds]),
        "Sq": SqAccumulator(mol.lattice_vectors()),
    }
    info_functions(mol, wf, accumulators)


def info_functions(mol, wf, accumulators):
    accumulators["energy"] = accumulators["pgrad"].enacc
    configs = pyq.initial_guess(mol, 100)
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
