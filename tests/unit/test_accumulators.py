import numpy as np
from pyqmc.energy import energy
from pyqmc.accumulators import LinearTransform


def test_transform():
    """ Just prints things out; 
    TODO: figure out a thing to test.
    """
    from pyscf import gto, scf
    import pyqmc

    r = 1.54 / 0.529177
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. %g" % r,
        ecp="bfd",
        basis="bfd_vtz",
        unit="bohr",
        verbose=1,
    )
    mf = scf.RHF(mol).run()
    wf = pyqmc.slater_jastrow(mol, mf)
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


if __name__ == "__main__":
    test_transform()
