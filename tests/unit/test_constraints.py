# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import pandas as pd
import numpy as np
from pyscf import lib, gto, scf, mcscf
from pyqmc import default_msj
from pyqmc.accumulators import LinearTransform
from pyqmc.coord import OpenConfigs


def test():
    # Default multi-Slater wave function
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas=2, nelecas=(1, 1))
    mc.kernel()

    wf, to_opt, freeze = default_msj(mol, mf, mc)
    old_parms = wf.parameters
    lt = LinearTransform(wf.parameters, to_opt, freeze)

    # Test serialize parameters
    x0 = lt.serialize_parameters(wf.parameters)
    x0 += np.random.normal(size=x0.shape)
    wf.parameters = lt.deserialize(x0)
    assert wf.parameters["wf1det_coeff"][0] == old_parms["wf1det_coeff"][0]
    assert np.sum(wf.parameters["wf2bcoeff"][0] - old_parms["wf2bcoeff"][0]) == 0

    # Test serialize gradients
    configs = OpenConfigs(np.random.randn(10, 4, 3))
    wf.recompute(configs)
    pgrad = wf.pgradient()
    pgrad_serial = lt.serialize_gradients(pgrad)
    assert np.sum(pgrad_serial[:, :3] - pgrad["wf1det_coeff"][:, 1:4]) == 0


if __name__ == "__main__":
    test()
