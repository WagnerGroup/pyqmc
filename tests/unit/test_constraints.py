import numpy as np
import pyqmc.api as pyq
from pyqmc.accumulators import LinearTransform
import copy

def test_constraints(H2_ccecp_casci_s0):
    mol, mf, mc = H2_ccecp_casci_s0

    wf, to_opt = pyq.generate_wf(mol, mf, mc=mc)

    old_parms = copy.deepcopy(wf.parameters)
    lt = LinearTransform(wf.parameters, to_opt)

    # Test serialize parameters
    x0 = lt.serialize_parameters(wf.parameters)
    x0 += np.random.normal(size=x0.shape)
    for k,it in lt.deserialize(x0).items():
        assert wf.parameters[k].shape==it.shape
        wf.parameters[k] = it

    # to_opt is supposed to be false for both of these.
    assert wf.parameters["wf1det_coeff"][0] == old_parms["wf1det_coeff"][0]
    assert np.sum(wf.parameters["wf2bcoeff"][0] - old_parms["wf2bcoeff"][0]) == 0
    #While this one is supposed to change.
    assert np.sum(wf.parameters["wf2bcoeff"][1] - old_parms["wf2bcoeff"][1]) != 0

    # Test serialize gradients
    configs = pyq.initial_guess(mol, 10)
    wf.recompute(configs)
    pgrad = wf.pgradient()
    pgrad_serial = lt.serialize_gradients(pgrad)

    #Pgrad should be walkers, configs
    assert pgrad_serial.shape[1] == x0.shape[0]
