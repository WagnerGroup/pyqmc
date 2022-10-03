import pyqmc.api as pyq
import pyqmc.three_body_jastrow
import pyqmc.testwf as testwf
import pyqmc.coord as coord
import numpy as np
import pyscf


mol = pyscf.gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="sto-3g", unit="bohr")
a_basis, b_basis = pyqmc.wftools.default_jastrow_basis(mol)
J=pyqmc.three_body_jastrow.Three_Body_JastrowSpin(mol,a_basis,b_basis)
J.parameters["ccoeff"] = np.random.random(J.parameters["ccoeff"].shape) * 0.02 - 0.01
configs = pyq.initial_guess(mol, 10)
epos = coord.OpenConfigs(np.random.random((configs.configs.shape[0],configs.configs.shape[2])))
print(J.recompute(configs))
print(J.testvalue(2,epos)[0])

print(testwf.test_updateinternals(J,configs))



def run_tests(wf, epos, epsilon):

   # _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

    for k, item in testwf.test_updateinternals(wf, epos).items():
        print(k, item)
        assert item < epsilon

    # testwf.test_mask(wf, 0, epos)

    # for fname, func in zip(
    #     ["gradient", "laplacian", "pgradient"],
    #     [testwf.test_wf_gradient, testwf.test_wf_laplacian, testwf.test_wf_pgradient,],
    # ):
    #     err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
    #     assert min(err) < epsilon, "epsilon {0}".format(epsilon)

    # for fname, func in zip(
    #     ["gradient_value", "gradient_laplacian"],
    #     [testwf.test_wf_gradient_value, testwf.test_wf_gradient_laplacian,],
    # ):
    #     d = func(wf, epos)
    #     for k, v in d.items():
    #         assert v < 1e-10, (k, v)


def test_obc_wfs(mol,J, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    mol=mol
    wf=J
    for k in wf.parameters:
        if k != "mo_coeff":
            wf.parameters[k] = np.asarray(np.random.rand(*wf.parameters[k].shape))

    epos = pyq.initial_guess(mol, nconf)
    run_tests(wf, epos, epsilon)

#test_obc_wfs(mol,J,epsilon=1e-5, nconf=10)