import pyqmc.api as pyq
# import pyqmc.three_body_jastrow_backup as three_body_jastrow
import three_body_jastrow as three_body_jastrow
import pyqmc.wftools as wftools
import pyqmc.mc as mc
import pyqmc.testwf as testwf
import pyqmc.coord as coord
import numpy as np
import pyscf


def run_tests(wf, epos, epsilon):

    #_, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

    for k, item in testwf.test_updateinternals(wf, epos).items():
        print(k, item)
        assert item < epsilon


    print(epos.configs.shape,'epos shape in runtests temp test')

    testwf.test_mask(wf, 0, epos.make_irreducible(0,epos.configs[:,0]))

    for fname, func in zip(
        ["gradient", "laplacian", "pgradient"],
        [testwf.test_wf_gradient, testwf.test_wf_laplacian, testwf.test_wf_pgradient],
    ):
        err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
        print(err)
        assert min(err) < epsilon, "epsilon {0}".format(epsilon)
        print('assertion done')

    for fname, func in zip(
        ["gradient_value", "gradient_laplacian"],
        [testwf.test_wf_gradient_value, testwf.test_wf_gradient_laplacian],
    ):
        d = func(wf, epos)
        for k, v in d.items():
            assert v < 1e-10, (k, v)


def test_obc_wfs(mol, J, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    mol = mol
    wf = J
    for k in wf.parameters:
        if k != "mo_coeff":
            wf.parameters[k] = np.asarray(np.random.rand(*wf.parameters[k].shape))

    epos = pyq.initial_guess(mol, nconf)
    print(epos.configs.shape,'epos shape in initial guess')
    run_tests(wf, epos, epsilon)


if __name__ == "__main__":
    mol = pyscf.gto.M(atom="H 0. 0. 0.,; Li 0. 0. 1.5", basis="sto-3g", unit="bohr")
    a_basis, b_basis = wftools.default_jastrow_basis(mol)
    J = three_body_jastrow.Three_Body_JastrowSpin(mol, a_basis, b_basis)
    J.parameters["ccoeff"] = np.random.random(J.parameters["ccoeff"].shape) * 0.02 - 0.01
    test_obc_wfs(mol, J, epsilon=1e-5, nconf=10)
