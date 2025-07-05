import numpy as np
import pyscf
import pyqmc
import pyqmc.api as pyq
from pyqmc.observables.jax_ecp import ECPAccumulator
import pyqmc.observables.eval_ecp as eval_ecp


def evaluate_local_energy_variance(mol, wf, configs, use_old_ecp):
    enacc = pyq.EnergyAccumulator(mol, use_old_ecp=use_old_ecp , threshold = -1)
    ens = enacc(configs, wf)
    return ens['ecp']

def test_accelerated_ecp():
    # Test that the accelerated ECP implementation gives the same results as the old one
    # for a simple molecule (H2) with a small basis set.
    mol = pyscf.gto.M(
        atom="H 0. 0. 0.; H 0. 0. 1.4", basis="ccECP_cc-pVTZ", ecp="ccecp", unit="bohr"
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10)
    num_random_walks = 2
    for i in range(num_random_walks):
        data, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)
        energies = {}
        for use_old_ecp in [False, True]:
            energies[use_old_ecp] = evaluate_local_energy_variance(
                mol, wf, configs, use_old_ecp
            )
        assert(np.allclose(energies[True],  energies[False]))




def test_selected(diamond_primitive):
    """
    Test accelerated ECP with determistic evaluation.
    """
    mol, mf = diamond_primitive
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10)
    data, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)


    deterministic = ECPAccumulator(mol, stochastic_rotation = False, nselect_deterministic = 1000)
    stochastic = ECPAccumulator(mol, stochastic_rotation = False)

    ref = deterministic(configs, wf)

    nsample = 100
    samples = np.zeros((nsample, ref.shape[0]))
    for i in range(nsample):
        samples[i,:] = stochastic(configs, wf)


    avg = np.mean(samples, axis=0)
    var = np.var(samples, axis=0, ddof=1)
    stderr = np.sqrt(var/nsample)
    chi2_stochastic = np.abs(avg-ref)/stderr
    # for this system, errors should be small enough that chi2 is dominated by numerical noise..
    assert(np.mean(np.abs(avg-ref)) < 0.001)



def test_accelerated_PBC(diamond_primitive):
    """
    Make sure accelerated ECP gets the same result as traditional ECP.
    """
    mol, mf = diamond_primitive
    wf, _ = pyq.generate_slater(mol, mf)
    nconfig = 10
    configs = pyq.initial_guess(mol, nconfig)
    data, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)

    stochastic = ECPAccumulator(mol)

    nsample = 100
    samples = np.zeros((nsample, nconfig))
    samples_trad = np.zeros_like(samples)
    for i in range(nsample):
        samples[i,:] = stochastic(configs, wf)
        samples_trad[i,:] = eval_ecp.ecp(mol, configs, wf, threshold=-1).real


    avg = np.mean(samples, axis=0)
    var = np.var(samples, axis=0, ddof=1)
    stderr = np.sqrt(var/nsample)

    avg_trad = np.mean(samples_trad, axis=0)
    var_trad = np.var(samples_trad, axis=0, ddof=1)
    err_trad = np.sqrt(var_trad/nsample)

    stderr = np.sqrt(stderr**2 + err_trad**2)
    chi2_stochastic = np.abs(avg-avg_trad)/stderr
    print(chi2_stochastic)

    assert np.mean(chi2_stochastic) < 4.0, "JAX ECP deviates too much from traditional ECP"

