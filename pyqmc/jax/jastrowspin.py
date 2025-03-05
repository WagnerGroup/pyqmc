import jax
import jax.numpy as jnp
from functools import partial
import pyqmc.wftools
from typing import NamedTuple


class BasisParameters(NamedTuple):
    """
    These are parameters for the basis functions (polypade and cusp).
    """
    beta_a: float
    beta_b: float
    ion_cusp: list
    rcut: float
    gamma: float


class CoefficientParameters(NamedTuple):
    """
    These are the Jastrow coefficients.
    """
    acoeff: jax.Array
    bcoeff: jax.Array


def polypade(r, beta, rcut):
    x = r / rcut
    z = ((3*x - 8) * x + 6) * x**2
    p = (1 - z) / (1 + beta * z)
    return jnp.where(r > rcut, 0.0, p)


# vectorize polypade function
_inner_polypade = jax.vmap(polypade, in_axes=(0, None, None), out_axes=0) # [(n,), float, float] -> (n,)
vmapped_polypade = jax.vmap(_inner_polypade, in_axes=(None, 0, None), out_axes=1) # [(n,), (nbasis,), float] -> (n, nbasis)


def cutoffcusp(r, gamma, rcut):
    y = r / rcut
    y1 = y - 1
    p = (y1 * y1 * y1 + 1) / 3
    func = -p / (1 + gamma * p) + 1 / (3 + gamma)
    return jnp.where(r > rcut, 0.0, func * rcut)


def compute_basis_sum(elec_coord, coords, coeff, basis, param, rcut):
    """
    For a given electron, compute the sum over basis functions and atom coordinates (one-body) or electron coordinates (two-body).
    To be vmapped over configurations and/or electrons for use in evaluators.
    In the following, n = natom or nelec_up or nelec_dn.
    
    Args:
        elec_coord (jax.Array): Electron coordinates. (3,)
        coords (jax.Array): Array of atom or electron coordinates. (n, 3)
        coeff (jax.Array): Jastrow coefficients. (n, nbasis) for polypade, (n,) for cutoffcusp
        basis (Callable[(n,), (nbasis,) or float, float -> (n, nbasis) or (n,)]): Vmapped polypade or cutoffcusp.
        param (jax.Array or float): The beta parameters (nbasis,) or the cusp gamma parameter (float).
        rcut (float): Cutoff radius.
            
    Return:
        float: Sum.
    """
    rij = jnp.linalg.norm(elec_coord - coords, axis=-1)
    basis_value = basis(rij, param, rcut)
    return jnp.sum(coeff * basis_value)


def compute_bdiag_corr(mol, basis_params, parameters):
    """
    Compute the diagonal sum of the b terms as a correction, a helper function for evaluate_jastrow().
    """
    nup, ndn = mol.nelec
    bcoeff = parameters.bcoeff
    rcut, gamma = basis_params.rcut, basis_params.gamma
    diag = jnp.sum(bcoeff[1:, 0]) * nup + jnp.sum(bcoeff[1:, 2]) * ndn
    diagc = rcut/(3+gamma) * (bcoeff[0, 0] * nup + bcoeff[0, 2] * ndn)
    return diag + diagc


def evaluate_jastrow(mol, basis_params, basis_sum_evaluators, parameters, configs):
    """
    Evaluate the log Jastrow factor.

    Args:
        mol (pyscf.gto.Mole): PySCF molecule object.
        basis_params (BasisParameters): Basis parameters.
        basis_sum_evaluators (list): List of vmapped compute_basis_sum() functions for evaluating different terms.
        parameters (CoefficientParameters): Jastrow coeffcients.
        configs (jax.Array): Electron configurations. (nconfig, nelec, 3)
    
    Return:
        a (jax.Array): Sum of a terms. (nconfig,)
        b (jax.Array): Sum of b terms. (nconfig,)
        logj (jax.Array): Log Jastrow factor. (nconfig,)
    """
    nup, ndn = mol.nelec
    nconfig = configs.shape[0]
    configs_up, configs_dn = configs[:, :nup, :], configs[:, nup:, :]
    atom_coords = jnp.array(mol.atom_coords())
    # tiled to match the expected shape of coords in basis_sum_evaluators 
    tiled_atom_coords = jnp.tile(atom_coords, (nconfig, 1, 1))
    acoeff, bcoeff = parameters

    a_value, b_value, cusp_value = basis_sum_evaluators
    
    # compute a terms for up and down electrons
    a_up = a_value(configs_up, tiled_atom_coords, acoeff[:, 1:, 0]) \
         + cusp_value(configs_up, tiled_atom_coords, acoeff[:, 0, 0])
    a_dn = a_value(configs_dn, tiled_atom_coords, acoeff[:, 1:, 1]) \
         + cusp_value(configs_dn, tiled_atom_coords, acoeff[:, 0, 1])

    # compute b terms for up-up, up-down, down-down electron pairs
    # bcoeff is tiled to match the first dimension (n) of acoeff
    b_upup = b_value(configs_up, configs_up, jnp.tile(bcoeff[1:, 0], (nup, 1))) \
           + cusp_value(configs_up, configs_up, jnp.tile(bcoeff[0, 0], (nup)))
    b_updn = b_value(configs_up, configs_dn, jnp.tile(bcoeff[1:, 1], (ndn, 1))) \
           + cusp_value(configs_up, configs_dn, jnp.tile(bcoeff[0, 1], (ndn)))
    b_dndn = b_value(configs_dn, configs_dn, jnp.tile(bcoeff[1:, 2], (ndn, 1))) \
           + cusp_value(configs_dn, configs_dn, jnp.tile(bcoeff[0, 2], (ndn)))
    bdiag_corr = compute_bdiag_corr(mol, basis_params, parameters)

    # sum the terms up to obtain the log Jastrow factor
    a = jnp.sum(a_up + a_dn, axis=1)
    b = jnp.sum((b_upup + b_dndn)/2 + b_updn, axis=1)
    b -= bdiag_corr / 2
    logj = a + b
    return a, b, logj


def evaluate_testvalue(mol, basis_sum_evaluators, parameters, configs_old, e, spin, epos):
    """
    Evaluate the test value (:math:`\log J(\vec{R}')/\logJ(\vec{R})`) for a single-electron update.

    Args:
        mol (pyscf.gto.Mole): PySCF molecule object.
        basis_sum_evaluators (list): List of vmapped compute_basis_sum() functions for evaluating different terms.
        parameters (CoefficientParameters): Jastrow coeffcients.
        configs_old (jax.Array): Electron configurations before the update. (nconfig, nelec, 3)
        e (int): Index of the updated electron.
        spin (int): Spin of the updated electron.
        epos (jax.Array): New positions of the updated electron. (nconfig, 3)
    
    Return:
        jax.Array: Test value. (nconfig,)
    """
    nup, ndn = mol.nelec
    nconfig = configs_old.shape[0]
    epos_old = configs_old[:, e, :]
    configs_old_up, configs_old_dn = configs_old[:, :nup, :], configs_old[:, nup:, :]
    configs = configs_old.copy().at[:, e, :].set(epos)
    configs_up, configs_dn = configs[:, :nup, :], configs[:, nup:, :]
    atom_coords = jnp.array(mol.atom_coords())
    # tiled to match the expected shape of coords in basis_sum_evaluators 
    tiled_atom_coords = jnp.tile(atom_coords, (nconfig, 1, 1))
    acoeff, bcoeff = parameters

    a_value, b_value, cusp_value = basis_sum_evaluators

    # compute a term contributed by electron e
    a = a_value(epos, tiled_atom_coords, acoeff[:, 1:, spin]) \
      + cusp_value(epos, tiled_atom_coords, acoeff[:, 0, spin])
    a_old = a_value(epos_old, tiled_atom_coords, acoeff[:, 1:, spin])\
          + cusp_value(epos_old, tiled_atom_coords, acoeff[:, 0, spin])

    # compute b terms contributed by electron e
    # bcoeff is tiled to match the first dimension (n) of acoeff
    b_up = b_value(epos, configs_up, jnp.tile(bcoeff[1:, spin+0], (nup, 1))) \
         + cusp_value(epos, configs_up, jnp.tile(bcoeff[0, spin+0], (nup)))
    b_dn = b_value(epos, configs_dn, jnp.tile(bcoeff[1:, spin+1], (ndn, 1))) \
         + cusp_value(epos, configs_dn, jnp.tile(bcoeff[0, spin+1], (ndn)))
    b_up_old = b_value(epos_old, configs_old_up, jnp.tile(bcoeff[1:, spin+0], (nup, 1))) \
             + cusp_value(epos_old, configs_old_up, jnp.tile(bcoeff[0, spin+0], (nup)))
    b_dn_old = b_value(epos_old, configs_old_dn, jnp.tile(bcoeff[1:, spin+1], (ndn, 1))) \
             + cusp_value(epos_old, configs_old_dn, jnp.tile(bcoeff[0, spin+1], (ndn)))

    delta_a = a - a_old
    # factor of 1/2 cancelled by the double appearance (row and column) of b terms contributed by electron e
    delta_b = b_up + b_dn - b_up_old - b_dn_old 
    delta_logj =  delta_a + delta_b
    return jnp.exp(delta_logj)


def create_jastrow_evaluator(mol, basis_params):
    """
    Create a set of functions that can be used to evaluate the Jastrow factor.
    """
    beta_a, beta_b, ion_cusp, rcut, gamma = basis_params
    basis_sum_evaluators_elecconf = [
        jax.vmap( # vmapping over configurations
            jax.vmap( # vmapping over electrons
                partial(compute_basis_sum, basis=basis, param=param, rcut=rcut), 
                in_axes=(0, None, None)
            ), in_axes=(0, 0, None)
        )
        for basis, param in zip([vmapped_polypade, vmapped_polypade, cutoffcusp], [beta_a, beta_b, gamma])]

    basis_sum_evaluators_conf = [
        jax.vmap( # vmapping over configurations
            partial(compute_basis_sum, basis=basis, param=param, rcut=rcut), 
            in_axes=(0, 0, None)
        )
        for basis, param in zip([vmapped_polypade, vmapped_polypade, cutoffcusp], [beta_a, beta_b, gamma])]

    value_func = jax.jit(partial(evaluate_jastrow, mol, basis_params, basis_sum_evaluators_elecconf))
    testval_func = jax.jit(partial(evaluate_testvalue, mol, basis_sum_evaluators_conf))
    return value_func, testval_func


class _parameterMap:
    """
    This class wraps the parameters so that we only need to transfer them to 
    the GPU if they are changed. 
    """
    def __init__(self, jax_parameters):
        self.jax_parameters = jax_parameters
        self.parameters= {'acoeff': np.asarray(self.jax_parameters[0]), 
                          'bcoeff': np.asarray(self.jax_parameters[1])}

    def __setitem__(self, key, item):
        self.parameters[key] = item
        self.jax_parameters = CoefficientParameters(jnp.array(self.parameters['acoeff']),
                                                    jnp.array(self.parameters['bcoeff']))

    def __getitem__(self, key):
        return self.parameters[key]

    def __repr__(self):
        return repr(self.parameters)

    def __len__(self):
        return len(self.parameters)

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete parameters")

    def clear(self):
        raise NotImplementedError("Cannot clear parameters")

    def copy(self):
        return self.parameters.copy()

    def has_key(self, k):
        return k in self.parameters

    def update(self, *args, **kwargs):
        raise NotImplementedError("Cannot update parameters")

    def keys(self):
        return self.parameters.keys()

    def values(self):
        return self.parameters.values()


class JAXJastrowSpin:
    def __init__(self, mol, ion_cusp=None, na=4, nb=3, rcut=None, gamma=None, beta0_a=0.2, beta0_b=0.5):
        self._mol = mol
        self._init_params(ion_cusp, na, nb, rcut, gamma, beta0_a, beta0_b)
        self._recompute, self._testvalue = create_jastrow_evaluator(self._mol, self.basis_params)


    def _init_params(self, ion_cusp, na, nb, rcut, gamma, beta0_a, beta0_b):
        """
        Initialize self.basis_params (basis parameters) and self.parameters (Jastrow coefficients).
        This replaces wftools.default_jastrow_basis() and wftools.generate_jastrow() since
        we don't use func3d.PolyPadeFunction() and func3d.CutoffCuspFunction() objects anymore.
        """
        mol = self._mol
        if ion_cusp is False:
            ion_cusp = []
            if not mol.has_ecp():
                print("Warning: using neither ECP nor ion_cusp")
        elif ion_cusp is True:
            ion_cusp = list(mol._basis.keys())
            if mol.has_ecp():
                print("Warning: using both ECP and ion_cusp")
        elif ion_cusp is None:
            ion_cusp = [l for l in mol._basis.keys() if l not in mol._ecp.keys()]
        else:
            assert isinstance(ion_cusp, list)

        if gamma is None:
            gamma = 24
        if rcut is None:
            if hasattr(mol, "a"):
                rcut = jnp.amin(jnp.pi / jnp.linalg.norm(mol.reciprocal_vectors(), axis=1))
            else:
                rcut = 7.5

        beta_a = pyqmc.wftools.expand_beta_qwalk(beta0_a, na)
        beta_b = pyqmc.wftools.expand_beta_qwalk(beta0_b, nb)
        self.basis_params = BasisParameters(beta_a, beta_b, ion_cusp, rcut, gamma)

        acoeff = jnp.zeros((self._mol.natm, len(beta_a)+1, 2))
        bcoeff = jnp.zeros((len(beta_b)+1, 3))
        if len(ion_cusp) > 0:
            coefs = jnp.array(mol.atom_charges(), dtype=jnp.float64)
            mask = jnp.array([l[0] not in ion_cusp for l in mol._atom])
            coefs = coefs.at[mask].set(0.0)
            acoeff = acoeff.at[:, 0, :].set(coefs[:, None])
        bcoeff = bcoeff.at[0, [0, 1, 2]].set(jnp.array([-0.25, -0.50, -0.25]))

        self.parameters = _parameterMap(CoefficientParameters(acoeff, bcoeff))
        

    def recompute(self, configs):
        _configs = jnp.array(configs.configs)
        self._a, self._b, self._logj = self._recompute(self.parameters.jax_parameters, _configs)
        return self._logj


    def testvalue(self, configs_old, e, epos, mask=None):
        _configs_old = jnp.array(configs_old.configs)
        _epos = jnp.array(epos.configs)
        spin = int(e >= self._mol.nelec[0])
        return self._testvalue(self.parameters.jax_parameters, _configs_old, e, spin, _epos)


if __name__ == "__main__":
    cpu=True
    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update("jax_enable_x64", True)
    else:
        pass 

    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyqmc.testwf
    import pyscf.gto
    import numpy as np

    import pyqmc.api as pyq
    mol = {}
    mf = {}

    mol['h2o'] = pyscf.gto.Mole(atom = '''O 0 0 0; H  0 2.0 0; H 0 0 2.0''', basis = 'cc-pVDZ', cart=True)
    mol['h2o'].build()
    
    jax_jastrow = JAXJastrowSpin(mol['h2o'])
    jastrow, _ = pyqmc.wftools.generate_jastrow(mol['h2o'])

    data = []
    # for nconfig in [10, 1000, 100000]:
    for nconfig in [10]:
        configs = pyq.initial_guess(mol['h2o'], nconfig)

        jax_jastrowval = jax_jastrow.recompute(configs) 
        jax.block_until_ready(jax_jastrowval)

        jax_start = time.perf_counter()
        jax_jastrowval = jax_jastrow.recompute(configs) 
        jax.block_until_ready(jax_jastrowval)
        jax_end = time.perf_counter()

        slater_start = time.perf_counter()
        pyqmc_jastrowval = jastrow.recompute(configs)
        slater_end = time.perf_counter()

        print("jax", jax_end-jax_start, "slater", slater_end-slater_start)
        print('MAD', jnp.mean(jnp.abs(pyqmc_jastrowval[1]- jax_jastrowval)))
        print("jax values", jax_jastrowval)
        print("pyqmc values", pyqmc_jastrowval[1])

        new_configs = configs.copy()
        new_configs.electron(7).configs += np.random.normal(0, 0.1, configs.electron(7).configs.shape)

        jax_testval = jax_jastrow.testvalue(configs, 7, new_configs.electron(7))
        pyqmc_testval = jastrow.testvalue(7, new_configs.electron(7))[0]
        print("jax testval", jax_testval)
        print("pyqmc testval", pyqmc_testval)

        data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax'})
        data.append({'N': nconfig, 'time': slater_end-slater_start, 'method': 'pyqmc'})
    
    sns.lineplot(data = pd.DataFrame(data), x='N', y='time', hue='method')
    plt.ylim(0)
    # plt.savefig("jax_vs_pyqmc.png")