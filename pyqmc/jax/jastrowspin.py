import jax
import jax.numpy as jnp
from functools import partial
import pyqmc.wftools
from typing import NamedTuple


class BasisParameters(NamedTuple):
    beta_a: float
    beta_b: float
    ion_cusp: list
    rcut: float
    gamma: float


def polypade(r, beta, rcut):
    x = r / rcut
    z = ((3*x - 8) * x + 6) * x**2
    p = (1 - z) / (1 + beta * z)
    return jnp.where(r > rcut, 0.0, p)


def cutoffcusp(r, gamma, rcut):
    y = r / rcut
    y1 = y - 1
    p = (y1 * y1 * y1 + 1) / 3
    func = -p / (1 + gamma * p) + 1 / (3 + gamma)
    return jnp.where(r > rcut, 0.0, func * rcut)


def compute_basis_sum(elec_coord, coords, coeff, basis, param, rcut):
    """
    For a given electron, compute the sum over basis functions and atom coordinates (one-body) or electron coordinates (two-body)
    In the following, n = natom or nelec_up or nelec_dn
    
    Args:
        elec_coord (jax.Array): Electron coordinates. (3,)
        coords (jax.Array): Array of atom or electron coordinates. (n, 3)
        coeff (jax.Array): Jastrow coefficients. (n, nbasis) for polypade, (n,) for cutoffcusp
        basis (Callable[(n,), (nbasis,) or float, float -> (n, nbasis) or (n,)]): vmapped polypade or cutoffcusp.
        param (jax.Array or float): The beta parameters (nbasis,) or the cusp gamma parameter (float).
        rcut (float): Cutoff radius.
            
    Return:
        float: Sum.
    """
    rij = jnp.linalg.norm(elec_coord - coords, axis=-1)
    basis_value = basis(rij, param, rcut)
    return jnp.sum(coeff * basis_value)


def vmap_over_configs(func):
    # vmapping over electrons
    elecs_vmapped_func = jax.vmap(func, in_axes=(0, None, None))
    # vmapping over configurations
    configs_vmapped_func = jax.vmap(elecs_vmapped_func, in_axes=(0, 0, None))
    return configs_vmapped_func


def create_jastrow_evaluator(basis_params):
    beta_a, beta_b, ion_cusp, rcut, gamma = basis_params

    # vectorize polypade functions
    _inner_polypade = jax.vmap(polypade, in_axes=(0, None, None), out_axes=0) # [(n,), float, float] -> (n,)
    jit_vmapped_polypade = jax.jit(jax.vmap(_inner_polypade, in_axes=(None, 0, None), out_axes=1)) # [(n,), (nbasis,), float] -> (n, nbasis)
    jit_cutoffcusp = jax.jit(cutoffcusp)

    a_sum_evaluator = partial(compute_basis_sum, basis=jit_vmapped_polypade, param=beta_a, rcut=rcut)
    b_sum_evaluator = partial(compute_basis_sum, basis=jit_vmapped_polypade, param=beta_b, rcut=rcut)
    cusp_sum_evaluator = partial(compute_basis_sum, basis=jit_cutoffcusp, param=gamma, rcut=rcut)

    return [jax.jit(vmap_over_configs(evaluator)) for evaluator in [a_sum_evaluator, b_sum_evaluator, cusp_sum_evaluator]]


class JAXJastrowSpin:

    def __init__(self, mol, ion_cusp=None, na=4, nb=3, rcut=None, gamma=None, beta0_a=0.2, beta0_b=0.5):
        self._mol = mol
        # initialize self.basis_params and self.parameters
        self._init_params(ion_cusp, na, nb, rcut, gamma, beta0_a, beta0_b)
        self._recompute = create_jastrow_evaluator(self.basis_params)


    def _init_params(self, ion_cusp, na, nb, rcut, gamma, beta0_a, beta0_b):
        # replacing wftools.default_jastrow_basis() and wftools.generate_jastrow()
        # since we don't use func3d.PolyPadeFunction() and func3d.CutoffCuspFunction() objects anymore
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

        self.parameters = {}
        if len(ion_cusp) > 0:
            self.parameters["acoeff"] = jnp.zeros((self._mol.natm, len(beta_a)+1, 2))
        else:
            self.parameters["acoeff"] = jnp.zeros((self._mol.natm, len(beta_a), 2))
        self.parameters["bcoeff"] = jnp.zeros((len(beta_b)+1, 3))
        
        if len(ion_cusp) > 0:
            coefs = mol.atom_charges().copy()
            coefs[[l[0] not in ion_cusp for l in mol._atom]] = 0.0
            self.parameters["acoeff"] = self.parameters["acoeff"].at[:, 0, :].set(jnp.array(coefs[:, None]))
        self.parameters["bcoeff"] = self.parameters["bcoeff"].at[0, [0, 1, 2]].set(jnp.array([-0.25, -0.50, -0.25]))
        

    def _compute_bdiag_corr(self):
        nup, ndn = self._mol.nelec
        bcoeff = self.parameters["bcoeff"]
        rcut = self.basis_params.rcut
        gamma = self.basis_params.gamma
        diag = jnp.sum(bcoeff[1:, 0]) * nup + jnp.sum(bcoeff[1:, 2]) * ndn
        diagc = rcut/(3+gamma) * (bcoeff[0, 0] * nup + bcoeff[0, 2] * ndn)
        return diag + diagc
    

    def recompute(self, configs):
        nup, ndn = self._mol.nelec
        nconfig = configs.configs.shape[0]
        atom_coords = jnp.array(self._mol.atom_coords())
        tiled_atom_coords = jnp.tile(atom_coords, (nconfig, 1, 1))

        configs_up = jnp.array(configs.configs[:, :nup, :])
        configs_dn = jnp.array(configs.configs[:, nup:, :])

        acoeff = self.parameters["acoeff"]
        bcoeff = self.parameters["bcoeff"]

        # Should we prioritize clarity or compactness?

        # compute a terms for up and down electrons
        if len(self.basis_params.ion_cusp) > 0:
            a_up = self._recompute[0](configs_up, tiled_atom_coords, acoeff[:, 1:, 0])
            a_dn = self._recompute[0](configs_dn, tiled_atom_coords, acoeff[:, 1:, 1])
            ac_up = self._recompute[2](configs_up, tiled_atom_coords, acoeff[:, 0, 0])
            ac_dn = self._recompute[2](configs_dn, tiled_atom_coords, acoeff[:, 0, 1])
        else:
            a_up = self._recompute[0](configs_up, tiled_atom_coords, acoeff[:, :, 0])
            a_dn = self._recompute[0](configs_dn, tiled_atom_coords, acoeff[:, :, 1])
            ac_up = jnp.zeros((nconfig, nup))
            ac_dn = jnp.zeros((nconfig, ndn))

        # compute b terms for up-up, up-down, down-down electron pairs
        b_upup = self._recompute[1](configs_up, configs_up, jnp.tile(bcoeff[1:, 0], (nup, 1)))
        b_updn = self._recompute[1](configs_up, configs_dn, jnp.tile(bcoeff[1:, 1], (ndn, 1)))
        b_dndn = self._recompute[1](configs_dn, configs_dn, jnp.tile(bcoeff[1:, 2], (ndn, 1)))
        bc_upup = self._recompute[2](configs_up, configs_up, jnp.tile(bcoeff[0, 0], (nup)))
        bc_updn = self._recompute[2](configs_up, configs_dn, jnp.tile(bcoeff[0, 1], (ndn)))
        bc_dndn = self._recompute[2](configs_dn, configs_dn, jnp.tile(bcoeff[0, 2], (ndn)))
        bdiag_corr = self._compute_bdiag_corr()

        # sum the terms up to obtain the log Jastrow factor
        a = jnp.sum(a_up + a_dn + ac_up + ac_dn, axis=1)
        b = jnp.sum((b_upup + b_dndn + bc_upup  + bc_dndn)/2 + b_updn + bc_updn, axis=1)
        b -= bdiag_corr / 2
        return a+b



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

    import pyqmc.api as pyq
    mol = {}
    mf = {}

    mol['h2o'] = pyscf.gto.Mole(atom = '''O 0 0 0; H  0 2.0 0; H 0 0 2.0''', basis = 'cc-pVDZ', cart=True)
    mol['h2o'].build()
    
    jax_jastrow = JAXJastrowSpin(mol['h2o'])
    jastrow, _ = pyqmc.wftools.generate_jastrow(mol['h2o'])

    data = []
    for nconfig in [10, 1000, 100000]:
        configs = pyq.initial_guess(mol['h2o'], nconfig)

        jastrowval = jax_jastrow.recompute(configs) 
        jax.block_until_ready(jastrowval)

        jax_start = time.perf_counter()
        jastrowval = jax_jastrow.recompute(configs) 
        jax.block_until_ready(jastrowval)
        jax_end = time.perf_counter()

        slater_start = time.perf_counter()
        values_ref = jastrow.recompute(configs)
        slater_end = time.perf_counter()


        print("jax", jax_end-jax_start, "slater", slater_end-slater_start)
        print('MAD', jnp.mean(jnp.abs(values_ref[1]- jastrowval)))
        print("jax values", jastrowval)
        print("pyqmc values", values_ref[1])

        data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax'})
        data.append({'N': nconfig, 'time': slater_end-slater_start, 'method': 'pyqmc'})
    
    sns.lineplot(data = pd.DataFrame(data), x='N', y='time', hue='method')
    plt.ylim(0)
    plt.savefig("jax_vs_pyqmc.png")