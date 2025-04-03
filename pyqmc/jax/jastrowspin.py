import jax
import jax.numpy as jnp
import numpy as np
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


def z(x): return x**2 * (6 - 8*x + 3*x**2)
def p(y): return ((y-1)**3 + 1)/3
def p_grad(y): return (y-1)**2
def z_grad(x): return 12*x * (1 - 2*x + x**2)


def polypade(rij, beta, rcut):
    """
    :math:`a(\vec{r}_i, \vec{r}_j, \beta_k, r_{cut}) = \frac{1-z(r)}{1+\beta_k z(r)}`, where
    :math:`r = r_{ij}/r_{cut}, z(x) = x^2(6-8x+3x^2)`.

    Args:
        rij (float): Distance.
        beta (float): Beta parameter.
        rcut (float): Cutoff radius.

    Return:
        float: Function value.
    """
    r = rij / rcut
    func = (1 - z(r)) / (1 + beta * z(r))
    return jnp.where(rij > rcut, 0.0, func)


def cutoffcusp(rij, gamma, rcut):
    """
    :math:`a(\vec{r}_i, \vec{r}_j, \gamma, r_{cut}) = r_{cut} (-\frac{p(r)}{1+\gamma p(r)} + \frac{1}{3+\gamma})`, where
    :math:`r = r_{ij}/r_{cut}, p(y) = \frac{(y-1)^3+1}{3}`.

    Args:
        rij (float): Distance.
        gamma (float): Gamma parameter.
        rcut (float): Cutoff radius.

    Return:
        float: Function value.
    """
    r = rij / rcut
    func = - p(r) / (1 + gamma * p(r)) + 1 / (3 + gamma)
    return jnp.where(rij > rcut, 0.0, func * rcut)


def polypade_grad(rij, beta, rcut):
    """
    Derivative of polypade with respect to rij.
    """
    r = rij / rcut
    func = - (1 + beta) * z_grad(r) / (rcut * (1 + beta * z(r))**2)
    return jnp.where(rij > rcut, 0.0, func)


def cutoffcusp_grad(rij, gamma, rcut):
    """
    Derivative of cutoffcusp with respect to rij.
    """
    r = rij / rcut
    func = - p_grad(r) / (1 + gamma * p(r))**2
    return jnp.where(rij > rcut, 0.0, func)


def partial_basis_sum(elec_coord, coords, coeff, basis, param, rcut):
    """
    For a given electron, compute the sum over basis functions and atom coordinates (one-body) or electron coordinates (two-body).
    Given :math:`i`, compute :math:`\sum_{I,k} c_{I,k,\sigma(i)}^{en} a(r_{iI}, \beta_{k}^a)` or :math:`\sum_{j,k} c_{k,\sigma(ij)}^{ee} b(r_{ij}, \beta_{k}^b)`.
    To be vmapped over configurations and/or electrons for use in evaluators.
    In the following,
    n = natom for a terms, (nelec_up or nelec_dn) for b terms.
    nbasis = (na or nb) for polypade, 1 for cutoffcusp.
    
    Args:
        elec_coord (jax.Array): Electron coordinates. (3,)
        coords (jax.Array): Array of atom or electron coordinates. (n, 3)
        coeff (jax.Array): Jastrow coefficients. (n, nbasis)
        basis (Callable[(3,), (n, 3), (nbasis,), float -> (n, nbasis)]): Vmapped basis function (polypade or cutoffcusp).
        param (jax.Array): The beta parameters or the cusp gamma parameter. (nbasis,)
        rcut (float): Cutoff radius.
            
    Return:
        float: Sum.
    """
    rij = jnp.linalg.norm(elec_coord - coords, axis=-1)
    basis_val = basis(rij, param, rcut) # (n, nbasis)
    return jnp.sum(coeff * basis_val)


def partial_basis_sum_grad(elec_coord, coords, coeff, basis_grad, param, rcut):
    """
    Analytic gradient of partial basis sum with respect to elec_coord.
    """
    rij_vec = elec_coord - coords
    rij = jnp.linalg.norm(rij_vec, axis=-1)
    rij_hat = rij_vec / rij[:, jnp.newaxis]
    basis_grad_val = basis_grad(rij, param, rcut) # (n, nbasis)
    return jnp.einsum("Ik,Ik,Ii->i", coeff, basis_grad_val, rij_hat)


def vmap_basis(basis):
    """
    Maps basis functions (polypade or cutoffcusp) or their derivatives over their parameters
    """
    return jax.vmap(
        jax.vmap(basis, in_axes=(0, None, None), out_axes=0), # [(n,), float, float] -> (n,)
        in_axes=(None, 0, None), 
        out_axes=1
    ) # [(n,), (nbasis,), float] -> (n, nbasis)


polypade_vm = vmap_basis(polypade)
cutoffcusp_vm = vmap_basis(cutoffcusp)
polypade_grad_vm = vmap_basis(polypade_grad)
cutoffcusp_grad_vm = vmap_basis(cutoffcusp_grad)


def compute_bdiag_corr(mol, basis_params, parameters):
    """
    Compute the diagonal sum of the b terms as a correction, a helper function for evaluate_jastrow().
    """
    nup, ndn = mol.nelec
    bcoeff = parameters.bcoeff
    rcut, gamma = basis_params.rcut, basis_params.gamma[0]
    diag = jnp.sum(bcoeff[1:, 0]) * nup + jnp.sum(bcoeff[1:, 2]) * ndn
    diagc = rcut/(3+gamma) * (bcoeff[0, 0] * nup + bcoeff[0, 2] * ndn)
    return diag + diagc


def evaluate_jastrow(mol, basis_params, partial_sum_evaluators, parameters, coords):
    """
    Evaluate the log Jastrow factor (:math:`\log J(\vec{R})`).

    Args:
        mol (pyscf.gto.Mole): PySCF molecule object.
        basis_params (BasisParameters): Basis parameters.
        partial_sum_evaluators (list): List of vmapped partial_basis_sum() functions.
        parameters (CoefficientParameters): Jastrow coeffcients.
        coords (jax.Array): Electron coordinates. (nelec, 3)
    
    Return:
        a (float): Sum of a terms.
        b (float): Sum of b terms.
        logj (float): Log Jastrow factor.
    """
    nup, ndn = mol.nelec
    coords_up, coords_dn = coords[:nup, :], coords[nup:, :]
    atom_coords = jnp.array(mol.atom_coords())
    acoeff, bcoeff = parameters

    a_eval, b_eval, cusp_eval = partial_sum_evaluators
    
    # compute a terms for up and down electrons
    a_up = a_eval(coords_up, atom_coords, acoeff[:, 1:, 0]) \
         + cusp_eval(coords_up, atom_coords, acoeff[:, :1, 0]) # (nelec,)
    a_dn = a_eval(coords_dn, atom_coords, acoeff[:, 1:, 1]) \
         + cusp_eval(coords_dn, atom_coords, acoeff[:, :1, 1])

    # compute b terms for up-up, up-down, down-down electron pairs
    # bcoeff is tiled to match the first dimension (n) of coords_spin
    b_upup = b_eval(coords_up, coords_up, jnp.tile(bcoeff[1:, 0], (nup, 1))) \
           + cusp_eval(coords_up, coords_up, jnp.tile(bcoeff[:1, 0], (nup, 1))) # (nelec,)
    b_updn = b_eval(coords_up, coords_dn, jnp.tile(bcoeff[1:, 1], (ndn, 1))) \
           + cusp_eval(coords_up, coords_dn, jnp.tile(bcoeff[:1, 1], (nup, 1)))
    b_dndn = b_eval(coords_dn, coords_dn, jnp.tile(bcoeff[1:, 2], (ndn, 1))) \
           + cusp_eval(coords_dn, coords_dn, jnp.tile(bcoeff[:1, 2], (nup, 1)))
    bdiag_corr = compute_bdiag_corr(mol, basis_params, parameters)

    # sum the terms over electrons
    a = jnp.sum(a_up + a_dn)
    b = jnp.sum((b_upup + b_dndn)/2 + b_updn)
    b -= bdiag_corr / 2
    logj = a + b
    return a, b, logj


def evaluate_testvalue(mol, partial_sum_evaluators, parameters, coords_old, e, spin, epos):
    """
    Evaluate the test value (:math:`\log (J(\vec{R}') / J(\vec{R}))`) for a single-electron update.

    Args:
        mol (pyscf.gto.Mole): PySCF molecule object.
        partial_sum_evaluators (list): List of partial_basis_sum() functions.
        parameters (CoefficientParameters): Jastrow coeffcients.
        coords_old (jax.Array): Electron coordinates before the update. (nelec, 3)
        e (int): Index of the updated electron.
        spin (int): Spin of the updated electron.
        epos (jax.Array): New position of the updated electron. (3)
    
    Return:
        float: Test value.
    """
    nup, ndn = mol.nelec
    epos_old = coords_old[e, :]
    coords_old_up, coords_old_dn = coords_old[:nup, :], coords_old[nup:, :]
    coords = coords_old.copy().at[e, :].set(epos)
    coords_up, coords_dn = coords[:nup, :], coords[nup:, :]
    atom_coords = jnp.array(mol.atom_coords())
    acoeff, bcoeff = parameters

    a_eval, b_eval, cusp_eval = partial_sum_evaluators

    # compute a term contributed by electron e
    a = a_eval(epos, atom_coords, acoeff[:, 1:, spin]) \
      + cusp_eval(epos, atom_coords, acoeff[:, :1, spin]) # (float)
    a_old = a_eval(epos_old, atom_coords, acoeff[:, 1:, spin])\
          + cusp_eval(epos_old, atom_coords, acoeff[:, :1, spin])

    # compute b terms contributed by electron e
    # bcoeff is tiled to match the first dimension (n) of coords_spin
    b_up = b_eval(epos, coords_up, jnp.tile(bcoeff[1:, spin+0], (nup, 1))) \
         + cusp_eval(epos, coords_up, jnp.tile(bcoeff[:1, spin+0], (nup, 1))) # (float)
    b_dn = b_eval(epos, coords_dn, jnp.tile(bcoeff[1:, spin+1], (ndn, 1))) \
         + cusp_eval(epos, coords_dn, jnp.tile(bcoeff[:1, spin+1], (nup, 1)))
    b_up_old = b_eval(epos_old, coords_old_up, jnp.tile(bcoeff[1:, spin+0], (nup, 1))) \
             + cusp_eval(epos_old, coords_old_up, jnp.tile(bcoeff[:1, spin+0], (nup, 1)))
    b_dn_old = b_eval(epos_old, coords_old_dn, jnp.tile(bcoeff[1:, spin+1], (ndn, 1))) \
             + cusp_eval(epos_old, coords_old_dn, jnp.tile(bcoeff[:1, spin+1], (nup, 1)))

    delta_a = a - a_old
    # factor of 1/2 cancelled by the double appearance (row and column) of b terms contributed by electron e
    delta_b = b_up + b_dn - b_up_old - b_dn_old 
    delta_logj =  delta_a + delta_b
    return jnp.exp(delta_logj)


def evaluate_derivative(mol, partial_sum_derv_evaluators, parameters, coords_up, coords_dn, spin, epos):
    """
    Evaluate the derivative (gradient or hessian) of the log Jastrow factor
    (:math:`\nabla_e \log J(\vec{R})` or :math:`H(\log J(\vec{R}))`).

    Args:
        mol (pyscf.gto.Mole): PySCF molecule object.
        partial_sum_derv_evaluators (list): List of partial_basis_sum() derivative functions.
        parameters (CoefficientParameters): Jastrow coeffcients.
        coords_up (jax.Array): Spin-up electron coordinates with electron e removed (if spin=0). (nup, 3)
        coords_dn (jax.Array): Spin-down electron coordinates with electron e removed (if spin=1). (ndn, 3)
        spin (int): Spin of the electron with respect to which the gradient is taken.
        epos (jax.Array): Position of the electron e. (3,)
        
    Return:
        jax.Array: Gradient values (3,) or Hessian values (3, 3)
    """
    nup = coords_up.shape[0]
    ndn = coords_dn.shape[0]
    atom_coords = jnp.array(mol.atom_coords())
    acoeff, bcoeff = parameters

    a_derv_eval, b_derv_eval, cusp_derv_eval = partial_sum_derv_evaluators

    # compute derivative of a term
    a_derv = a_derv_eval(epos, atom_coords, acoeff[:, 1:, spin]) \
      + cusp_derv_eval(epos, atom_coords, acoeff[:, :1, spin]) # (3,) or (3, 3)
    
    # compute derivative of b terms
    # bcoeff is tiled to match the first dimension (n) of coords_spin
    b_derv_up = b_derv_eval(epos, coords_up, jnp.tile(bcoeff[1:, spin+0], (nup, 1))) \
         + cusp_derv_eval(epos, coords_up, jnp.tile(bcoeff[:1, spin+0], (nup, 1))) # (3,) or (3, 3)
    b_derv_dn = b_derv_eval(epos, coords_dn, jnp.tile(bcoeff[1:, spin+1], (ndn, 1))) \
         + cusp_derv_eval(epos, coords_dn, jnp.tile(bcoeff[:1, spin+1], (ndn, 1)))
    
    derv_logj = a_derv + b_derv_up + b_derv_dn
    return derv_logj


def evaluate_pgradient(mol, basis_params, partial_sum_pgrad_evaluators, parameters, coords):
    """
    Evaluate the coefficient gradient of the log Jastrow factor
    (:math:`\nabla_c \log J(\vec{R})`).

    Args:
        mol (pyscf.gto.Mole): PySCF molecule object.
        basis_params (BasisParameters): Basis parameters.
        partial_sum_pgrad_evaluators (list): List of vmapped partial_basis_sum() pgradient functions.
        parameters (CoefficientParameters): Jastrow coeffcients.
        coords (jax.Array): Electron coordinates. (nelec, 3)
    
    Return:
        a_pgrad (jax.Array): Gradient with respect to a coefficients. (natom, nbasis, 2)
        b_pgrad (jax.Array): Gradient with respect to b coefficients. (nbasis, 3)
    """
    nup, ndn = mol.nelec
    coords_up, coords_dn = coords[:nup, :], coords[nup:, :]
    atom_coords = jnp.array(mol.atom_coords())
    acoeff, bcoeff = parameters

    a_pgrad_eval, b_pgrad_eval, cusp_pgrad_eval = partial_sum_pgrad_evaluators

    # compute gradient with respect to a coefficients
    a_pgrad = jnp.zeros_like(acoeff)
    a_up_pgrad = a_pgrad_eval(coords_up, atom_coords, acoeff[:, 1:, 0]) # (nelec, natom, nbasis, 2)
    ac_up_pgrad = cusp_pgrad_eval(coords_up, atom_coords, acoeff[:, :1, 0])
    a_dn_pgrad = a_pgrad_eval(coords_dn, atom_coords, acoeff[:, 1:, 1])
    ac_dn_pgrad = cusp_pgrad_eval(coords_dn, atom_coords, acoeff[:, :1, 1])
    a_pgrad = a_pgrad.at[:, 1:, 0].set(jnp.sum(a_up_pgrad, axis=0))
    a_pgrad = a_pgrad.at[:, :1, 0].set(jnp.sum(ac_up_pgrad, axis=0))
    a_pgrad = a_pgrad.at[:, 1:, 1].set(jnp.sum(a_dn_pgrad, axis=0))
    a_pgrad = a_pgrad.at[:, :1, 1].set(jnp.sum(ac_dn_pgrad, axis=0))

    # compute gradient with respect to b coefficients
    b_pgrad = jnp.zeros_like(bcoeff)
    b_upup_pgrad = b_pgrad_eval(coords_up, coords_up, jnp.tile(bcoeff[1:, 0], (nup, 1))) / 2 # (nelec, nelec, nbasis, 3)
    bc_upup_pgrad = cusp_pgrad_eval(coords_up, coords_up, jnp.tile(bcoeff[:1, 0], (nup, 1))) / 2
    b_updn_pgrad = b_pgrad_eval(coords_up, coords_dn, jnp.tile(bcoeff[1:, 1], (ndn, 1)))
    bc_updn_pgrad = cusp_pgrad_eval(coords_up, coords_dn, jnp.tile(bcoeff[:1, 1], (nup, 1)))
    b_dndn_pgrad = b_pgrad_eval(coords_dn, coords_dn, jnp.tile(bcoeff[1:, 2], (ndn, 1))) / 2
    bc_dndn_pgrad = cusp_pgrad_eval(coords_dn, coords_dn, jnp.tile(bcoeff[:1, 2], (nup, 1))) / 2
    rcut, gamma = basis_params.rcut, basis_params.gamma[0]
    diagc = rcut / (3+gamma) # gradient of the diagonal correction term
    b_pgrad = b_pgrad.at[1:, 0].set(jnp.sum(b_upup_pgrad, axis=(0, 1)) - nup/2)
    b_pgrad = b_pgrad.at[:1, 0].set(jnp.sum(bc_upup_pgrad, axis=(0, 1)) - nup*diagc/2)
    b_pgrad = b_pgrad.at[1:, 1].set(jnp.sum(b_updn_pgrad, axis=(0, 1)))
    b_pgrad = b_pgrad.at[:1, 1].set(jnp.sum(bc_updn_pgrad, axis=(0, 1)))
    b_pgrad = b_pgrad.at[1:, 2].set(jnp.sum(b_dndn_pgrad, axis=(0, 1)) - ndn/2)
    b_pgrad = b_pgrad.at[:1, 2].set(jnp.sum(bc_dndn_pgrad, axis=(0, 1)) - ndn*diagc/2)

    return a_pgrad, b_pgrad


def create_jastrow_evaluator(mol, basis_params):
    """
    Create a set of functions that can be used to evaluate the Jastrow factor.
    """
    beta_a, beta_b, ion_cusp, rcut, gamma = basis_params

    # lists of basis functions and parameters for constructing psum evaluators for a, b and cusp terms
    basis_vm_lst = [polypade_vm, polypade_vm, cutoffcusp_vm]
    basis_grad_vm_lst = [polypade_grad_vm, polypade_grad_vm, cutoffcusp_grad_vm]
    basis_params_lst = [beta_a, beta_b, gamma]

    # lists of partial basis sum evaluators for a, b and cusp terms
    psum_evaluators_elec = [
            jax.vmap( # vmapping over electrons
                partial(partial_basis_sum, basis=basis, param=param, rcut=rcut), 
                in_axes=(0, None, None)
            )
        for basis, param in zip(basis_vm_lst, basis_params_lst)]
    psum_evaluators = [partial(partial_basis_sum, basis=basis, param=param, rcut=rcut)
        for basis, param in zip(basis_vm_lst, basis_params_lst)]
    psum_grad_evaluators = [partial(partial_basis_sum_grad, basis_grad=basis, param=param, rcut=rcut)
        for basis, param in zip(basis_grad_vm_lst, basis_params_lst)]
    psum_hess_evaluators = [partial(jax.hessian(partial_basis_sum, argnums=0), basis=basis, param=param, rcut=rcut)
        for basis, param in zip(basis_vm_lst, basis_params_lst)]
    psum_pgrad_evaluators = [
            jax.vmap( # vmapping over electrons
                partial(jax.grad(partial_basis_sum, argnums=2), basis=basis, param=param, rcut=rcut), 
                in_axes=(0, None, None)
            )
        for basis, param in zip(basis_vm_lst, basis_params_lst)]

    # create evaluators vmapped over configurations
    value_evaluator = jax.jit(jax.vmap(
        partial(evaluate_jastrow, mol, basis_params, psum_evaluators_elec), 
        in_axes=(None, 0))
    )
    testval_evaluator = jax.jit(jax.vmap(
        partial(evaluate_testvalue, mol, psum_evaluators),
        in_axes=((None, 0, None, None, 0)))
    )
    testval_aux_evaluator = jax.jit(jax.vmap(
        jax.vmap(
            partial(evaluate_testvalue, mol, psum_evaluators),
            in_axes=((None, None, None, None, 0))
        ),
        in_axes=((None, 0, None, None, 0)), out_axes=0)
    )
    grad_evaluator = jax.jit(jax.vmap(
        partial(evaluate_derivative, mol, psum_grad_evaluators),
        in_axes=(None, 0, 0, None, 0), out_axes=0)
    )
    hess_evaluator = jax.jit(jax.vmap(
        partial(evaluate_derivative, mol, psum_hess_evaluators),
        in_axes=(None, 0, 0, None, 0), out_axes=0)
    )
    pgrad_evaluator = jax.jit(jax.vmap(
        partial(evaluate_pgradient, mol, basis_params, psum_pgrad_evaluators), 
        in_axes=(None, 0))
    )
    return value_evaluator, testval_evaluator, testval_aux_evaluator, grad_evaluator, hess_evaluator, pgrad_evaluator


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
        self.dtype = float
        self._recompute, self._testvalue, self._testvalue_aux, self._gradient, \
            self._hessian, self._pgradient = create_jastrow_evaluator(self._mol, self.basis_params)


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

        beta_a = jnp.array(pyqmc.wftools.expand_beta_qwalk(beta0_a, na))
        beta_b = jnp.array(pyqmc.wftools.expand_beta_qwalk(beta0_b, nb))
        gamma = jnp.array([gamma])
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
        self._configscurrent = configs.copy()
        _configs = jnp.array(configs.configs)
        a, b, self._logj = self._recompute(self.parameters.jax_parameters, _configs)
        return jnp.ones(len(self._logj)), self._logj


    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        """
        Update the saved log Jastrow value.
        """
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        jnpmask = jnp.array(mask, dtype=bool)
        if saved_values is None:
            delta_logj = self.testvalue(e, epos)[1]
        else:
            delta_logj = saved_values
        delta_logj = delta_logj.at[jnp.logical_not(jnpmask)].set(0)
        self._logj += delta_logj
        self._configscurrent.move(e, epos, mask)


    def value(self):
        return self._logj


    def testvalue(self, e, epos, mask=None):
        """
        Compute the test value (:math:`\log (J(\vec{R}') / J(\vec{R}))`) for a single-electron update.
        epos.configs can have shape (nconfig, 3) or (nconfig, naip, 3).
        The second return value is delta log Jastrow, used as saved_values for updateinternals()
        """
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        jnpmask = jnp.array(mask, dtype=bool)
        _configs_old = jnp.array(self._configscurrent.configs)[jnpmask]
        _epos = jnp.array(epos.configs)[jnpmask]
        spin = int(e >= self._mol.nelec[0])
        if len(_epos.shape) == 2:
            evaluator = self._testvalue
        else:
            evaluator = self._testvalue_aux
        testval = evaluator(self.parameters.jax_parameters, _configs_old, e, spin, _epos)
        return testval, jnp.log(testval)


    def testvalue_many(self, e, epos, mask=None):
        """
        Compute the test values for moving electrons in e to epos.
        """
        testvals = []
        for ei in e:
            testvals.append(self.testvalue(ei, epos, mask)[0])
        return jnp.array(testvals).T 


    def _split_configs(self, e):
        """
        Split the configurations according to spin and remove electron e, helper function for
        computing gradient and laplacian.
        """
        _configs = jnp.array(self._configscurrent.configs)
        # _epos = _configs[:, e, :]
        spin = int(e >= self._mol.nelec[0])
        nelec = self._mol.nelec[0] + self._mol.nelec[1]

        # remove electron e from the configurations to avoid singularity in gradient
        mask = jnp.arange(nelec) != e
        nup, ndn = self._mol.nelec
        nup = nup - (1-spin)
        ndn = ndn - spin
        configs_up = _configs[:, mask, :][:, :nup, :]
        configs_dn = _configs[:, mask, :][:, nup:, :]
        return configs_up, configs_dn, spin


    def gradient(self, e, epos):
        """
        :math:`\frac{\nabla_e J(\vec{R})}{J(\vec{R})} = \nabla_e \log J(\vec{R})`.
        """
        configs_up, configs_dn, spin = self._split_configs(e)
        _epos = jnp.array(epos.configs)
        return self._gradient(self.parameters.jax_parameters, configs_up, configs_dn, spin, _epos).T


    def gradient_value(self, e, epos):
        grad = self.gradient(e, epos)
        testval, delta_logj = self.testvalue(e, epos)
        return grad, testval, delta_logj


    def gradient_laplacian(self, e, epos):
        """
        :math:`\frac{\nabla_e^2 J(\vec{R})}{J(\vec{R})} = \nabla_e^2 \log J(\vec{R}) + |\nabla_e \log J(\vec{R})|^2.`
        """
        configs_up, configs_dn, spin = self._split_configs(e)
        _epos = jnp.array(epos.configs)
        grad = self._gradient(self.parameters.jax_parameters, configs_up, configs_dn, spin, _epos).T
        lap = jnp.trace(
            self._hessian(self.parameters.jax_parameters, configs_up, configs_dn, spin, _epos), 
            axis1=1, axis2=2
        ) + jnp.sum(grad**2, axis=0)
        return grad, lap
    

    def pgradient(self):
        _configs = jnp.array(self._configscurrent.configs)
        a_pgrad, b_pgrad = self._pgradient(self.parameters.jax_parameters, _configs)
        return {
            "acoeff": a_pgrad,
            "bcoeff": b_pgrad,
        }


def run_tests(benchmark=False, mad_only=True):
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyscf.gto
    import pyqmc.api as pyq
    from pyqmc.coord import OpenElectron

    mol = {}

    mol['h2o'] = pyscf.gto.Mole(atom = '''O 0 0 0; H  0 2.0 0; H 0 0 2.0''', basis = 'cc-pVDZ', cart=True)
    mol['h2o'].build()
    
    jax_jastrow = JAXJastrowSpin(mol['h2o'])
    jastrow, _ = pyqmc.wftools.generate_jastrow(mol['h2o'])

    data = []
    nconfigs = [10, 1000, 100000] if benchmark else [3]

    for nconfig in nconfigs:
        configs = pyq.initial_guess(mol['h2o'], nconfig)
        new_configs = configs.copy()
        new_configs.electron(7).configs += np.random.normal(0, 0.1, configs.electron(7).configs.shape)


        # Test recompute
        jax_jastrowval = jax_jastrow.recompute(configs)[1]
        jax.block_until_ready(jax_jastrowval)

        jax_start = time.perf_counter()
        jax_jastrowval = jax_jastrow.recompute(configs)[1]
        jax.block_until_ready(jax_jastrowval)
        jax_end = time.perf_counter()

        pyqmc_start = time.perf_counter()
        pyqmc_jastrowval = jastrow.recompute(configs)[1]
        pyqmc_end = time.perf_counter()

        if not benchmark:
            if mad_only:
                print('Jastrow MAD', jnp.mean(jnp.abs(pyqmc_jastrowval - jax_jastrowval)))
            else:
                print("jax Jastrow", jax_jastrowval)
                print("pyqmc Jastrow", pyqmc_jastrowval)
        else:
            data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax', 'value': 'Jastrow'})
            data.append({'N': nconfig, 'time': pyqmc_end-pyqmc_start, 'method': 'pyqmc', 'value': 'Jastrow'})


        # Test testvalue
        mask = None
        # mask = [False, True, True]
        aux_electron7 = OpenElectron(np.tile(new_configs.configs[:, np.newaxis, 7], (1, 3, 1)), new_configs.dist)

        jax_testval = jax_jastrow.testvalue(7, new_configs.electron(7), mask)[0]
        # jax_testval = jax_jastrow.testvalue(7, aux_electron7, mask)[0]
        jax.block_until_ready(jax_testval)

        jax_start = time.perf_counter()
        jax_testval = jax_jastrow.testvalue(7, new_configs.electron(7), mask)[0]
        # jax_testval = jax_jastrow.testvalue(7, aux_electron7, mask)[0]
        jax.block_until_ready(jax_testval)
        jax_end = time.perf_counter()

        pyqmc_start = time.perf_counter()
        pyqmc_testval = jastrow.testvalue(7, new_configs.electron(7), mask)[0]
        # pyqmc_testval = jastrow.testvalue(7, aux_electron7, mask)[0]
        pyqmc_end = time.perf_counter()

        if not benchmark:
            if mad_only:
                print('Testvalue MAD', jnp.mean(jnp.abs(pyqmc_testval - jax_testval)))
            else:
                print("jax testval", jax_testval)
                print("pyqmc testval", pyqmc_testval)
        else:
            data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax', 'value': 'Testvalue'})
            data.append({'N': nconfig, 'time': pyqmc_end-pyqmc_start, 'method': 'pyqmc', 'value': 'Testvalue'})


        # Test testvalue_many
        jax_testval_many = jax_jastrow.testvalue_many(np.array([2, 7]), new_configs.electron(7))
        jax.block_until_ready(jax_testval_many)

        jax_start = time.perf_counter()
        jax_testval_many = jax_jastrow.testvalue_many(np.array([2, 7]), new_configs.electron(7))
        jax.block_until_ready(jax_testval_many)
        jax_end = time.perf_counter()

        pyqmc_start = time.perf_counter()
        pyqmc_testval_many = jastrow.testvalue_many(np.array([2, 7]), new_configs.electron(7))
        pyqmc_end = time.perf_counter()

        if not benchmark:
            if mad_only:
                print('Testvalue many MAD', jnp.mean(jnp.abs(pyqmc_testval_many - jax_testval_many)))
            else:
                print("jax testval many", jax_testval_many)
                print("pyqmc testval many", pyqmc_testval_many)
        else:
            data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax', 'value': 'Testvalue many'})
            data.append({'N': nconfig, 'time': pyqmc_end-pyqmc_start, 'method': 'pyqmc', 'value': 'Testvalue many'})



        # Test gradient_value
        jax_gradient, jax_testval2, jax_delta_logj = jax_jastrow.gradient_value(7, new_configs.electron(7))
        jax.block_until_ready(jax_gradient)

        jax_start = time.perf_counter()
        jax_gradient, jax_testval2, jax_delta_logj = jax_jastrow.gradient_value(7, new_configs.electron(7))
        jax.block_until_ready(jax_gradient)
        jax_end = time.perf_counter()

        pyqmc_start = time.perf_counter()
        pyqmc_gradient, pyqmc_testvalue2, _ = jastrow.gradient_value(7, new_configs.electron(7))
        pyqmc_end = time.perf_counter()

        if not benchmark:
            if mad_only:
                print('Gradient MAD', jnp.mean(jnp.abs(pyqmc_gradient - jax_gradient)))
                print('Gradient value MAD', jnp.mean(jnp.abs(pyqmc_testvalue2 - jax_testval2)))
            else:
                print("jax gradient value", jax_gradient, jax_testval2)
                print("pyqmc gradient value", pyqmc_gradient, pyqmc_testvalue2)
        else:
            data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax', 'value': 'Gradient'})
            data.append({'N': nconfig, 'time': pyqmc_end-pyqmc_start, 'method': 'pyqmc', 'value': 'Gradient'})


        # Test gradient_laplacian
        jax_laplacian = jax_jastrow.gradient_laplacian(7, new_configs.electron(7))[1]
        jax.block_until_ready(jax_laplacian)

        jax_start = time.perf_counter()
        jax_laplacian = jax_jastrow.gradient_laplacian(7, new_configs.electron(7))[1]
        jax.block_until_ready(jax_laplacian)
        jax_end = time.perf_counter()

        pyqmc_start = time.perf_counter()
        pyqmc_laplacian = jastrow.gradient_laplacian(7, new_configs.electron(7))[1]
        pyqmc_end = time.perf_counter()

        if not benchmark:
            if mad_only:
                print('Laplacian MAD', jnp.mean(jnp.abs(pyqmc_laplacian - jax_laplacian)))
            else:
                print("jax laplacian", jax_laplacian)
                print("pyqmc laplacian", pyqmc_laplacian)
        else:
            data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax', 'value': 'Laplacian'})
            data.append({'N': nconfig, 'time': pyqmc_end-pyqmc_start, 'method': 'pyqmc', 'value': 'Laplacian'})


        # Test pgradient
        jax_pgrad = jax_jastrow.pgradient() 
        jax.block_until_ready(jax_pgrad)

        jax_start = time.perf_counter()
        jax_pgrad = jax_jastrow.pgradient() 
        jax.block_until_ready(jax_pgrad)
        jax_end = time.perf_counter()

        pyqmc_start = time.perf_counter()
        pyqmc_pgrad = jastrow.pgradient()
        pyqmc_end = time.perf_counter()

        if not benchmark:
            if mad_only:
                print('Pgradient acoeff MAD', jnp.mean(jnp.abs(pyqmc_pgrad["acoeff"] - jax_pgrad["acoeff"])))
                print('Pgradient bcoeff MAD', jnp.mean(jnp.abs(pyqmc_pgrad["bcoeff"] - jax_pgrad["bcoeff"])))
            else:
                print("jax pgradient", jax_pgrad)
                print("pyqmc pgradient", pyqmc_pgrad)
        else:
            data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax', 'value': 'PGradient'})
            data.append({'N': nconfig, 'time': pyqmc_end-pyqmc_start, 'method': 'pyqmc', 'value': 'PGradient'})
    
    if benchmark:
        data = pd.DataFrame(data)
        g = sns.relplot(data=data, x='N', y='time', hue='method', col='value', kind='line', col_wrap=3)
        plt.ylim(0)
        plt.savefig("jax_vs_pyqmc.png")


if __name__ == "__main__":
    cpu=True
    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update("jax_enable_x64", True)
    else:
        pass
    run_tests(benchmark=False, mad_only=True)
    