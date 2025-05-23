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


def minimum_image(rij_vecs, lattice_vectors):
    """
    Wrap displacement vectors according to the minimum-image convention.

    Args:
        rij_vecs (jax.Array): displacement vectors. (n, 3)
        lattice_vectors (jax.Array): lattice vectors. (3, 3)

    Return:
        jax.Array: wrapped displacement vectors. (n, 3)
    """
    frac_vecs = rij_vecs @ jnp.linalg.inv(lattice_vectors)
    frac_vecs -= jnp.round(frac_vecs)
    return frac_vecs @ lattice_vectors


def partial_basis_sum(elec_coord, coords, coeff, basis, param, rcut, wrap_vecs):
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
        basis (Callable[(n), (nbasis,), float -> (n, nbasis)]): Vmapped basis function (polypade or cutoffcusp).
        param (jax.Array): The beta parameters or the cusp gamma parameter. (nbasis,)
        rcut (float): Cutoff radius.
        wrap_vecs (Callable[(n, 3)]): Mininum image distance function (for PBC) or identity function (for OBC).
            
    Return:
        float: Sum.
    """
    rij_vecs = wrap_vecs(elec_coord - coords)
    rij = jnp.linalg.norm(rij_vecs, axis=-1)
    basis_val = basis(rij, param, rcut) # (n, nbasis)
    return jnp.sum(coeff * basis_val)


def partial_basis_sum_grad(elec_coord, coords, coeff, basis_grad, param, rcut, wrap_vecs):
    """
    Analytic gradient of partial basis sum with respect to elec_coord.
    """
    rij_vecs = wrap_vecs(elec_coord - coords)
    rij = jnp.linalg.norm(rij_vecs, axis=-1)
    rij_hat = rij_vecs / rij[:, jnp.newaxis]
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
           + cusp_eval(coords_up, coords_dn, jnp.tile(bcoeff[:1, 1], (ndn, 1)))
    b_dndn = b_eval(coords_dn, coords_dn, jnp.tile(bcoeff[1:, 2], (ndn, 1))) \
           + cusp_eval(coords_dn, coords_dn, jnp.tile(bcoeff[:1, 2], (ndn, 1)))
    bdiag_corr = compute_bdiag_corr(mol, basis_params, parameters)

    # sum the terms over electrons
    a = jnp.sum(a_up) + jnp.sum(a_dn)
    b = (jnp.sum(b_upup) + jnp.sum(b_dndn))/2 + jnp.sum(b_updn)
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
         + cusp_eval(epos, coords_dn, jnp.tile(bcoeff[:1, spin+1], (ndn, 1)))
    b_up_old = b_eval(epos_old, coords_old_up, jnp.tile(bcoeff[1:, spin+0], (nup, 1))) \
             + cusp_eval(epos_old, coords_old_up, jnp.tile(bcoeff[:1, spin+0], (nup, 1)))
    b_dn_old = b_eval(epos_old, coords_old_dn, jnp.tile(bcoeff[1:, spin+1], (ndn, 1))) \
             + cusp_eval(epos_old, coords_old_dn, jnp.tile(bcoeff[:1, spin+1], (ndn, 1)))

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
    bc_updn_pgrad = cusp_pgrad_eval(coords_up, coords_dn, jnp.tile(bcoeff[:1, 1], (ndn, 1)))
    b_dndn_pgrad = b_pgrad_eval(coords_dn, coords_dn, jnp.tile(bcoeff[1:, 2], (ndn, 1))) / 2
    bc_dndn_pgrad = cusp_pgrad_eval(coords_dn, coords_dn, jnp.tile(bcoeff[:1, 2], (ndn, 1))) / 2
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

    # use minimum image distance function for periodic systems
    if hasattr(mol, "lattice_vectors"):
        wrap_vecs = lambda rij_vecs: minimum_image(rij_vecs, jnp.array(mol.lattice_vectors()))
    else:
        wrap_vecs = lambda rij_vecs: rij_vecs

    # lists of basis functions and parameters for constructing psum evaluators for a, b and cusp terms
    basis_vm_lst = [polypade_vm, polypade_vm, cutoffcusp_vm]
    basis_grad_vm_lst = [polypade_grad_vm, polypade_grad_vm, cutoffcusp_grad_vm]
    basis_params_lst = [beta_a, beta_b, gamma]

    # lists of partial basis sum evaluators for a, b and cusp terms
    psum_evaluators_elec = [
            jax.vmap( # vmapping over electrons
                partial(partial_basis_sum, basis=basis, param=param, rcut=rcut, wrap_vecs=wrap_vecs), 
                in_axes=(0, None, None)
            )
        for basis, param in zip(basis_vm_lst, basis_params_lst)]
    psum_evaluators = [partial(partial_basis_sum, basis=basis, param=param, rcut=rcut, wrap_vecs=wrap_vecs)
        for basis, param in zip(basis_vm_lst, basis_params_lst)]
    psum_grad_evaluators = [partial(partial_basis_sum_grad, basis_grad=basis, param=param, rcut=rcut, wrap_vecs=wrap_vecs)
        for basis, param in zip(basis_grad_vm_lst, basis_params_lst)]
    psum_hess_evaluators = [partial(jax.hessian(partial_basis_sum, argnums=0), basis=basis, param=param, rcut=rcut, wrap_vecs=wrap_vecs)
        for basis, param in zip(basis_vm_lst, basis_params_lst)]
    psum_pgrad_evaluators = [
            jax.vmap( # vmapping over electrons
                partial(jax.grad(partial_basis_sum, argnums=2), basis=basis, param=param, rcut=rcut, wrap_vecs=wrap_vecs), 
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
    def __init__(self, jax_parameters, has_ion_cusp):
        self.has_ion_cusp = has_ion_cusp
        self.jax_parameters = jax_parameters
        if self.has_ion_cusp:
            self.parameters= {'acoeff': np.asarray(self.jax_parameters.acoeff), 
                              'bcoeff': np.asarray(self.jax_parameters.bcoeff)}
        else:
            self.parameters= {'acoeff': np.asarray(self.jax_parameters.acoeff[:, 1:, :]), 
                              'bcoeff': np.asarray(self.jax_parameters.bcoeff)}

    def __setitem__(self, key, item):
        self.parameters[key] = item
        if self.has_ion_cusp:
            self.jax_parameters = CoefficientParameters(jnp.array(self.parameters['acoeff']),
                                                        jnp.array(self.parameters['bcoeff']))
        else:
            padded_acoeff = np.pad(self.parameters['acoeff'], ((0,0), (1,0), (0,0)))
            self.jax_parameters = CoefficientParameters(jnp.array(padded_acoeff),
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

        if ion_cusp is None:
            self.has_ion_cusp = False
        else:
            self.has_ion_cusp = len(ion_cusp) > 0
        if self.has_ion_cusp is True:
            coefs = jnp.array(mol.atom_charges(), dtype=jnp.float64)
            mask = jnp.array([l[0] not in ion_cusp for l in mol._atom])
            coefs = coefs.at[mask].set(0.0)
            acoeff = acoeff.at[:, 0, :].set(coefs[:, None])
        bcoeff = bcoeff.at[0, [0, 1, 2]].set(jnp.array([-0.25, -0.50, -0.25]))

        self.parameters = _parameterMap(CoefficientParameters(acoeff, bcoeff), self.has_ion_cusp)
        

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
        return jnp.ones(len(self._logj)), self._logj


    def testvalue(self, e, epos, mask=None):
        """
        Compute the test value (:math:`\log (J(\vec{R}') / J(\vec{R}))`) for a single-electron update.
        epos.configs can have shape (nconfig, 3) or (nconfig, naip, 3).
        The second return value is delta log Jastrow, used as saved_values for updateinternals()
        """
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        jnpmask = jnp.array(mask, dtype=bool)
        _configs_old = jnp.array(self._configscurrent.configs)
        _epos = jnp.array(epos.configs)
        spin = int(e >= self._mol.nelec[0])
        if len(_epos.shape) == 2:
            evaluator = self._testvalue
        else:
            evaluator = self._testvalue_aux
        testval = evaluator(self.parameters.jax_parameters, _configs_old, e, spin, _epos)[jnpmask]
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
    
    def laplacian(self, e, epos):
        """
        :math:`\frac{\nabla_e^2 J(\vec{R})}{J(\vec{R})} = \nabla_e^2 \log J(\vec{R}) + |\nabla_e \log J(\vec{R})|^2.`
        """
        return self.gradient_laplacian(e, epos)[1]
    

    def pgradient(self):
        _configs = jnp.array(self._configscurrent.configs)
        a_pgrad, b_pgrad = self._pgradient(self.parameters.jax_parameters, _configs)
        if self.has_ion_cusp:
            return {
                "acoeff": a_pgrad,
                "bcoeff": b_pgrad,
            }
        else:
            return {
                "acoeff": a_pgrad[:, :, 1:, :],
                "bcoeff": b_pgrad,
            }
    