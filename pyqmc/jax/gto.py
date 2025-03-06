import jax
import jax.numpy as jnp
import logging
from functools import partial
from typing import Generator

import numpy as np

log = logging.getLogger()

def angular_momentum_xyz(ell: int) -> Generator[tuple[int, int, int], None, None]:
    for lx in reversed(range(ell + 1)):
        for ly in reversed(range(ell + 1 - lx)):
            lz = ell - lx - ly
            yield (lx, ly, lz)
            #basis = 'x' * lx + 'y' * ly + 'z' * lz
 

# This is intended to be used with partial, vmap, and jit.
def _cartesian_gto(
    centers: jax.Array,
    ijk: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
    xyz: jax.Array) -> jax.Array:
  ''' Evaluate a Cartesian Gaussian-type orbital.

    This evaluates a potentially large collection of basis functions, each of
    which is a mixture of Gaussians.  For JAX reasons we assume that all the
    mixtures have the same number of Gaussians.  We take there to be N basis
    functions, each with M Gaussians.

    This function is based on code originally written by Ryan Adams, with modification by Lucas Wagner

  Args:
    centers: The centers of each Gaussian mixture. (N, 3)

    ijk: The angular momentum of each Gaussian mixture. (N, 3)

    expts: The exponents of each Gaussian mixture. (N, M)

    coeffs: The coefficients of each Gaussian mixture. (N, M) 
             It's assumed these are chose such that x^i y^j z^k exp(-alpha r^2) 
             is normalized to whatever standard you want (we don't do any normalization of the functions).

    xyz: The point at which to evaluate the basis functions. (3,)

  Returns:
    The value of each basis function at the point. (N,)
  '''
  centers2d: jax.Array = jnp.atleast_2d(centers)
  ctr_xyz: jax.Array = xyz[jnp.newaxis,:] - centers2d # (N, 3)
  # Cartesian monomials.
  xyz_ijk: jax.Array = jnp.prod(ctr_xyz**ijk, axis=1)[:,jnp.newaxis] # (N,1)
  # Actual Gaussian.
  gauss: jax.Array = jnp.exp(-expts * jnp.sum(ctr_xyz**2, axis=1)[:, jnp.newaxis]) # (N,M)
  all_prod: jax.Array = coeffs* gauss * xyz_ijk 

  return jnp.sum(all_prod, axis=1) #(N)


def create_gto_evaluator(mol):
    centers = mol.atom_coords()
    centers = jnp.array(centers)
    natom = mol.natm
    atom_names = [mol.atom_pure_symbol(i) for i in range(natom)]
    centers_aos = []
    ijks = []
    expts = []
    coeffs = []
    norms = [np.sqrt(1.0/(4*np.pi)), np.sqrt(1./(4.0*np.pi)), np.sqrt(1.0/15), 
             np.sqrt(1.0/105), np.sqrt(1.0/945)]
    for nm, atom in zip(atom_names, centers):
        basis = mol._basis[nm]
        for b in basis:
            ell = b[0]
            exp_coeffs = np.asarray(b[1:])
            for ijk in angular_momentum_xyz(ell):
                centers_aos.append(atom)
                ijks.append(ijk)
                expts.append(exp_coeffs[:,0])
                ijk_sum = sum(ijk) # (1,)
                norm = jnp.sqrt(2*(4*exp_coeffs[:,0])**(ijk_sum+1) * jnp.sqrt(2*exp_coeffs[:,0]/jnp.pi) )  # (N,1)

                coeffs.append(exp_coeffs[:,1]*norms[ell]*norm)


    max_len: int = max(len(e) for e in expts)
    for ii in range(len(expts)):
      expts[ii] = jnp.pad(expts[ii], (0, max_len - len(expts[ii])))
      coeffs[ii] = jnp.pad(coeffs[ii], (0, max_len - len(coeffs[ii])))

    evaluator = partial(
        _cartesian_gto,
        jnp.array(centers_aos),
        jnp.array(ijks),
        jnp.array(expts),
        jnp.array(coeffs),
    )
    return evaluator


if __name__=="__main__":
    cpu=True
    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update("jax_enable_x64", True)
    else:
        pass 

    import pyscf
    mol = pyscf.gto.Mole(atom = '''O 0 0 0; H  0 2.0 0; H 0 0 2.0''', basis = 'unc-ccecp-ccpvdz', ecp='ccecp', cart=True)
    mol.build()

    import time
    evaluator = create_gto_evaluator(mol)
    evaluator_val = jax.jit(jax.vmap(evaluator, in_axes=(0)))
    evaluator_gradient = jax.jacfwd(evaluator)
    evaluator_gradient = jax.vmap(evaluator_gradient, in_axes=(0))
    evaluator_gradient = jax.jit(evaluator_gradient)


    nconfig = 2000
    seed = 32123
    key = jax.random.key(seed)
    coords = jax.random.normal(key, (nconfig, 3))

    res = evaluator_val(coords)
    jax.block_until_ready(res)

    start = time.time()
    res = evaluator_val(coords)
    jax.block_until_ready(res)
    end = time.time()
    val_time = end-start
    print("Time taken for value: ", end-start)


    res = evaluator_gradient(coords)
    jax.block_until_ready(res)

    start = time.time()
    res = evaluator_gradient(coords)
    jax.block_until_ready(res)
    end = time.time()
    grad_time = end-start
    print("Time taken for gradient: ", end-start)

    print("grad ratio", grad_time/val_time)
