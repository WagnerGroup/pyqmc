import jax
import jax.numpy as jnp
import logging
from functools import partial
from typing import Generator
import itertools
import numpy as np

log = logging.getLogger()


def angular_momentum_xyz(ell: int) -> Generator[tuple[int, int, int], None, None]:
    for lx in reversed(range(ell + 1)):
        for ly in reversed(range(ell + 1 - lx)):
            lz = ell - lx - ly
            yield (lx, ly, lz)
            # basis = 'x' * lx + 'y' * ly + 'z' * lz


# This is intended to be used with partial, vmap, and jit.
def _cartesian_gto(
    centers: jax.Array,
    ijk: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
    images: jax.Array,
    xyz: jax.Array,
) -> jax.Array:
    """Evaluate a Cartesian Gaussian-type orbital.

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

      images: what images to evaluate the basis functions at: (N, nimages, 3)

      xyz: The point at which to evaluate the basis functions. (3,)

    Returns:
      The value of each basis function at the point. (N,)
    """
    centers2d: jax.Array = jnp.atleast_2d(centers)
    ctr_xyz_first: jax.Array = xyz[jnp.newaxis, :] - centers2d  # (N, 3)
    ctr_xyz = ctr_xyz_first[:, jnp.newaxis, :] + images  # (N, nimages, 3)
    # Cartesian monomials.
    xyz_ijk: jax.Array = jnp.prod(ctr_xyz ** ijk[:, jnp.newaxis, :], axis=-1)[
        :, :, jnp.newaxis
    ]  # (N,nimages, 1)
    # Actual Gaussian.
    gauss: jax.Array = jnp.exp(
        -expts[:, jnp.newaxis, :] * jnp.sum(ctr_xyz**2, axis=-1)[:, :, jnp.newaxis]
    )  # (N,nimages, M)
    all_prod: jax.Array = coeffs * jnp.sum(
        gauss * xyz_ijk, axis=1
    )  # (N,M)   sum over images

    return jnp.sum(all_prod, axis=1)  # (N)


def _cartesian_gto_grad(
    centers: jax.Array,
    ijk: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
    xyz: jax.Array,
) -> jax.Array:
    """Evaluate a Cartesian Gaussian-type orbital.

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
    """
    # Distances
    centers2d: jax.Array = jnp.atleast_2d(centers)
    ctr_xyz: jax.Array = xyz[jnp.newaxis, :] - centers2d  # (N, 3)
    ctr_r = jnp.linalg.norm(ctr_xyz, axis=1)  # (N,1)

    # value
    xyz_ijk: jax.Array = jnp.prod(ctr_xyz**ijk, axis=1)  # (N)
    gauss: jax.Array = jnp.exp(-expts * ctr_r[:, jnp.newaxis] ** 2)  # (N,)
    value: jax.Array = coeffs * gauss * xyz_ijk[:, jnp.newaxis]

    # Gradient
    grad: jax.Array = value[:, :, jnp.newaxis] * (
        ijk[:, jnp.newaxis, :] / ctr_xyz[:, jnp.newaxis, :]
        - 2 * expts[:, :, jnp.newaxis] * ctr_xyz[:, jnp.newaxis, :]
    )  # (N,3)
    return jnp.concatenate(
        [jnp.sum(value, axis=1)[:, jnp.newaxis], jnp.sum(grad, axis=1)], axis=1
    )  # (N,4)


def _cartesian_gto_vgl(
    centers: jax.Array,
    ijk: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
    xyz: jax.Array,
) -> jax.Array:
    """Evaluate a Cartesian Gaussian-type orbital.

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
    """
    # Distances
    centers2d: jax.Array = jnp.atleast_2d(centers)
    ctr_xyz: jax.Array = xyz[jnp.newaxis, :] - centers2d  # (N, 3)
    ctr_r = jnp.linalg.norm(ctr_xyz, axis=1)  # (N,1)

    # value
    xyz_ijk: jax.Array = jnp.prod(ctr_xyz**ijk, axis=1)  # (N)
    gauss: jax.Array = jnp.exp(-expts * ctr_r[:, jnp.newaxis] ** 2)  # (N,)
    value: jax.Array = coeffs * gauss * xyz_ijk[:, jnp.newaxis]

    # Gradient dimensions are basis, coeff, xyz
    gpart: jax.Array = (
        ijk[:, jnp.newaxis, :] / ctr_xyz[:, jnp.newaxis, :]
        - 2 * expts[:, :, jnp.newaxis] * ctr_xyz[:, jnp.newaxis, :]
    )
    grad: jax.Array = value[:, :, jnp.newaxis] * gpart  # (N,3)
    # print("gpart", gpart.shape)
    lap_part = (
        gpart**2
        - ijk[:, jnp.newaxis, :] / ctr_xyz[:, jnp.newaxis, :] ** 2
        - 2 * expts[:, :, jnp.newaxis]
    )
    # print("lap_part", lap_part.shape)
    laplacian: jax.Array = value * jnp.sum(
        lap_part, axis=2
    )  # value is (N, M), lap_part is (N, M, 3)
    # print(laplacian.shape)
    return jnp.concatenate(
        [
            jnp.sum(value, axis=1)[:, jnp.newaxis],
            jnp.sum(grad, axis=1),
            jnp.sum(laplacian, axis=1)[:, jnp.newaxis],
        ],
        axis=1,
    )  # (N,4)


def create_gto_evaluator(mol, nimages=1):
    """
    Create a JAX evaluator for a Gaussian type orbital
    Args:
        mol: pyscf molecule
        nimages: number of images to evaluate the basis functions at (relevant for PBC)
    Returns:
        evaluator: function that evaluates the basis functions
        gradient: function that evaluates the gradient of the basis functions 
        vgl: function that evaluates the value, gradient and laplacian of the basis functions

    Note that the PBC implementation only works for uncontracted basis functions for now.
    """
    centers = mol.atom_coords()
    centers = jnp.array(centers)
    natom = mol.natm
    atom_names = [mol.atom_pure_symbol(i) for i in range(natom)]
    centers_aos = []
    ijks = []
    expts = []
    coeffs = []
    norms = [
        np.sqrt(1.0 / (4 * np.pi)),
        np.sqrt(1.0 / (4.0 * np.pi)),
        np.sqrt(1.0 / 15),
        np.sqrt(1.0 / 105),
        np.sqrt(1.0 / 945),
    ]
    for nm, atom in zip(atom_names, centers):
        basis = mol._basis[nm]
        for b in basis:
            ell = b[0]
            exp_coeffs = np.asarray(b[1:])
            for ijk in angular_momentum_xyz(ell):
                centers_aos.append(atom)
                ijks.append(ijk)
                expts.append(exp_coeffs[:, 0])
                ijk_sum = sum(ijk)  # (1,)
                norm = jnp.sqrt(
                    2
                    * (4 * exp_coeffs[:, 0]) ** (ijk_sum + 1)
                    * jnp.sqrt(2 * exp_coeffs[:, 0] / jnp.pi)
                )  # (N,1)
                print("norm", norm*norms[ell], norms[ell])

                coeffs.append(exp_coeffs[:, 1] * norms[ell] * norm)

    max_len: int = max(len(e) for e in expts)
    for ii in range(len(expts)):
        expts[ii] = jnp.pad(expts[ii], (0, max_len - len(expts[ii])))
        coeffs[ii] = jnp.pad(coeffs[ii], (0, max_len - len(coeffs[ii])))

    centers_aos = jnp.array(centers_aos)
    ijks = jnp.array(ijks)
    expts = jnp.array(expts)
    coeffs = jnp.array(coeffs)
    if hasattr(mol, "lattice_vectors"):
        if expts.shape[1] > 1:
          raise ValueError("Only unc basis is supported for PBC systems, at least for now. unc is faster with current implementation anyway.")

        nlat = []
        for i, j, k in itertools.product(range(-nimages, nimages+1), repeat=3):
            nlat.append([i, j, k])
        images = jnp.array(nlat)@jnp.array(mol.lattice_vectors()) 
        print("images", images)
    else:
        images = jnp.array([[0, 0, 0]])

    evaluator = partial(
        _cartesian_gto,
        centers_aos,
        ijks,
        expts,
        coeffs,
        images,
    )

    gradient = partial(
        _cartesian_gto_grad,
        centers_aos,
        ijks,
        expts,
        coeffs,
    )

    vgl = partial(
        _cartesian_gto_vgl,
        centers_aos,
        ijks,
        expts,
        coeffs,
    )
    return evaluator, gradient, vgl


def test_laplacian():
    cpu = True
    if cpu:
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", True)
    else:
        pass

    import pyscf

    mol = pyscf.gto.Mole(
        atom="""O 0 0 0; H  0 2.0 0; H 0 0 2.0""",
        basis="unc-ccecp-ccpvdz",
        # basis='sto-3g',
        ecp="ccecp",
        cart=True,
    )
    mol.build()

    import time

    evaluator, gradient, vgl = create_gto_evaluator(mol)
    evaluator_val = jax.vmap(evaluator, in_axes=(0))
    evaluator_gradient = jax.jacfwd(evaluator)
    ad_laplacian = jax.jacfwd(evaluator_gradient)
    evaluator_gradient = jax.vmap(evaluator_gradient, in_axes=(0))
    evaluator_gradient = jax.jit(evaluator_gradient)

    ad_laplacian = jax.vmap(ad_laplacian, in_axes=(0))
    ad_laplacian = jax.jit(ad_laplacian)

    analytic_gradient = jax.vmap(gradient, in_axes=(0))
    analytic_gradient = jax.jit(analytic_gradient)

    vgl = jax.vmap(vgl, in_axes=(0))
    vgl = jax.jit(vgl)

    nconfig = 200
    seed = 32123
    key = jax.random.key(seed)
    coords = jax.random.normal(key, (nconfig, 3))

    res = evaluator_val(coords)
    jax.block_until_ready(res)

    start = time.perf_counter()
    res_val = evaluator_val(coords)
    jax.block_until_ready(res_val)
    end = time.perf_counter()
    val_time = end - start
    print("Time taken for value: ", end - start)

    res = evaluator_gradient(coords)
    jax.block_until_ready(res)
    start = time.perf_counter()
    res = evaluator_gradient(coords)
    jax.block_until_ready(res)
    end = time.perf_counter()
    grad_time = end - start
    print("Time taken for gradient: ", end - start)

    analytic_res = analytic_gradient(coords)
    jax.block_until_ready(analytic_res)
    start = time.perf_counter()
    analytic_res = analytic_gradient(coords)
    jax.block_until_ready(analytic_res)
    end = time.perf_counter()
    analytic_grad_time = end - start
    print(
        "gradient squared err", jnp.mean(jnp.linalg.norm(res - analytic_res[:, :, 1:]))
    )
    print("Time taken for analytic gradient: ", end - start)

    print(
        "rms errors for value",
        jnp.mean(jnp.linalg.norm(res_val - analytic_res[:, :, 0])),
    )
    print("grad ratio for AD", grad_time / val_time)
    print("grad ratio for analytic", analytic_grad_time / val_time)

    res = ad_laplacian(coords)
    jax.block_until_ready(res)
    start = time.perf_counter()
    res = ad_laplacian(coords)
    res = jnp.trace(res, axis1=2, axis2=3)
    jax.block_until_ready(res)
    print("laplacian ", res.shape)
    end = time.perf_counter()
    grad_time = end - start
    print("Time taken for AD laplacian: ", end - start)

    analytic_res = vgl(coords)
    jax.block_until_ready(analytic_res)
    start = time.perf_counter()
    analytic_res = vgl(coords)
    jax.block_until_ready(analytic_res)
    end = time.perf_counter()
    analytic_grad_time = end - start
    print("Time taken for analytic laplacian: ", end - start)
    # print('AD laplacian', res)
    # print('analytic laplacian', analytic_res[:,:, 4])

    print(
        "laplacian squared err", jnp.mean(jnp.linalg.norm(res - analytic_res[:, :, 4]))
    )


def test_pbc():
    """
    Test the periodic boundary conditions
    """
    import pyqmc.api as pyq
    cpu = True
    if cpu:
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", True)
    else:
        pass

    import pyscf.pbc.gto
    cell = pyscf.pbc.gto.Cell()

    cell.verbose = 5
    cell.atom = [
        ["C", np.array([0.0, 0.0, 0.0])],
        ["C", np.array([0.8917, 0.8917, 0.8917])],
    ]
    cell.a = [[0.0, 1.7834, 1.7834], [1.7834, 0.0, 1.7834], [1.7834, 1.7834, 0.0]]
    cell.basis = "unc-ccecp-ccpvtz"
    cell.ecp = "ccecp"
    cell.exp_to_discard = 0.3
    cell.cart = True
    cell.build()

    evaluator, gradient, vgl = create_gto_evaluator(cell, nimages=2)
    evaluator_val = jax.vmap(evaluator, in_axes=(0))

    nconfig = 1
    coords = pyq.initial_guess(cell, nconfig)
    #coords = np.array([[0.0, 0.0, 0.0]])
    #coords = np.array([[0.8917, 0.8917, 0.8917]])
    jax_val = evaluator_val(jnp.array(coords.configs[:,0,:]))
    print('jax values', jax_val)
    pyscf_val = cell.eval_gto("PBCGTOval_cart", coords.configs[:,0,:])
    print('pyscf values', pyscf_val)
    print("errors", jax_val-pyscf_val)
    print("MAD", jnp.mean(jnp.abs(jax_val-pyscf_val)))

if __name__ == "__main__":
    test_pbc()
