# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""
Recipe for converting spherical contracted basis to Cartesian uncontracted basis.
Cartesian basis is required when using PyQMC with JAX.
For periodic systems using JAX, the basis must also be uncontracted.
"""

from pyscf.scf.addons import project_mo_nr2nr
from pyscf import gto, scf, mcscf
import pyqmc.api as pyq


def sph_to_cart(mol, mf, mc=None):
    assert mol.cart == False
    mol_cart = mol.copy()
    mol_cart.build(cart=True)
    
    mf_cart = mf.copy()
    mf_cart.mo_coeff = project_mo_nr2nr(mol, mf.mo_coeff, mol_cart)

    if mc is not None:
        mc_cart = mc.copy()
        mc_cart.mo_coeff = project_mo_nr2nr(mol, mc.mo_coeff, mol_cart)
        return mol_cart, mf_cart, mc_cart
    else:
        return mol_cart, mf_cart


def con_to_unc(mol, mf, mc=None):
    mol_unc = mol.copy()
    mol_unc.build(basis="unc-" + mol.basis)
    
    mf_unc = mf.copy()
    mf_unc.mo_coeff = project_mo_nr2nr(mol, mf.mo_coeff, mol_unc)

    if mc is not None:
        mc_unc = mc.copy()
        mc_unc.mo_coeff = project_mo_nr2nr(mol, mc.mo_coeff, mol_unc)
        return mol_unc, mf_unc, mc_unc
    else:
        return mol_unc, mf_unc


def run_mf():
    mol = gto.Mole()
    mol.atom = \
    """
    O  0.000  0.000  0.000
    H  0.000  0.757  0.586
    H  0.000 -0.757  0.586
    """
    mol.basis = "ccecp-ccpvdz"
    mol.ecp = "ccecp"
    mol.verbose = 4
    mol.build()

    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mol, mf


def run_casscf(mf):
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.verbose = 4
    mc.kernel()
    return mc


def run_optimization(mol, mf, mc):
    slater_kws = {
        "optimize_orbitals": False, 
        "optimize_zeros": True, 
        "optimize_determinants": True
    }
    wf, to_opt = pyq.generate_wf(mol, mf, mc=mc, slater_kws=slater_kws, jax=True)
    configs = pyq.initial_guess(mol, nconfig=1000)
    acc = pyq.gradient_generator(mol, wf, to_opt, use_old_ecp=False)

    pyq.line_minimization(
        wf, 
        configs, 
        acc, 
        hdf_file="h2o_sj.hdf5",
        max_iterations=10
    )


if __name__ == "__main__":
    mol, mf = run_mf()
    mc = run_casscf(mf)

    # Convert spherical contracted basis to Cartesian uncontracted basis
    mol, mf, mc = sph_to_cart(mol, mf, mc)
    mol, mf, mc = con_to_unc(mol, mf, mc)

    # Run optimization with JAX
    run_optimization(mol, mf, mc)
