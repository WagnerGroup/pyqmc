# MIT License
# 
# Copyright (c) 2019 Lucas K Wagner
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

from pyscf import mcscf, fci, lib 
from pyscf import gto, scf, tools
import pyqmc.api as pyq 
import os
import h5py


def run_scf(scf_checkfile):
    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 2.", basis=f'ccecpccpvdz', unit='bohr')
    mf = scf.RHF(mol)
    mf.chkfile = scf_checkfile
    dm = mf.init_guess_by_atom()
    mf.kernel(dm)


def run_casscf(scf_checkfile, ci_checkfile):
    cell, mf = pyq.recover_pyscf(scf_checkfile, cancel_outputs=False)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.chkfile = ci_checkfile
    mc.kernel()
    
    with h5py.File(mc.chkfile, "a") as f:
        f["mcscf/nelecas"] = list(mc.nelecas)
        f["mcscf/ci"] = mc.ci
    return mc


def run_casci(scf_checkfile, ci_checkfile):
    cell, mf = pyq.recover_pyscf(scf_checkfile, cancel_outputs=False)
    mc = mcscf.CASCI(mf, 2, 2)
    mc.kernel()
    
    print(mc.__dict__.keys())
    with h5py.File(ci_checkfile, "a") as f:
        f.create_group("ci")
        f["ci/ncas"] = mc.ncas
        f["ci/nelecas"] = list(mc.nelecas)
        f["ci/ci"] = mc.ci
        f["ci/mo_coeff"] = mc.mo_coeff
    return mc


def make_wf_object(scf_checkfile, ci_checkfile):
    mol, mf, mc = pyq.recover_pyscf(scf_checkfile, ci_checkfile=ci_checkfile)
    wf, _ = pyq.generate_wf(mol, mf, mc=mc)
    return wf


if __name__ == "__main__":
    scf_checkfile = "scf.chk"
    ci_checkfile = "ci.chk"
    run_scf(scf_checkfile)
    run_casscf(scf_checkfile, ci_checkfile) # or can use run_casci
    pyq.OPTIMIZE(scf_checkfile, "optimize.chk", ci_checkfile=ci_checkfile, max_iterations=1, verbose=True)
