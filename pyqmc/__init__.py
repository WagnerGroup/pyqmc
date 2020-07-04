name = "pyqmc"
from pyqmc.mc import vmc, initial_guess
from pyqmc.slater import PySCFSlater
from pyqmc.multislater import MultiSlater

from pyqmc.multiplywf import MultiplyWF
from pyqmc.jastrowspin import JastrowSpin
from pyqmc.manybody_jastrow import J3
from pyqmc.supercell import get_supercell

from pyqmc.accumulators import EnergyAccumulator, PGradTransform, LinearTransform
from pyqmc.func3d import (
    PolyPadeFunction,
    PadeFunction,
    GaussianFunction,
    CutoffCuspFunction,
)
from pyqmc.optvariance import optvariance
from pyqmc.linemin import line_minimization
from pyqmc.optimize_ortho import optimize_orthogonal
from pyqmc.dmc import rundmc
from pyqmc.reblock import reblock as avg_reblock
import numpy as np
import h5py


def slater_jastrow(mol, mf, abasis=None, bbasis=None):
    raise NotImplementedError(
        "slater_jastrow() is no longer supported. Please use default_sj instead."
    )


def gradient_generator(mol, wf, to_opt=None, **ewald_kwargs):
    return PGradTransform(
        EnergyAccumulator(mol, **ewald_kwargs), LinearTransform(wf.parameters, to_opt)
    )


def default_slater(mol, mf, optimize_orbitals=False):

    wf = PySCFSlater(mol, mf)
    to_opt = {}
    if optimize_orbitals:
        for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
            to_opt[k] = np.ones(wf.parameters[k].shape).astype(bool)
    return wf, to_opt


def default_multislater(mol, mf, mc, tol=None, optimize_orbitals=False):
    import numpy as np

    wf = MultiSlater(mol, mf, mc, tol)
    to_opt = ["det_coeff"]
    to_opt = {"det_coeff": np.ones(wf.parameters["det_coeff"].shape).astype(bool)}
    to_opt["det_coeff"][0] = False  # Determinant coefficient pivot
    if optimize_orbitals:
        for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
            to_opt[k] = np.ones(wf.parameters[k].shape).astype(bool)

    return wf, to_opt


def default_jastrow(mol, ion_cusp=False, na=4, nb=3, rcut=7.5):
    """         
    Default 2-body jastrow from qwalk,
    Args:
      ion_cusp (bool): add an extra term to satisfy electron-ion cusp.
    Returns:
      jastrow, to_opt
    """
    import numpy as np

    def expand_beta_qwalk(beta0, n):
        """polypade expansion coefficients 
        for n basis functions with first 
        coeff beta0"""
        beta = np.zeros(n)
        beta[0] = beta0
        beta1 = np.log(beta0 + 1.00001)
        for i in range(1, n):
            beta[i] = np.exp(beta1 + 1.6 * i) - 1
        return beta

    beta_abasis = expand_beta_qwalk(0.2, na)
    beta_bbasis = expand_beta_qwalk(0.5, nb)
    if ion_cusp:
        abasis = [CutoffCuspFunction(gamma=24, rcut=rcut)]
    else:
        abasis = []
    abasis += [PolyPadeFunction(beta=beta_abasis[i], rcut=rcut) for i in range(4)]
    bbasis = [CutoffCuspFunction(gamma=24, rcut=rcut)]
    bbasis += [PolyPadeFunction(beta=beta_bbasis[i], rcut=rcut) for i in range(3)]

    jastrow = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    if ion_cusp:
        jastrow.parameters["acoeff"][:, 0, :] = mol.atom_charges()[:, None]
    jastrow.parameters["bcoeff"][0, [0, 1, 2]] = np.array([-0.25, -0.50, -0.25])

    to_opt = {}
    to_opt["acoeff"] = np.ones(jastrow.parameters["acoeff"].shape).astype(bool)
    if ion_cusp:
        to_opt["acoeff"][:, 0, :] = False  # Cusp conditions
    to_opt["bcoeff"] = np.ones(jastrow.parameters["bcoeff"].shape).astype(bool)
    to_opt["bcoeff"][0, [0, 1, 2]] = False  # Cusp conditions
    return jastrow, to_opt


def default_msj(mol, mf, mc, tol=None, freeze_orb=None, ion_cusp=False):
    wf1, to_opt1 = default_multislater(mol, mf, mc, tol, freeze_orb)
    wf2, to_opt2 = default_jastrow(mol, ion_cusp)
    wf = MultiplyWF(wf1, wf2)
    to_opt = {"wf1" + x: opt for x, opt in to_opt1.items()}
    to_opt.update({"wf2" + x: opt for x, opt in to_opt2.items()})

    return wf, to_opt


def default_sj(mol, mf, ion_cusp=False):
    wf1, to_opt1 = default_slater(mol, mf)
    wf2, to_opt2 = default_jastrow(mol, ion_cusp)
    wf = MultiplyWF(wf1, wf2)
    to_opt = {"wf1" + x: opt for x, opt in to_opt1.items()}
    to_opt.update({"wf2" + x: opt for x, opt in to_opt2.items()})

    return wf, to_opt


def generate_wf(
    mol, mf, jastrow=default_jastrow, jastrow_kws=None, slater_kws=None, mc=None
):
    """
    mol and mf are pyscf objects

    jastrow may be either a function that returns wf, to_opt, or 
    a list of such functions.

    jastrow_kws is a dictionary of keyword arguments for the jastrow function, or
    a list of those functions.
    """
    if jastrow_kws is None:
        jastrow_kws = {}

    if slater_kws is None:
        slater_kws = {}

    if not isinstance(jastrow, list):
        jastrow = [jastrow]
        jastrow_kws = [jastrow_kws]

    if mc is None:
        wf1, to_opt1 = default_slater(mol, mf, **slater_kws)
    elif hasattr(mol, "a"):
        raise NotImplementedError("No defaults for multislater with PBCs")
    else:
        wf1, to_opt1 = default_multislater(mol, mf, mc, **slater_kws)

    pack = [jast(mol, **kw) for jast, kw in zip(jastrow, jastrow_kws)]
    wfs = [p[0] for p in pack]
    to_opts = [p[1] for p in pack]
    wf = MultiplyWF(wf1, *wfs)
    to_opt = {"wf1" + k: v for k, v in to_opt1.items()}
    for i, to_opt2 in enumerate(to_opts):
        to_opt.update({f"wf{i+2}" + k: v for k, v in to_opt2.items()})
    return wf, to_opt


def recover_pyscf(chkfile):
    import pyscf

    mol = pyscf.lib.chkfile.load_mol(chkfile)
    mol.output = None
    mol.stdout = None
    if hasattr(mol, "a"):
        from pyscf import pbc

        mol = pbc.gto.cell.loads(pyscf.lib.chkfile.load(chkfile, "mol"))
        mf = pbc.scf.KRHF(mol)
    # It actually doesn't matter what type of object we make it for
    # pyqmc. Now if you try to run this, it might cause issues.
    else:
        mf = pyscf.scf.RHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))
    return mol, mf


def read_wf(wf, wf_file):
    with h5py.File(wf_file, "r") as hdf:
        if "wf" in hdf.keys():
            grp = hdf["wf"]
            for k in grp.keys():
                wf.parameters[k] = np.array(grp[k])
