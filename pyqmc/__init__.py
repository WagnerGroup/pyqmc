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


def default_slater(mol, mf, optimize_orbitals=False, twist=None):

    wf = PySCFSlater(mol, mf, twist=twist)
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


def default_jastrow(mol, ion_cusp=None, na=4, nb=3, rcut=None):
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
        if n == 0:
            return np.zeros(0)
        beta = np.zeros(n)
        beta[0] = beta0
        beta1 = np.log(beta0 + 1.00001)
        for i in range(1, n):
            beta[i] = np.exp(beta1 + 1.6 * i) - 1
        return beta

    if rcut is None:
        if hasattr(mol, "a"):
            rcut = np.amin(np.pi / np.linalg.norm(mol.reciprocal_vectors(), axis=1))
        else:
            rcut = 7.5

    beta_abasis = expand_beta_qwalk(0.2, na)
    beta_bbasis = expand_beta_qwalk(0.5, nb)
    if ion_cusp == False:
        ion_cusp = []
        if not mol.has_ecp():
            print("Warning: using neither ECP nor ion_cusp")
    elif ion_cusp == True:
        ion_cusp = list(mol._basis.keys())
        if mol.has_ecp():
            print("Warning: using both ECP and ion_cusp")
    elif ion_cusp is None:
        ion_cusp = [l for l in mol._basis.keys() if l not in mol._ecp.keys()]
        print("default ion_cusp:", ion_cusp)
    else:
        assert isinstance(ion_cusp, list)

    if len(ion_cusp) > 0:
        abasis = [CutoffCuspFunction(gamma=24, rcut=rcut)]
    else:
        abasis = []
    abasis += [PolyPadeFunction(beta=ba, rcut=rcut) for ba in beta_abasis]
    bbasis = [CutoffCuspFunction(gamma=24, rcut=rcut)]
    bbasis += [PolyPadeFunction(beta=bb, rcut=rcut) for bb in beta_bbasis]

    jastrow = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    if len(ion_cusp) > 0:
        coefs = mol.atom_charges().copy()
        coefs[[l[0] not in ion_cusp for l in mol._atom]] = 0.0
        jastrow.parameters["acoeff"][:, 0, :] = coefs[:, None]
    jastrow.parameters["bcoeff"][0, [0, 1, 2]] = np.array([-0.25, -0.50, -0.25])

    to_opt = {}
    to_opt["acoeff"] = np.ones(jastrow.parameters["acoeff"].shape).astype(bool)
    if len(ion_cusp) > 0:
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


def default_sj(mol, mf, optimize_orbitals=False, twist=None, **jastrow_kws):
    wf1, to_opt1 = default_slater(mol, mf, optimize_orbitals, twist)
    wf2, to_opt2 = default_jastrow(mol, **jastrow_kws)
    wf = MultiplyWF(wf1, wf2)
    to_opt = {"wf1" + x: opt for x, opt in to_opt1.items()}
    to_opt.update({"wf2" + x: opt for x, opt in to_opt2.items()})

    return wf, to_opt


def generate_wf(
    mol, mf, jastrow=default_jastrow, jastrow_kws=None, slater_kws=None, mc=None
):
    """
    Generate a wave function from pyscf objects. 

    :param mol: The molecule or cell
    :type mol: pyscf Mole or Cell
    :param mf: a pyscf mean-field object
    :type mf: Any mean-field object that PySCFSlater can read
    :param jastrow: a function that returns wf, to_opt, or a list of such functions.
    :param jastrow_kws: a dictionary of keyword arguments for the jastrow function, or a list of those functions.
    :param slater_kws: a dictionary of keyword arguments for the default_slater function
    :param mc: A CAS object (optional) for multideterminant wave functions.

    :return: wf
    :rtype: A (multi) Slater-Jastrow wave function object
    :return: to_opt
    :rtype: A dictionary of parameters to optimize, given the settings. 
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


def recover_pyscf(chkfile, cancel_outputs=True):
    """Generate pyscf objects from a pyscf checkfile, in a way that is easy to use for pyqmc. The chkfile should be saved by setting mf.chkfile in a pyscf SCF object. 
    
It is recommended to write and recover the objects, rather than trying to use pyscf objects directly when dask parallelization is being used, since by default the pyscf objects 

cancel_outputs will set the outputs of the objects to None. You may need to make cancel_outputs False if you are using this to input to other pyscf functions.

Typical usage:

mol, mf = recover_pyscf("dft.hdf5")

:param chkfile: The filename to read from. 
:type chkfile: string
:return: mol, mf
:rtype: pyscf Mole, SCF objects
"""

    import pyscf

    mol = pyscf.lib.chkfile.load_mol(chkfile)
    if cancel_outputs:
        mol.output = None
        mol.stdout = None

    if hasattr(mol, "a"):
        from pyscf import pbc

        mol = pbc.lib.chkfile.load_cell(chkfile)
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        mf = pbc.scf.KRHF(mol)
    else:
        # It actually doesn't matter what type of object we make it for
        # pyqmc. Now if you try to run this, it might cause issues.
        mf = pyscf.scf.RHF(mol)

    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))
    return mol, mf


def read_wf(wf, wf_file):
    """Read the wave function parameters from wf_file into wf. 

Typical usage:

.. code-block::python

linemin(wf, coords, ..., hdf_file="linemin.hdf5")
read_wf(wf, "linemin.hdf5")

:param wf: A pyqmc wave function object. This will 
:type wf: wave function object with parameters dictionary
:param wf_file: A HDF5 file with "wf" key. The parameters in this file will be read into the wave function in-place
:type wf_file: string

:return: nothing
"""

    with h5py.File(wf_file, "r") as hdf:
        if "wf" in hdf.keys():
            grp = hdf["wf"]
            for k in grp.keys():
                wf.parameters[k] = np.array(grp[k])
