import pyqmc.slater as slater
import pyqmc.multiplywf as multiplywf
import pyqmc.addwf as addwf
import pyqmc.jastrowspin as jastrowspin
import pyqmc.func3d as func3d
import pyqmc.gpu as gpu
import numpy as np
import h5py


def generate_slater(
    mol,
    mf,
    optimize_determinants=False,
    optimize_orbitals=False,
    optimize_zeros=True,
    epsilon=1e-8,
    **kwargs,
):
    """Construct a Slater determinant

    :parameter boolean optimize_orbitals: make `to_opt` true for orbital parameters
    :parameter array-like twist: The twist to extract from the mean-field object
    :parameter boolean optimize_zeros: optimize coefficients that are zero in the mean-field object
    :returns: slater, to_opt
    """
    wf = slater.Slater(mol, mf, **kwargs)
    to_opt = {}
    to_opt["det_coeff"] = np.zeros_like(wf.parameters["det_coeff"], dtype=bool)
    if optimize_determinants:
        to_opt["det_coeff"] = np.ones_like(wf.parameters["det_coeff"], dtype=bool)
        to_opt["det_coeff"][np.argmax(wf.parameters["det_coeff"])] = False
    if optimize_orbitals:
        for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
            to_opt[k] = np.ones(wf.parameters[k].shape, dtype=bool)
            if not optimize_zeros:
                to_opt[k][np.abs(gpu.asnumpy(wf.parameters[k])) < epsilon] = False

    return wf, to_opt


def expand_beta_qwalk(beta0, n):
    """polypade expansion coefficients for n basis functions with first coeff beta0"""
    if n == 0:
        return np.zeros(0)
    beta = np.zeros(n)
    beta[0] = beta0
    beta1 = np.log(beta0 + 1.00001)
    for i in range(1, n):
        beta[i] = np.exp(beta1 + 1.6 * i) - 1
    return beta


def default_jastrow_basis(mol, ion_cusp=False, na=4, nb=3, rcut=None):
    if rcut is None:
        if hasattr(mol, "a"):
            rcut = np.amin(np.pi / np.linalg.norm(mol.reciprocal_vectors(), axis=1))
        else:
            rcut = 7.5

    beta_abasis = expand_beta_qwalk(0.2, na)
    beta_bbasis = expand_beta_qwalk(0.5, nb)
    if ion_cusp:
        abasis = [func3d.CutoffCuspFunction(gamma=24, rcut=rcut)]
    else:
        abasis = []
    abasis += [func3d.PolyPadeFunction(beta=ba, rcut=rcut) for ba in beta_abasis]
    bbasis = [func3d.CutoffCuspFunction(gamma=24, rcut=rcut)]
    bbasis += [func3d.PolyPadeFunction(beta=bb, rcut=rcut) for bb in beta_bbasis]
    return abasis, bbasis


def generate_jastrow(mol, ion_cusp=None, na=4, nb=3, rcut=None):
    """
    Default 2-body jastrow from QWalk,

    :parameter boolean ion_cusp: add an extra term to satisfy electron-ion cusp.
    :returns: jastrow, to_opt
    """
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
    else:
        assert isinstance(ion_cusp, list)

    abasis, bbasis = default_jastrow_basis(mol, len(ion_cusp) > 0, na, nb, rcut)
    jastrow = jastrowspin.JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    if len(ion_cusp) > 0:
        coefs = mol.atom_charges().copy()
        coefs[[l[0] not in ion_cusp for l in mol._atom]] = 0.0
        jastrow.parameters["acoeff"][:, 0, :] = gpu.cp.asarray(coefs[:, None])
    jastrow.parameters["bcoeff"][0, [0, 1, 2]] = gpu.cp.array([-0.25, -0.50, -0.25])

    to_opt = {"acoeff": np.ones(jastrow.parameters["acoeff"].shape).astype(bool)}
    if len(ion_cusp) > 0:
        to_opt["acoeff"][:, 0, :] = False  # Cusp conditions
    to_opt["bcoeff"] = np.ones(jastrow.parameters["bcoeff"].shape).astype(bool)
    to_opt["bcoeff"][0, [0, 1, 2]] = False  # Cusp conditions
    return jastrow, to_opt


def generate_sj(mol, mf, optimize_orbitals=False, twist=None, **jastrow_kws):
    wf1, to_opt1 = generate_slater(mol, mf, optimize_orbitals, twist)
    wf2, to_opt2 = generate_jastrow(mol, **jastrow_kws)
    wf = multiplywf.MultiplyWF(wf1, wf2)
    to_opt = {"wf1" + x: opt for x, opt in to_opt1.items()}
    to_opt.update({"wf2" + x: opt for x, opt in to_opt2.items()})

    return wf, to_opt


def generate_wf(
    mol, mf, jastrow=generate_jastrow, jastrow_kws=None, slater_kws=None, mc=None
):
    """
    Generate a wave function from pyscf objects.

    :param mol: The molecule or cell
    :type mol: pyscf Mole or Cell
    :param mf: a pyscf mean-field object
    :type mf: Any mean-field object that Slater can read
    :param jastrow: a function that returns wf, to_opt, or a list of such functions.
    :param jastrow_kws: a dictionary of keyword arguments for the jastrow function, or a list of those functions.
    :param slater_kws: a dictionary of keyword arguments for the generate_slater function
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

    wf1, to_opt1 = generate_slater(mol, mf, mc=mc, **slater_kws)

    pack = [jast(mol, **kw) for jast, kw in zip(jastrow, jastrow_kws)]
    wfs = [p[0] for p in pack]
    to_opts = [p[1] for p in pack]
    wf = multiplywf.MultiplyWF(wf1, *wfs)
    to_opt = {"wf1" + k: v for k, v in to_opt1.items()}
    for i, to_opt2 in enumerate(to_opts):
        to_opt.update({f"wf{i+2}" + k: v for k, v in to_opt2.items()})
    return wf, to_opt


def read_wf(wf, wf_file):
    """Read the wave function parameters from wf_file into wf.

    Typical usage:

    .. code-block:: python

       linemin(wf, coords, ..., hdf_file="linemin.hdf5")
       read_wf(wf, "linemin.hdf5")

    :param wf: object to load saved parameters into
    :type wf: wave function object with parameters dictionary
    :param wf_file: A HDF5 file with "wf" key. The parameters in this file will be read into the wave function in-place
    :type wf_file: string

    :return: nothing"""

    with h5py.File(wf_file, "r") as hdf:
        if "wf" not in hdf.keys():
            raise Exception("Did not find wf in hdf file")
        grp = hdf["wf"]
        for k in grp.keys():
            new_parms = gpu.cp.array(grp[k])
            if wf.parameters[k].shape != new_parms.shape:
                raise Exception(
                    f"For wave function parameter {k}, shape in {wf_file} is {new_parms.shape}, while current shape is {wf.parameters[k].shape}"
                )
            wf.parameters[k] = new_parms
    return wf


def read_superposition(mol, mf, wf_files, coeffs, mc=None):
    """Generate a wf that is a linear superposition of the given wfs with the given coefficients

    Typical usage:

    .. code-block:: python

       wf_coeffs = 1/np.sqrt(2)*np.ones(2)
       wf_files = ["wf1.chk", "wf2.chk"]
       wf, to_opt = pyqmc.wftools.generate_superposewf(mol, mf, wf_files, wf_coeffs, mc)

    :param mol: The molecule or cell
    :type mol: pyscf Mole or Cell
    :param mf: a pyscf mean-field object
    :type mf: Any mean-field object that Slater can read
    :param mc: A CAS object (optional) for multideterminant wave functions.
    :param wf_files: A list of HDF5 files with "wf" key. The parameters in this file will be read into the wave function in-place
    :param coefs: A list of superposition coefficients
    :type coefs: list of complex
    :type wf_files: list of string
    :rtype: A superposition of (multi) Slater-Jastrow wave functions object
    :return: to_opt"""

    wfs = []
    to_opt = {}
    for iwf, wf_file in enumerate(wf_files):
        wf_tmp, to_opt_tmp = generate_wf(mol, mf, mc=mc)
        wf_tmp = read_wf(wf_tmp, wf_file)
        wfs.append(wf_tmp)
        for k, v in to_opt_tmp.items():
            to_opt[f"wf{iwf}" + k] = v
    wf = addwf.AddWF(coeffs, wfs)
    return wf, to_opt
