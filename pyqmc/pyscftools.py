import pyscf
import pyscf.pbc
import pyscf.mcscf
import h5py
import json


def recover_pyscf(chkfile, ci_checkfile=None, cancel_outputs=True):
    """Generate pyscf objects from a pyscf checkfile, in a way that is easy to use for pyqmc. The chkfile should be saved by setting mf.chkfile in a pyscf SCF object.

    It is recommended to write and recover the objects, rather than trying to use pyscf objects directly when dask parallelization is being used, since by default the pyscf objects contain unserializable objects. (this may be changed in the future)

    `cancel_outputs` will set the outputs of the objects to None. You may need to set `cancel_outputs=False` if you are using this to input to other pyscf functions.

    Typical usage::

        mol, mf = recover_pyscf("dft.hdf5")

    :param chkfile: The filename to read from.
    :type chkfile: string
    :return: mol, mf
    :rtype: pyscf Mole, SCF objects"""

    with h5py.File(chkfile, "r") as f:
        periodic = "a" in json.loads(f["mol"][()]).keys()

    if not periodic:
        mol = pyscf.lib.chkfile.load_mol(chkfile)
        with h5py.File(chkfile, "r") as f:
            mo_occ_shape = f["scf/mo_occ"].shape
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if len(mo_occ_shape) == 2:
            mf = pyscf.scf.UHF(mol)
        elif len(mo_occ_shape) == 1:
            mf = pyscf.scf.ROHF(mol) if mol.spin != 0 else pyscf.scf.RHF(mol)
        else:
            raise Exception("Couldn't determine type from chkfile")
    else:

        mol = pyscf.pbc.lib.chkfile.load_cell(chkfile)
        with h5py.File(chkfile, "r") as f:
            has_kpts = "mo_occ__from_list__" in f["/scf"].keys()
            if has_kpts:
                rhf = "000000" in f["/scf/mo_occ__from_list__/"].keys()
            else:
                rhf = len(f["/scf/mo_occ"].shape) == 1
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if not rhf and has_kpts:
            mf = pyscf.pbc.scf.KUHF(mol)
        elif has_kpts:
            mf = pyscf.pbc.scf.KROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.KRHF(mol)
        elif rhf:
            mf = pyscf.pbc.scf.ROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.RHF(mol)
        else:
            mf = pyscf.pbc.scf.UHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))

    if ci_checkfile is not None:
        casdict = pyscf.lib.chkfile.load(ci_checkfile, "ci")
        with h5py.File(ci_checkfile, "r") as f:
            hci = "ci/_strs" in f.keys()
        if hci:
            mc = pyscf.hci.SCI(mol)
        else:
            if len(casdict["mo_coeff"].shape) == 3:
                mc = pyscf.mcscf.UCASCI(mol, casdict["ncas"], casdict["nelecas"])
            else:
                mc = pyscf.mcscf.CASCI(mol, casdict["ncas"], casdict["nelecas"])
        mc.__dict__.update(casdict)

        return mol, mf, mc
    return mol, mf
