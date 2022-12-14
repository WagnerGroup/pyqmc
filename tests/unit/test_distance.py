import pyqmc.mc as mc
import pyscf.gto as gto
import pyscf.scf as scf


def test_pairwise(LiH_sto3g_uhf, tol=1e-10):
    """dupdown is the (nup,ndown) shaped array of electron distances between up spin and down spin electrons
    ij_updown is a list of tuples (i,j) where i<nup, j<ndown

    each element (i,j) of ij_updown indicates that the corresponding element of dupdown is the distance between jth component of
    configs2 and ith component of configs1
    This test checks whether this is true.
    """
    mol, mf = LiH_sto3g_uhf
    nconfigs = 10
    configs = mc.initial_guess(mol, nconfigs)
    nup = mol.nelec[0]
    configs1 = configs.configs[:, :nup]
    configs2 = configs.configs[:, nup:]

    dupdown, ij_updown = configs.dist.pairwise(configs1, configs2)

    for index, ij in enumerate(ij_updown):
        i, j = ij
        assert (dupdown[:, index] - (configs2[:, j] - configs1[:, i]) < tol).all()
