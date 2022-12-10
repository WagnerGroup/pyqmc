import pyqmc.mc as mc


def test_distance(LiH_sto3g_uhf):
    mol, mf = LiH_sto3g_uhf
    nconfigs = 10
    configs = mc.initial_guess(mol, nconfigs)
    nup = mol.nelec[0]
    # dupdown is the (nup,ndown) shaped array of electron distances between up spin and down spin electrons
    configs1 = configs.configs[:, :nup]
    configs2 = configs.configs[:, nup:]

    dupdown, ij_updown = configs.dist.pairwise(configs1, configs2)

    for index, ij in enumerate(ij_updown):
        i, j = ij
        assert (dupdown[:, index] == (configs2[:, j] - configs1[:, i])).all()
