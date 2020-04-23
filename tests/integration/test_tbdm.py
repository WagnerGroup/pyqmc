import numpy as np
from pyscf import gto, scf, lo
from pyqmc import PySCFSlaterUHF
from pyqmc.mc import initial_guess, vmc
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.tbdm import TBDMAccumulator, normalize_tbdm
from pyqmc.obdm import OBDMAccumulator, normalize_obdm
from pandas import DataFrame


###########################################################
def singledet_tbdm(mf, mfobdm):
    """Computes the single Sltater determinant tbdm."""
    if isinstance(mf, scf.hf.RHF):
        # norb=mf.mo_energy.size
        norb = mfobdm.shape[1]
        mftbdm = np.tile(np.nan, [norb, norb, norb, norb])
        mftbdm = np.einsum("ik,jl->ijkl", mfobdm, mfobdm) - np.einsum(
            "il,jk->ijkl", mfobdm, mfobdm
        )
        # Rotation into pySCF's 2-RDMs notation
        mftbdm = np.transpose(mftbdm, axes=(0, 2, 1, 3))
    elif isinstance(mf, scf.uhf.UHF):
        # norb=mf.mo_energy[0].size
        norb = mfobdm.shape[1]
        mftbdm = np.tile(np.nan, [2, 2] + [norb, norb, norb, norb])
        for spin in 0, 1:
            mftbdm[spin, spin] = np.einsum(
                "ik,jl->ijkl", mfobdm[spin], mfobdm[spin]
            ) - np.einsum("il,jk->ijkl", mfobdm[spin], mfobdm[spin])
        mftbdm[0, 1] = np.einsum("ik,jl->ijkl", mfobdm[0], mfobdm[1])
        mftbdm[1, 0] = np.einsum("ik,jl->ijkl", mfobdm[1], mfobdm[0])
        # Rotation into pySCF's 2-RDMs notation
        mftbdm = np.transpose(mftbdm, axes=(0, 1, 2, 4, 3, 5))

    return mftbdm


###########################################################


###########################################################
def make_combined_spin_iaos(cell, mf, mos, iao_basis="minao"):
    """ Make IAOs for up and down MOs together for all k points. 
  Args:
    cell (PySCF cell): Cell for the calculation.
    mf (PySCF UKS or UHF object): Contains the MOs information.
    mos (array): indices of the MOs to use to make the IAOs.
    basis (basis): IAO basis desired (in PySCF format).
  Returns:
    iaos_all (list): each list entry is np array of IAO orbitals 
                     in the basis of cell for a given k point.
  """
    # print("Making combined spin-up and spin-dw IAOs...")
    ovlp = mf.get_ovlp()
    # Concatenates the spin-up and the spin-down chosen MOs
    coefs = np.array(mf.mo_coeff)[
        :, :, mos
    ]  # Notice that, unlike the KUHF case, here we do not need to transpose the matrix
    coefs = np.concatenate([coefs[0].T, coefs[1].T]).T
    iaos = lo.iao.iao(cell, coefs, minao=iao_basis)
    iaos = lo.vec_lowdin(iaos, ovlp)
    return np.array([iaos, iaos])


###########################################################


###########################################################
def make_separate_spin_iaos(cell, mf, mos, iao_basis="minao"):
    """ Make IAOs for up and down MOs separately for all k points. 
  Args:
    cell (PySCF cell): Cell for the calculation.
    mf (PySCF UKS or UHF object): Contains the MOs information.
    mos (array): indices of the MOs to use to make the IAOs.
    basis (basis): IAO basis desired (in PySCF format).
  Returns:
    iaos_all (list): each list entry is np array of IAO orbitals 
                     in the basis of cell for a given k point.
  """
    # print("Making combined spin-up and spin-dw IAOs...")
    ovlp = mf.get_ovlp()
    coefs = np.array(mf.mo_coeff)[:, :, mos]
    iaos_up = lo.iao.iao(cell, coefs[0], minao=iao_basis)
    iaos_up = lo.vec_lowdin(iaos_up, ovlp)
    iaos_down = lo.iao.iao(cell, coefs[1], minao=iao_basis)
    iaos_down = lo.vec_lowdin(iaos_down, ovlp)
    return np.array([iaos_up, iaos_down])


###########################################################


###########################################################
def reps_combined_spin_iaos(iaos, mf, mos):
    """ Representation of MOs in IAO basis.
  Args:
    iaos (array): coefficients of IAOs in AO basis.
    mf (UKS or UHF object): the MOs are in here.
    mos (array): MOs to find representation of. Not necessarily the same as in make_combined_spin_iaos()!!!
  Returns:
    array: MO coefficients in IAO basis (remember that MOs are the columns).
  """
    # Checks if mos has 2 spins
    if len(mos) != 2:
        mos = np.array([mos, mos])
    # Computes MOs passed in array 'mos' in terms of the 'iaos' basis
    if len(iaos.shape) == 2:
        iao_reps = [
            np.dot(
                np.dot(iaos.T, mf.get_ovlp()),
                (np.array(mf.mo_coeff)[s, :, mos[s]]).transpose((1, 0)),
            )
            for s in range(np.array(mf.mo_coeff).shape[0])
        ]
    else:
        iao_reps = [
            np.dot(
                np.dot(iaos[s].T, mf.get_ovlp()),
                (np.array(mf.mo_coeff)[s, :, mos[s]]).transpose((1, 0)),
            )
            for s in range(np.array(mf.mo_coeff).shape[0])
        ]

    return iao_reps


###########################################################


def test(atom="He", total_spin=0, total_charge=0, scf_basis="sto-3g"):
    mol = gto.M(
        atom="%s 0. 0. 0.; %s 0. 0. 1.5" % (atom, atom),
        basis=scf_basis,
        unit="bohr",
        verbose=4,
        spin=total_spin,
        charge=total_charge,
    )
    mf = scf.UHF(mol).run()
    # Intrinsic Atomic Orbitals
    iaos = make_separate_spin_iaos(
        mol, mf, np.array([i for i in range(mol.natm)]), iao_basis="minao"
    )
    # iaos=make_combined_spin_iaos(mol,mf,np.array([i for i in range(mol.natm)]),iao_basis='minao')
    # MOs in the IAO basis
    mo = reps_combined_spin_iaos(
        iaos,
        mf,
        np.einsum("i,j->ji", np.arange(mf.mo_coeff[0].shape[1]), np.array([1, 1])),
    )
    # Mean-field obdm in IAO basis
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)
    # Mean-field tbdm in IAO basis
    mftbdm = singledet_tbdm(mf, mfobdm)

    ### Test TBDM calculation.
    # VMC params
    nconf = 500
    n_vmc_steps = 400
    vmc_tstep = 0.3
    vmc_warmup = 30
    # TBDM params
    tbdm_sweeps = 4
    tbdm_tstep = 0.5

    wf = PySCFSlaterUHF(mol, mf)  # Single-Slater (no jastrow) wf
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
    obdm_up = OBDMAccumulator(mol=mol, orb_coeff=iaos[0], nsweeps=tbdm_sweeps, spin=0)
    obdm_down = OBDMAccumulator(mol=mol, orb_coeff=iaos[1], nsweeps=tbdm_sweeps, spin=1)
    tbdm_upup = TBDMAccumulator(
        mol=mol, orb_coeff=iaos, nsweeps=tbdm_sweeps, tstep=tbdm_tstep, spin=(0, 0)
    )
    tbdm_updown = TBDMAccumulator(
        mol=mol, orb_coeff=iaos, nsweeps=tbdm_sweeps, tstep=tbdm_tstep, spin=(0, 1)
    )
    tbdm_downup = TBDMAccumulator(
        mol=mol, orb_coeff=iaos, nsweeps=tbdm_sweeps, tstep=tbdm_tstep, spin=(1, 0)
    )
    tbdm_downdown = TBDMAccumulator(
        mol=mol, orb_coeff=iaos, nsweeps=tbdm_sweeps, tstep=tbdm_tstep, spin=(1, 1)
    )

    print("VMC...")
    df, coords = vmc(
        wf,
        configs,
        nsteps=n_vmc_steps,
        tstep=vmc_tstep,
        accumulators={
            "energy": energy,
            "obdm_up": obdm_up,
            "obdm_down": obdm_down,
            "tbdm_upup": tbdm_upup,
            "tbdm_updown": tbdm_updown,
            "tbdm_downup": tbdm_downup,
            "tbdm_downdown": tbdm_downdown,
        },
        verbose=True,
    )
    df = DataFrame(df)

    # Compares obdm from QMC and MF
    obdm_est = {}
    for k in ["obdm_up", "obdm_down"]:
        avg_norm = np.array(df.loc[vmc_warmup:, k + "norm"].values.tolist()).mean(
            axis=0
        )
        avg_obdm = np.array(df.loc[vmc_warmup:, k + "value"].values.tolist()).mean(
            axis=0
        )
        obdm_est[k] = normalize_obdm(avg_obdm, avg_norm)
    qmcobdm = np.array([obdm_est["obdm_up"], obdm_est["obdm_down"]])
    print("\nComparing QMC and MF obdm:")
    for s in [0, 1]:
        # print('QMC obdm[%d]:\n'%s,qmcobdm[s])
        # print('MF obdm[%d]:\n'%s,mfobdm[s])
        print("diff[%d]:\n" % s, qmcobdm[s] - mfobdm[s])

    # Compares tbdm from QMC and MF
    avg_norm = {}
    avg_tbdm = {}
    tbdm_est = {}
    for t in ["tbdm_upup", "tbdm_updown", "tbdm_downup", "tbdm_downdown"]:
        for k in df.keys():
            if k.startswith(t + "norm_"):
                avg_norm[k.split("_")[-1]] = np.array(
                    df.loc[vmc_warmup:, k].values.tolist()
                ).mean(axis=0)
            if k.startswith(t + "value"):
                avg_tbdm[k.split("_")[-1]] = np.array(
                    df.loc[vmc_warmup:, k].values.tolist()
                ).mean(axis=0)
    for k in avg_tbdm.keys():
        tbdm_est[k] = normalize_tbdm(
            avg_tbdm[k].reshape(2, 2, 2, 2), avg_norm["a"], avg_norm["b"]
        )
    qmctbdm = np.array(
        [
            [tbdm_est["upupvalue"], tbdm_est["updownvalue"]],
            [tbdm_est["downupvalue"], tbdm_est["downdownvalue"]],
        ]
    )
    print("\nComparing QMC and MF tbdm:")
    for sa, sb in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        # print('QMC tbdm[%d,%d]:\n'%(sa,sb),qmctbdm[sa,sb])
        # print('MF tbdm[%d,%d]:\n'%(sa,sb),mftbdm[sa,sb])
        diff = qmctbdm[sa, sb] - mftbdm[sa, sb]
        print("diff[%d,%d]:\n" % (sa, sb), diff)
        assert np.max(np.abs(diff)) < 0.05


if __name__ == "__main__":
    # Tests He2 molecule (Sz=0)
    test(atom="He", total_spin=0, scf_basis="cc-pvdz")
    # Tests He2- molecule (Sz=1)
    test(atom="He", total_spin=1, total_charge=-1, scf_basis="cc-pvdz")
