""" Evaluate the TBDM for a wave function object. """
import numpy as np
import pyqmc.mc as mc
import pyqmc.obdm as obdm
import pyqmc.gpu as gpu
import pyqmc.orbitals
import pyqmc.supercell as supercell
import warnings


class TBDMAccumulator:
    r"""Returns one spin sector of the tbdm[s1,s2] as an array (norb_s1,norb_s1,norb_s2,norb_s2)

    We use PySCF's index convention (note that Eq. 10 in DOI:10.1063/1.4793531 uses QWalk's).
    PySCF -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}

    .. math:: \rho_{ijkl}^{\sigma_1\sigma_2} = \left\langle c^\dagger_{\sigma_1, i} c^\dagger_{\sigma_2, k} c_{\sigma_2, l} c_{\sigma_1, j} \right\rangle = \phi^*_{\sigma_1,j} \phi^*_{\sigma_2,l} \phi_{\sigma_2,k} \phi_{\sigma_1,i}.

    .. math:: \rho_{ijkl} = \left\langle \sum_{a<b} \frac{\Psi(\mathbf{R}'_{ab})}{\Psi(\mathbf{R})} \frac{\phi^*_{j}(\mathbf{r}_a') \phi^*_{l}(\mathbf{r}_b') \phi_{i}(\mathbf{r}_a) \phi_{k}(\mathbf{r}_b) }{\rho_{\rm aux}(\mathbf{r}_a')\rho_{\rm aux}(\mathbf{r}_b')} \right\rangle_{\begin{subarray}{l}\mathbf{R}\sim|\Psi|^2;\\ \mathbf{r}_a'\sim\rho_{\rm aux}\end{subarray}},

    QWalk -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,j} c_{s2,l} c_{s1,k} > = \phi*_{s1,k} \phi*_{s2,l} \phi_{s2,j} \phi_{s1,i}

    .. math:: \rho_{ijkl}^{\sigma_1\sigma_2} = \left\langle c^\dagger_{\sigma_1, i} c^\dagger_{\sigma_2, j} c_{\sigma_2, l} c_{\sigma_1, k} \right\rangle = \phi^*_{\sigma_1,k} \phi^*_{\sigma_2,l} \phi_{\sigma_2,j} \phi_{\sigma_1,i}.

    Args:

      mol (Mole): PySCF Mole object.

      orb_coeff (array): coefficients with size (2,nbasis,norb) relating mol basis to basis
        of the 2-RDM.

      nsweeps (int): number of sweeps over all the electron pairs.

      tstep (float): width of the Gaussian to update a walker position for each extra coordinate.

      warmup (int): number of warmup steps for single-particle local orbital sampling.

      naux (int): number of auxiliary configurations for extra moves sampling the local
         orbitals.

      spin (tuple): size 2 indicates spin sector to be computed, either (0,0), (0,1), (1,0) or (1,1)
         for up-up, up-down, down-up or down-down.

      ijkl (array): contains M tbdm matrix elements to calculate with dim (M,4).
    """

    def __init__(
        self,
        mol,
        orb_coeff,
        spin,
        nsweeps=4,
        tstep=0.50,
        warmup=200,
        naux=None,
        ijkl=None,
        kpts=None,
    ):

        self._mol = mol
        self._tstep = tstep
        self._nsweeps = nsweeps
        self._spin = spin
        self._naux = naux
        self._warmup = warmup

        if kpts is None:
            self.orbitals = pyqmc.orbitals.MoleculeOrbitalEvaluator(mol, orb_coeff)
            norb_up = orb_coeff[0].shape[1]
            norb_down = orb_coeff[1].shape[1]
            if hasattr(mol, "a"):
                warnings.warn(
                    "Using molecular orbital evaluator for a periodic system. This is likely wrong unless you know what you're doing. Make sure to pass kpts into TBDM if you want to use the periodic orbital evaluator."
                )
        else:
            if not hasattr(mol, "original_cell"):
                mol = supercell.get_supercell(mol, np.eye(3))
            self.orbitals = pyqmc.orbitals.PBCOrbitalEvaluatorKpoints(
                mol, orb_coeff, kpts
            )
            norb_up = np.sum([o.shape[1] for o in orb_coeff[0]])
            norb_down = np.sum([o.shape[1] for o in orb_coeff[1]])

        self.dtype = complex if self.orbitals.iscomplex else float
        self._spin_sector = spin
        self._electrons = [
            np.arange(spin[s] * mol.nelec[0], mol.nelec[0] + spin[s] * mol.nelec[1])
            for s in [0, 1]
        ]

        # Default to full 2rdm if ijkl not specified
        if ijkl is None:
            ijkl = [
                [i, j, k, l]
                for i in range(norb_up)
                for j in range(norb_up)
                for k in range(norb_down)
                for l in range(norb_down)
            ]
        self._ijkl = np.array(ijkl).T
        self._warmed_up = False

    def warm_up(self, naux):
        # Initialization and warmup of configurations
        nwalkers = int(naux / sum(self._mol.nelec)) + 1
        self._aux_configs = []
        for spin in [0, 1]:
            self._aux_configs.append(mc.initial_guess(self._mol, nwalkers))
            self._aux_configs[spin].reshape((-1, 1, 3))
            self._aux_configs[spin].resample(range(naux))
            _, self._aux_configs[spin], _ = obdm.sample_onebody(
                self._aux_configs[spin], self.orbitals, 0, nsamples=self._warmup
            )
            self._aux_configs[spin] = self._aux_configs[spin][-1]

    def get_configurations(self, nconf):
        """
        Obtain a sequence of auxilliary configurations. This function returns one auxilliary configuration
        for each nconf.
        Changes internal state: self._aux_configs is updated to the last sampled location.

        This will resample the auxilliary configurations to match the number of walkers.

        returns a dictionary with the following elements, separated by spin:
            assignments: [nsweeps, nconf]: assignment of configurations for each sweep to an auxilliary walker.
            orbs: [nsweeps, conf, norb]: orbital values
            configs: [nsweeps] Configuration object with nconf configurations of 1 electron
            acceptance: [nsweeps, naux] acceptance probability for each auxilliary walker
        """
        configs = []
        assignments = []
        orbs = []
        acceptance = []

        for spin in [0, 1]:
            naux = self._aux_configs[spin].configs.shape[0]
            accept, tmp_config, tmp_orbs = obdm.sample_onebody(
                self._aux_configs[spin],
                self.orbitals,
                spin,
                self._nsweeps,
                tstep=self._tstep,
            )
            assignments.append(np.random.randint(0, naux, size=(self._nsweeps, nconf)))
            self._aux_configs[spin] = tmp_config[-1].copy()
            acceptance.append(accept)
            for conf, assign in zip(tmp_config, assignments[-1]):
                conf.resample(assign)
            configs.append(tmp_config)
            orbs.append(
                [orb[assign, ...] for orb, assign in zip(tmp_orbs, assignments[-1])]
            )

        return {
            "acceptance": acceptance,
            "orbs": orbs,
            "configs": configs,
            "assignments": assignments,
        }

    def __call__(self, configs, wf):
        """Gathers quantities from equation (10) of DOI:10.1063/1.4793531.

        assignments maps from the auxilliary walkers onto the main walkers.
        It should be of length [nsweeps,nconf], and contain integers between 0 and naux.
        """

        nconf, nelec = configs.configs.shape[:2]
        if not self._warmed_up:
            naux = nconf if self._naux is None else self._naux
            self.warm_up(naux)
            self._warmed_up = True

        aux = self.get_configurations(nconf)

        # Evaluate orbital values for the primary samples
        ao_configs = self.orbitals.aos("GTOval_sph", configs)
        ao_configs = ao_configs.reshape(
            (ao_configs.shape[0], nconf, -1, ao_configs.shape[-1])
        )
        orb_configs = [
            self.orbitals.mos(ao_configs[..., self._electrons[spin], :], spin)
            for spin in [0, 1]
        ]
        results = {
            "value": np.zeros((nconf, self._ijkl.shape[1]), dtype=self.dtype),
            "norm_a": np.zeros((nconf, orb_configs[0].shape[-1])),
            "norm_b": np.zeros((nconf, orb_configs[1].shape[-1])),
        }
        orb_configs = [orb_configs[s][:, :, self._ijkl[2 * s]] for s in [0, 1]]

        down_start = [np.min(self._electrons[s]) for s in [0, 1]]

        _, saved0 = list(
            zip(*[wf.testvalue(e, configs.electron(e)) for e in range(nelec)])
        )
        for sweep in range(self._nsweeps):
            fsum = [
                gpu.cp.sum(gpu.cp.abs(aux["orbs"][spin][sweep]) ** 2, axis=1)
                for spin in [0, 1]
            ]
            norm = [
                gpu.cp.abs(aux["orbs"][spin][sweep]) ** 2 / fsum[spin][:, np.newaxis]
                for spin in [0, 1]
            ]

            wfratio = []
            electrons_a_ind = []
            electrons_b_ind = []
            for ea in self._electrons[0]:
                # Don't move the same electron twice
                electrons_b = self._electrons[1][self._electrons[1] != ea]
                epos_a = aux["configs"][0][sweep].electron(0)
                epos_b = aux["configs"][1][sweep].electron(0)
                wfratio_a, saved_a = wf.testvalue(ea, epos_a)
                wf.updateinternals(ea, epos_a, configs, saved_values=saved_a)
                wfratio_b = wf.testvalue_many(electrons_b, epos_b)
                wf.updateinternals(
                    ea, configs.electron(ea), configs, saved_values=saved0[ea]
                )
                wfratio.append(wfratio_a[:, np.newaxis] * wfratio_b)
                electrons_a_ind.extend([ea - down_start[0]] * len(electrons_b))
                electrons_b_ind.extend(electrons_b - down_start[1])
            wfratio = np.concatenate(wfratio, axis=1)

            """
            orbratio collects 
            phi_i(r1) phi_j(r1') phi_k(r2) phi_l(r2')/rho(r1') rho(r2')
            """
            phi_j_r1p = aux["orbs"][0][sweep][..., self._ijkl[1]]
            phi_l_r2p = aux["orbs"][1][sweep][..., self._ijkl[3]]
            rho1rho2 = 1.0 / (fsum[0] * fsum[1])
            # n is the walker number, i is the electron pair index, o is the orbital
            orbratio = gpu.cp.einsum(
                "nio,nio,no,no,n ->nio",
                orb_configs[0][:, electrons_a_ind, :],  # phi_i(r1)
                orb_configs[1][:, electrons_b_ind, :],  # phi_k(r2)
                phi_j_r1p.conj(),  # phi_j^*(r1)
                phi_l_r2p.conj(),  # phi_l^*(r2)
                rho1rho2,
            )

            results["value"] += gpu.asnumpy(
                gpu.cp.einsum("in,inj->ij", wfratio, orbratio)
            )
            results["norm_a"] += gpu.asnumpy(norm[0])
            results["norm_b"] += gpu.asnumpy(norm[1])

        results["value"] /= self._nsweeps
        for e in ["a", "b"]:
            results["norm_%s" % e] /= self._nsweeps

        return results

    def keys(self):
        return set(["value", "norm_a", "norm_b"])

    def shapes(self):
        d = {
            "value": (self._ijkl.shape[1],),
        }
        nmo = self.orbitals.nmo()
        for e, s in zip(["a", "b"], self._spin_sector):
            d["norm_%s" % e] = (nmo[s],)
        return d

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}


def normalize_tbdm(tbdm, norm_a, norm_b):
    """Returns tbdm by taking the ratio of the averages in Eq. (10) of DOI:10.1063/1.4793531."""
    # We are using PySCF's notation:
    #  tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}
    return tbdm / np.einsum("i,j,k,l->ijkl", norm_a, norm_a, norm_b, norm_b) ** 0.5
