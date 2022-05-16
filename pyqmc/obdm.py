""" Evaluate the OBDM for a wave function object. """
import pyqmc.orbitals
import numpy as np
import pyqmc.mc as mc
from pyqmc.gpu import cp, asnumpy

import pyqmc.supercell as supercell


class OBDMAccumulator:
    r"""Return the obdm as an array with indices rho[spin][i][j] = <c^+_{spin,i}c_{spin,j}>

    .. math:: \rho^\sigma_{ij} = \langle c^\dagger_{\sigma, i} c_{\sigma, j} \rangle

    We are measuring the amplitude of moving one electron (e.g. the first one) from orbital :math:`\phi_i` to orbital :math:`\phi_j` (Eq (7) of DOI:10.1063/1.4793531)

    .. math:: \rho_{i,j} = \int dR dr' \Psi^*(R') \phi_i(r') \phi_j^*(r) \Psi(R)

    (The complex conjugate is on the wrong orbital in Eq (7) in the paper.) Sampling :math:`R` from :math:`|\Psi(R)^2|` and :math:`r'` from :math:`f(r) = \sum_i |\phi(r)|^2`

    .. math:: \rho_{i,j} = \int dR dr' \frac{\Psi^*(R')}{\Psi^*(R)} \left[\Psi^*(R) \Psi(R)\right] \frac{\phi_i(r') \phi_j^*(r)}{f(r)} \left[f(r)\right]

    The distributions (in square brackets) are accounted for by the Monte Carlo integration

    .. math:: \rho_{i,j} = \left\langle \frac{\Psi^*(R')}{\Psi^*(R)} \frac{\phi_i(r') \phi_j^*(r)}{f(r)} \right\rangle

    Eq (9) in the paper is the complex conjugate of this

    .. math:: \rho_{i,j}^* = \left\langle \frac{\Psi(R')}{\Psi(R)} \frac{\phi_j^*(r') \phi_i(r)}{f(r)} \right\rangle


    Args:

      mol (Mole): PySCF Mole object.

      configs (array): electron positions.

      wf (pyqmc wave function object): wave function to evaluate on.

      orb_coeff (array): coefficients with size (nbasis,norb) relating mol basis to basis
        of 1-RDM desired.

      tstep (float): width of the Gaussian to update a walker position for the
        extra coordinate.

      spin: 0 or 1 for up or down. Defaults to all electrons.
    """

    def __init__(
        self,
        mol,
        orb_coeff,
        nsweeps=5,
        tstep=0.50,
        warmup=100,
        naux=None,
        spin=None,
        electrons=None,
        kpts=None,
    ):

        if spin is not None:
            if spin == 0:
                self._electrons = np.arange(0, mol.nelec[0])
            elif spin == 1:
                self._electrons = np.arange(mol.nelec[0], np.sum(mol.nelec))
            else:
                raise ValueError("Spin not equal to 0 or 1")
        elif electrons is not None:
            self._electrons = electrons
        else:
            self._electrons = np.arange(0, np.sum(mol.nelec))

        if kpts is None:
            self.orbitals = pyqmc.orbitals.MoleculeOrbitalEvaluator(
                mol, [orb_coeff, orb_coeff]
            )
            if hasattr(mol, "a"):
                raise ValueError("kpts is required if the system is periodic")
        else:
            if not hasattr(mol, "original_cell"):
                mol = supercell.get_supercell(mol, np.eye(3))
            self.orbitals = pyqmc.orbitals.PBCOrbitalEvaluatorKpoints(
                mol, [orb_coeff, orb_coeff], kpts
            )

        self.iscomplex = self.orbitals.iscomplex

        self._tstep = tstep
        self.nelec = len(self._electrons)
        self._nsweeps = nsweeps
        self._nstep = nsweeps * self.nelec
        self._warmup = warmup
        self._naux = naux
        self._warmed_up = False
        self._mol = mol
        self.norb = self.orbitals.parameters["mo_coeff_alpha"].shape[-1]

    def warm_up(self, naux):
        self._extra_config = mc.initial_guess(self._mol, int(naux / self.nelec) + 1)
        self._extra_config.reshape((-1, 1, 3))
        self._extra_config.resample(range(naux))
        _, extra_configs, _ = sample_onebody(
            self._extra_config,
            self.orbitals,
            spin=0,
            nsamples=self._warmup,
            tstep=self._tstep,
        )
        self._extra_config = extra_configs[-1]

    def __call__(self, configs, wf):
        """"""

        nconf = configs.configs.shape[0]
        if not self._warmed_up:
            naux = nconf if self._naux is None else self._naux
            self.warm_up(naux)
            self._warmed_up = True

        dtype = complex if self.iscomplex else float
        results = {
            "value": np.zeros((nconf, self.norb, self.norb), dtype=dtype),
            "norm": np.zeros((nconf, self.norb)),
        }
        naux = self._extra_config.configs.shape[0]

        auxassignments = np.random.randint(0, naux, size=(self._nsweeps, nconf))
        accept, extra_configs, borb_aux = sample_onebody(
            self._extra_config,
            self.orbitals,
            spin=0,
            nsamples=self._nsweeps,
            tstep=self._tstep,
        )
        self._extra_config = extra_configs[-1]

        for conf, assign in zip(extra_configs, auxassignments):
            conf.resample(assign)
        borb_aux = cp.asarray(
            [orb[assign, ...] for orb, assign in zip(borb_aux, auxassignments)]
        )

        borb_configs = self.evaluate_orbitals(configs.electron(self._electrons))
        borb_configs = borb_configs.reshape(nconf, self.nelec, -1)

        bauxsquared = cp.abs(borb_aux) ** 2
        fsum = cp.sum(bauxsquared, axis=-1, keepdims=True)
        norm = bauxsquared / fsum
        baux_f = borb_aux / fsum

        for sweep, aux in enumerate(auxassignments):
            wfratio = wf.testvalue_many(
                self._electrons, extra_configs[sweep].electron(0)
            )
            ratio = np.einsum(
                "ie,ij,iek->ijk",
                wfratio.conj(),
                baux_f[sweep, :],
                borb_configs.conj(),
                optimize=True,
            )

            results["value"] += asnumpy(ratio)
            results["norm"] += asnumpy(norm[sweep, :])

        results["value"] /= self._nstep
        results["norm"] = results["norm"] / self._nstep
        return results

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def evaluate_orbitals(self, configs):
        ao = self.orbitals.aos("GTOval_sph", configs)
        return self.orbitals.mos(ao, spin=0)

    def keys(self):
        return set(
            ["value", "norm"],
        )

    def shapes(self):
        norb = self.norb
        return {
            "value": (norb, norb),
            "norm": (norb,),
        }


def sample_onebody(configs, orbitals, spin, nsamples=1, tstep=0.5):
    r"""For a set of orbitals defined by orb_coeff, return samples from :math:`f(r) = \sum_i \phi_i(r)^2`."""
    n = configs.configs.shape[0]
    ao = orbitals.aos("GTOval_sph", configs)
    borb = orbitals.mos(ao, spin=spin)
    fsum = (cp.abs(borb) ** 2).sum(axis=1)

    allaccept = np.zeros((nsamples, n))
    allconfigs = []
    allorbs = []
    for s in range(nsamples):
        shift = np.sqrt(tstep) * np.random.randn(*configs.configs.shape)
        newconfigs = configs.make_irreducible(0, configs.configs + shift)
        ao = orbitals.aos("GTOval_sph", newconfigs)
        borbnew = orbitals.mos(ao, spin=spin)
        fsumnew = (cp.abs(borbnew) ** 2).sum(axis=1)
        accept = asnumpy(fsumnew / fsum) > np.random.rand(n)
        configs.move_all(newconfigs, accept)
        borb[accept] = borbnew[accept]
        fsum[accept] = fsumnew[accept]
        allconfigs.append(configs.copy())
        allaccept[s] = accept
        allorbs.append(borb.copy())

    return allaccept, allconfigs, allorbs


def normalize_obdm(obdm, norm):
    return obdm / (norm[np.newaxis, :] * norm[:, np.newaxis]) ** 0.5
