import numpy as np
import pyqmc
import pyqmc.energy
import pyqmc.gpu as gpu


class Ewald:

    def __init__(self, cell, ewald_gmax=200, nlatvec=1):
        """
        :parameter cell: pyscf Cell object (simulation cell)
        :parameter int ewald_gmax: how far to take reciprocal sum; probably never needs to be changed.
        :parameter int nlatvec: how far to take real-space sum; probably never needs to be changed.
        """
        self.latvec = cell.lattice_vectors()
        self.atom_coords = cell.atom_coords()
        self.nelec = np.array(cell.nelec)
        self.atom_charges = gpu.cp.asarray(cell.atom_charges())
        self.dist = pyqmc.distance.MinimalImageDistance(self.latvec)
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        """
        Generates list of lattice-vector displacements to add together for real-space sum

        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        """
        XYZ = np.meshgrid(*[np.arange(-nlatvec, nlatvec + 1)] * 2, indexing="ij")
        xyz = np.stack(XYZ, axis=-1).reshape((-1, 2))
        z_zeros = np.zeros((xyz.shape[0], 1))
        xyz = np.concatenate([xyz, z_zeros], axis=1)
        self.lattice_displacements = gpu.cp.asarray(np.dot(xyz, self.latvec))

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        r"""
        Determine parameters for Ewald sums.

        :math:`\alpha` determines the partitioning of the real and reciprocal-space parts.

        We define a weight `gweight` for the part of the reciprocal-space sums that doesn't depend on the coordinates:

        .. math:: W_G = \frac{4\pi}{V |\vec{G}|^2} e^{- \frac{|\vec{G}|^2}{ 4\alpha^2}}

        :parameter int ewald_gmax: max number of reciprocal lattice vectors to check away from 0
        """
        recvec = np.linalg.inv(self.latvec).T
        recvec[2, 2] = 0

        # Determine alpha
        smallestheight = np.amin(1 / np.linalg.norm(recvec[:2, :2], axis=1))
        self.alpha = 5.0 / smallestheight

        # Determine G points to include in reciprocal Ewald sum
        self.gpoints_all = generate_positive_gpoints(ewald_gmax, recvec)
        self.cellvolume = np.linalg.det(self.latvec[:2, :2])

        self.gpoints, self.gweight = select_big(self.gpoints_all, self.cellvolume, self.alpha)
        self.set_ewald_constants(self.cellvolume)

    def set_ewald_constants(self, cellvolume):
        r"""
        Compute Ewald constants (independent of particle positions): self energy and charged-system energy. Here we compute the combined terms. These terms are independent of the convergence parameters `gmax` and `nlatvec`, but do depend on the partitioning parameter :math:`\alpha`.

        We define two constants, `squareconst`, the coefficient of the squared charges,
        and `ijconst`, the coefficient of the pairs:

        .. math:: C_{ij} = - \frac{\pi}{V\alpha^2}

        .. math:: C_{\rm square} = - \frac{\alpha}{\sqrt{\pi}}  - \frac{\pi}{2V\alpha^2}
                  = - \frac{\alpha}{\sqrt{\pi}}  - \frac{C_{ij}}{2}

        The Ewald object doesn't retain information about the configurations, including number of electrons, so the electron constants are defined as functions of :math:`N_e`.


        Self plus charged-system energy:

        .. math:: E_{\rm self+charged}^{ee} = N_e C_{\rm square} + \frac{N_e(N_e-1)}{2} C_{ij}

        .. math:: E_{\rm self+charged}^{e\text{-ion}} = - N_e \sum_{I=1}^{N_{ion}} Z_I C_{ij}

        .. math:: E_{\rm self+charged}^{\text{ion-ion}} = \sum_{I=1}^{N_{ion}} Z_I^2 C_{\rm square} + \sum_{I<J}^{N_{ion}} Z_I Z_J C_{ij}

        We also compute contributions from a single electron, to separate the Ewald sum by electron.

        .. math:: E_{\rm self+charged}^{\rm single} = C_{\rm square} + \frac{N_e-1}{2} C_{ij} - \sum_{I=1}^{N_{ion}} Z_I C_{ij}

        .. math:: E_{\rm self+charged}^{\text{single-test}} = C_{\rm square} - \sum_{I=1}^{N_{ion}} Z_I C_{ij}

        """
        self.i_sum = np.sum(self.atom_charges)
        self.ii_sum2 = np.sum(self.atom_charges**2)
        ii_sum = (self.i_sum**2 - self.ii_sum2) / 2
        self.ijconst = -2*np.pi**0.5 / (cellvolume * self.alpha)
        self.squareconst = -self.alpha / np.sqrt(np.pi) + self.ijconst / 2

        self.ii_const = ii_sum * self.ijconst + self.ii_sum2 * self.squareconst
        self.e_single_test = -self.i_sum * self.ijconst + self.squareconst
        self.ion_ion = self.ewald_ion()

    def ee_const(self, ne):
        return ne * (ne - 1) / 2 * self.ijconst + ne * self.squareconst

    def ei_const(self, ne):
        return -ne * self.i_sum * self.ijconst

    def e_single(self, ne):
        return (
            0.5 * (ne - 1) * self.ijconst - self.i_sum * self.ijconst + self.squareconst
        )

    def ewald_ion(self):
        r"""
        Compute ion contribution to Ewald sums.  Since the ions don't move in our calculations, the ion-ion term only needs to be computed once.

        Note: We ignore the constant term :math:`\frac{1}{2} \sum_{I} Z_I^2 C_{\rm self\ image}` in the real-space ion-ion sum corresponding to the interaction of an ion with its own image in other cells.

        The real-space part:

        .. math:: E_{\rm real\ space}^{\text{ion-ion}} = \sum_{\vec{n}} \sum_{I<J}^{N_{ion}} Z_I Z_J \frac{{\rm erfc}(\alpha |\vec{x}_{IJ}+\vec{n}|)}{|\vec{x}_{IJ}+\vec{n}|}

        The reciprocal-space part:

        .. math:: E_{\rm reciprocal\ space}^{\text{ion-ion}} = \sum_{\vec{G} > 0 } W_G \left| \sum_{I=1}^{N_{ion}} Z_I e^{-i\vec{G}\cdot\vec{x}_I} \right|^2

        :returns: ion-ion component of Ewald sum
        :rtype: float
        """
        # Real space part
        if len(self.atom_charges) == 1:
            ion_ion_real = 0
        else:
            ion_distances, ion_inds = self.dist.dist_matrix(self.atom_coords[np.newaxis])
            ion_distances = gpu.cp.asarray(ion_distances)
            rvec = ion_distances[:, :, np.newaxis, :] + self.lattice_displacements
            r = gpu.cp.linalg.norm(rvec, axis=-1)
            charge_ij = gpu.cp.prod(self.atom_charges[np.asarray(ion_inds)], axis=1)
            ion_ion_real = gpu.cp.einsum(
                "j,ijk->", charge_ij, gpu.erfc(self.alpha * r) / r
            )

        # Reciprocal space part
        GdotR = gpu.cp.dot(self.gpoints, gpu.cp.asarray(self.atom_coords.T))
        self.ion_exp = gpu.cp.dot(gpu.cp.exp(1j * GdotR), self.atom_charges)

        ion_distances, ion_inds = self.dist.dist_matrix(self.atom_coords[np.newaxis])
        if len(ion_inds) == 0:
            ion_ion_rec_cross = 0
        else:
            ion_distances = gpu.cp.asarray(ion_distances) # (nconf, npairs, ndim)
            gweights = self.calc_weight_2d_cross(ion_distances) # (nk, npairs)
            Gdotr_ij = np.dot(ion_distances, self.gpoints.T) # (nconf, npairs, nk)
            mul_weight = np.sum(np.exp(1j * Gdotr_ij) * gweights.T[None, :, :], axis=2)
            charge_ij = gpu.cp.prod(self.atom_charges[np.asarray(ion_inds)], axis=1) # (npairs,)
            ion_ion_rec_cross = (mul_weight @ charge_ij).real
        ion_ion_rec_self = self.ii_sum2 * np.sum(self.gweight)
        ion_ion_rec = ion_ion_rec_self + ion_ion_rec_cross
        ion_ion = ion_ion_real + ion_ion_rec
        return ion_ion

    def _real_cij(self, dists):
        dists = gpu.cp.asarray(dists)
        r = gpu.cp.zeros(dists.shape[:-1])
        cij = gpu.cp.zeros(r.shape)
        for ld in self.lattice_displacements:
            r[:] = np.linalg.norm(dists + ld, axis=-1)
            cij += gpu.erfc(self.alpha * r) / r
        return cij

    def ewald_electron(self, configs):
        nconf, nelec, ndim = configs.configs.shape

        # Real space electron-ion part
        # ei_distances shape (elec, conf, atom, dim)
        ei_distances = configs.dist.dist_i(self.atom_coords, configs.configs)
        ei_cij = self._real_cij(ei_distances)
        ei_real_separated = gpu.cp.einsum("k,ijk->ji", -self.atom_charges, ei_cij)

        # Real space electron-electron part
        ee_real_separated = gpu.cp.zeros((nconf, nelec))
        if nelec > 1:
            ee_distances, ee_inds = configs.dist.dist_matrix(configs.configs)
            ee_cij = self._real_cij(ee_distances)

            for (i, j), val in zip(ee_inds, ee_cij.T):
                ee_real_separated[:, i] += val
                ee_real_separated[:, j] += val
            ee_real_separated /= 2

        ee_recip, ei_recip = self.reciprocal_space_electron(configs.configs)
        ee = ee_real_separated.sum(axis=1) + ee_recip
        ei = ei_real_separated.sum(axis=1) + ei_recip
        return ee, ei

    def reciprocal_space_electron(self, configs):
        # Reciprocal space electron-electron part
        e_GdotR = gpu.cp.einsum("hik,jk->hij", gpu.cp.asarray(configs), self.gpoints) # (nconf, nelec, nk)

        sum_e_sin = gpu.cp.sin(e_GdotR).sum(axis=1) # (nconf, nk)
        sum_e_cos = gpu.cp.cos(e_GdotR).sum(axis=1)
        ee_recip_self = gpu.cp.dot(sum_e_sin**2 + sum_e_cos**2, self.gweight)

        ## Reciprocal space electron-ion part
        # coscos_sinsin = -self.ion_exp.real * sum_e_cos - self.ion_exp.imag * sum_e_sin # (nconf, nk)
        # ei_recip = 2 * gpu.cp.dot(coscos_sinsin, gweight)

        ee_distances, ee_inds = self.dist.dist_matrix(configs) # (nconf, npairs, ndim)
        gweight_ee_test = self.calc_weight_2d_cross(ee_distances)
        sum_e_sin_test = gpu.cp.sin(e_GdotR) # (nconf, npairs, nk)
        sum_e_cos_test = gpu.cp.cos(e_GdotR)
        ee_recip_cross = np.sum((sum_e_sin_test**2 + sum_e_cos_test**2) * gweight_ee_test.T[None, :, :], axis=(1, 2))

        coscos_sinsin_test = -self.ion_exp.real * sum_e_cos_test - self.ion_exp.imag * sum_e_sin_test # (nconf, npairs, nk)
        ei_distances_test = self.dist.dist_i(configs, self.atom_coords)
        gweight_ei_test = self.calc_weight_2d_cross(ei_distances_test)
        ei_recip = np.sum(coscos_sinsin_test * gweight_ei_test.T[None, :, :], axis=(1, 2))

        ee_recip = ee_recip_self + ee_recip_cross
        return ee_recip, ei_recip
    def calc_weight_2d_cross(self, dist_matrix):
        dist_z = dist_matrix[:, :, 2]
        gsquared = gpu.cp.einsum("jk,jk->j", self.gpoints, self.gpoints)
        gnorm = gsquared**0.5
        gweight = 2 * gpu.cp.pi * (
                gpu.cp.exp(gnorm.reshape(-1, 1)*dist_z)*gpu.erfc(self.alpha*dist_z + gnorm.reshape(-1, 1)/(2*self.alpha)) +
                gpu.cp.exp(-gnorm.reshape(-1, 1)*dist_z)*gpu.erfc(-self.alpha*dist_z + gnorm.reshape(-1, 1)/(2*self.alpha))
            )
        gweight /= self.cellvolume * gnorm.reshape(-1, 1)
        return gweight

    def reciprocal_space_electron_separated(self, configs):
        # Reciprocal space electron-electron part
        e_GdotR = np.einsum("hik,jk->hij", gpu.cp.asarray(configs), self.gpoints)
        e_sin = np.sin(e_GdotR)
        e_cos = np.cos(e_GdotR)
        sinsin = e_sin.sum(axis=1, keepdims=True) * e_sin
        coscos = e_cos.sum(axis=1, keepdims=True) * e_cos
        ee_recip = np.dot(coscos + sinsin - 0.5, self.gweight)
        ## Reciprocal space electron-ion part
        coscos_sinsin = -self.ion_exp.real * e_cos + self.ion_exp.imag * e_sin
        ei_recip = np.dot(coscos_sinsin, self.gweight)
        return ee_recip, ei_recip

    def save_separated(self, ee_recip, ei_recip, ee_real, ei_real):
        # Combine parts
        self.ei_separated = ei_real + 2 * ei_recip
        self.ee_separated = ee_real + 1 * ee_recip
        self.ewalde_separated = self.ei_separated + self.ee_separated
        nelec = ee_recip.shape[1]
        ### Add back the 0.5 that was subtracted earlier
        ee = self.ee_separated.sum(axis=1) + nelec / 2 * self.gweight.sum()
        ei = self.ei_separated.sum(axis=1)
        return ee, ei

    def energy(self, configs):
        r"""
        Compute Coulomb energy for a set of configs.

        .. math:: E_{\rm Coulomb} &= E_{\rm real+reciprocal}^{ee}
                + E_{\rm self+charged}^{ee}
                \\&+ E_{\rm real+reciprocal}^{e\text{-ion}}
                + E_{\rm self+charged}^{e\text{-ion}}
                \\&+ E_{\rm real+reciprocal}^{\text{ion-ion}}
                + E_{\rm self+charged}^{\text{ion-ion}}

        :parameter configs: electron positions (walkers)
        :type configs: (nconf, nelec, 3) PeriodicConfigs object
        :returns:
            * ee: electron-electron part
            * ei: electron-ion part
            * ii: ion-ion part
        :rtype: float, float, float
        """
        nelec = configs.configs.shape[1]
        ee, ei = self.ewald_electron(configs)
        ee += self.ee_const(nelec)
        ei += self.ei_const(nelec)
        ii = self.ion_ion + self.ii_const
        return gpu.asnumpy(ee), gpu.asnumpy(ei), gpu.asnumpy(ii)

    def energy_separated(self, configs):
        """
        Compute Coulomb energy contribution from each electron (does not include ion-ion energy).

        NOTE: energy() needs to be called first to update the separated energy values

        :parameter configs: electron positions (walkers)
        :type configs: (nconf, nelec, 3) PeriodicConfigs object
        :returns: energies
        :rtype: (nelec,) array
        """
        raise NotImplementedError("ewalde_separated is currently not computed anywhere")
        nelec = configs.configs.shape[1]
        return self.e_single(nelec) + self.ewalde_separated



def select_big(gpoints, cellvolume, alpha, tol=1e-10):
    gsquared = gpu.cp.einsum("jk,jk->j", gpoints, gpoints)
    gnorm = gsquared**0.5
    gweight = gpu.cp.pi * gpu.erfc(gnorm/(2*alpha)) * 2
    gweight /= cellvolume * gnorm
    bigweight = gweight > tol
    return gpoints[bigweight], gweight[bigweight]

def generate_positive_gpoints(gmax, recvec):
    gXpos = gpu.cp.mgrid[1 : gmax + 1, -gmax: gmax + 1, 0:1].reshape(3, -1)
    gX0Ypos = gpu.cp.mgrid[0:1, 1: gmax + 1, 0:1].reshape(3, -1)
    gpts = gpu.cp.concatenate([gXpos, gX0Ypos], axis=1)
    gpoints = gpu.cp.einsum("ji,jk->ik", gpts, gpu.cp.asarray(recvec) * 2 * np.pi)
    return gpoints
