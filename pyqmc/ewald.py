import numpy as np
import pyqmc
import pyqmc.energy
import pyqmc.gpu as gpu


class Ewald:
    r"""
    The Ewald summation computes the Coulomb energy of a periodic arrangement of charges.
    The sum (of pair interactions) is separated into real-space (short range) and reciprocal-space (long range) sums, each of which converges quickly.
    The separation is determined by the parameter :math:`\alpha`

    The Ewald separation is:

    .. math:: E_{\rm Coulomb} = E_{\rm real\ space} + E_{\rm reciprocal\ space} + E_{\rm self} + E_{\rm charged}

    .. math:: E_{\rm real\ space} = \frac{1}{2} {\sum_{\vec{n}}}^\dagger \sum_{i=1}^N \sum_{j=1}^N q_i q_j \frac{{\rm erfc}(\alpha |\vec{x}_{ij}+\vec{n}|)}{|\vec{x}_{ij}+\vec{n}|}

    :math:`\qquad\qquad` (The :math:`{\sum}^\dagger` means to exclude the self-terms, i.e. :math:`\vec{n}=0, \, i=j`)

    .. math:: E_{\rm reciprocal\ space} = \frac{4\pi}{V} \frac{1}{2} \sum_{k \ne 0} \frac{1}{k^2} e^{-\frac{k^2}{4\alpha^2}} \left| \sum_{i=1}^N q_i e^{-i\vec{k}\cdot\vec{x}_i} \right|^2

    .. math:: E_{\rm self}  = -\frac{\alpha}{\sqrt{\pi}} \sum_{i=1}^N q_i^2
        \qquad
        E_{\rm charged}  = -\frac{\pi}{2V\alpha^2} \left| \sum_{i=1}^N q_i \right|^2

    The self energy corrects for a self-interaction included in the reciprocal-space term, and the charged-system correction is only necessary for systems with nonzero net charge.

    In our implementation, the parts are further split into electron-electron, electron-ion, and ion-ion contributions. We use lower-case summation indices for electrons, and upper case for ions.

    For ease of notation (and reading the code), let pair distances be denoted by

    .. math:: r_{ijn} = |\vec{x}_{ij}+\vec{n}|

    .. math:: r_{iIn} = |\vec{x}_{iI}+\vec{n}|

    .. math:: r_{IJn} = |\vec{x}_{IJ}+\vec{n}|

    Real space terms, arranged to sum over each pair only once:

        .. math:: E_{\rm real\ space}^{\text{ion-ion}} = \sum_{\vec{n}} \sum_{I<J}^{N_{ion}} Z_I Z_J \frac{{\rm erfc}(\alpha r_{IJn})}{r_{IJn}}
            + \frac{1}{2} \sum_{I=1}^{N_{ion}} Z_I^2 C_{\rm self\ image}

        .. math:: E_{\rm real\ space}^{ee} = \sum_{\vec{n}} \sum_{i<j}^{N_e} \frac{{\rm erfc}(\alpha r_{ijn})}{r_{ijn}}
            + \frac{N_e}{2} C_{\rm self\ image}

    .. math:: E_{\rm real\ space}^{e\text{-ion}} = {\sum_{\vec{n}}} \sum_{i=1}^{N_e} \sum_{I=1}^{N_{ion}} -Z_I \frac{{\rm erfc}(\alpha r_{iIn})}{r_{iIn}}

    where the interactions between a particle and its own image in other cells is represented by the sum

    .. math:: C_{\rm self\ image} = \frac{1}{2} \sum_{\vec{n} \ne 0} \frac{{\rm erfc}(\alpha |\vec{n}|)}{|\vec{n}|}

    Reciprocal space terms, summing over :math:`\vec{G}>0` -- reciprocal lattice vectors in positive octants. In other words, only one of :math:`\vec{G}` and :math:`-\vec{G}` is included in the sum, and :math:`\vec{G}=0` is not included.

    .. math:: E_{\rm reciprocal\ space}^{\text{ion-ion}} = \sum_{\vec{G} > 0 } W_G \left| \sum_{I=1}^{N_{ion}} Z_I e^{-i\vec{G}\cdot\vec{x}_I} \right|^2

    .. math:: E_{\rm reciprocal\ space}^{ee} = \sum_{\vec{G}>0} W_G \left| \sum_{i=1}^{N_e} e^{-i\vec{k}\cdot\vec{x}_i} \right|^2

    .. math:: E_{\rm reciprocal\ space}^{e\text{-ion}} = \sum_{\vec{G}>0} W_G {\rm Re} \left[ 2 \sum_{i=1}^{N_e} \sum_{I=1}^{N_{ion}} -Z_I e^{-i\vec{k}\cdot\vec{x}_i} e^{i\vec{k}\cdot\vec{x}_I} \right]

    where `gweight` is a factor that doesn't depend on the coordinates:

    .. math:: W_G = \frac{4\pi}{V |\vec{G}|^2} e^{- \frac{|\vec{G}|^2}{ 4\alpha^2}}

    Self energy:

    .. math:: E_{\rm self}^{e} = - \frac{\alpha}{\sqrt{\pi}} N_e
              \qquad
              E_{\rm self}^{\rm ion} = - \frac{\alpha}{\sqrt{\pi}} \sum_{I=1}^{N_{ion}} Z_I^2

    Charged-system energy:

    .. math:: E_{\rm charged}^{ee} = - \frac{\pi}{2V\alpha^2} N_e^2
              \qquad
              E_{\rm charged}^{e\text{-ion}} =   \frac{\pi}{2V\alpha^2} 2 N_e \sum_{I=1}^{N_{ion}} Z_I

    .. math:: E_{\rm charged}^{\text{ion-ion}} = - \frac{\pi}{2V\alpha^2} \left[ \sum_{I=1}^{N_{ion}} Z_I^2 + 2 \sum_{I<J}^{N_{ion}} Z_I Z_J \right]

    """

    def __init__(self, cell, ewald_gmax=200, nlatvec=1):
        """
        :parameter cell: pyscf Cell object (simulation cell)
        :parameter int ewald_gmax: how far to take reciprocal sum; probably never needs to be changed.
        :parameter int nlatvec: how far to take real-space sum; probably never needs to be changed.
        """
        self.nelec = np.array(cell.nelec)
        self.atom_coords = cell.atom_coords()
        self.atom_charges = gpu.cp.asarray(cell.atom_charges())
        self.latvec = cell.lattice_vectors()
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        """
        Generates list of lattice-vector displacements to add together for real-space sum

        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        """
        XYZ = np.meshgrid(*[np.arange(-nlatvec, nlatvec + 1)] * 3, indexing="ij")
        xyz = np.stack(XYZ, axis=-1).reshape((-1, 3))
        self.lattice_displacements = gpu.cp.asarray(np.dot(xyz, self.latvec))

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        r"""
        Determine parameters for Ewald sums.

        :math:`\alpha` determines the partitioning of the real and reciprocal-space parts.

        We define a weight `gweight` for the part of the reciprocal-space sums that doesn't depend on the coordinates:

        .. math:: W_G = \frac{4\pi}{V |\vec{G}|^2} e^{- \frac{|\vec{G}|^2}{ 4\alpha^2}}

        :parameter int ewald_gmax: max number of reciprocal lattice vectors to check away from 0
        """
        cellvolume = np.linalg.det(self.latvec)
        recvec = np.linalg.inv(self.latvec).T

        # Determine alpha
        smallestheight = np.amin(1 / np.linalg.norm(recvec, axis=1))
        self.alpha = 5.0 / smallestheight

        # Determine G points to include in reciprocal Ewald sum
        gptsXpos = gpu.cp.meshgrid(
            gpu.cp.arange(1, ewald_gmax + 1),
            *[gpu.cp.arange(-ewald_gmax, ewald_gmax + 1)] * 2,
            indexing="ij"
        )
        zero = gpu.cp.asarray([0])
        gptsX0Ypos = gpu.cp.meshgrid(
            zero,
            gpu.cp.arange(1, ewald_gmax + 1),
            gpu.cp.arange(-ewald_gmax, ewald_gmax + 1),
            indexing="ij",
        )
        gptsX0Y0Zpos = gpu.cp.meshgrid(
            zero, zero, gpu.cp.arange(1, ewald_gmax + 1), indexing="ij"
        )
        gs = zip(
            *[
                select_big(x, cellvolume, recvec, self.alpha)
                for x in (gptsXpos, gptsX0Ypos, gptsX0Y0Zpos)
            ]
        )
        self.gpoints, self.gweight = [gpu.cp.concatenate(x, axis=0) for x in gs]
        self.set_ewald_constants(cellvolume)

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
        ii_sum2 = np.sum(self.atom_charges**2)
        ii_sum = (self.i_sum**2 - ii_sum2) / 2

        self.ijconst = -np.pi / (cellvolume * self.alpha**2)
        self.squareconst = -self.alpha / np.sqrt(np.pi) + self.ijconst / 2

        self.ii_const = ii_sum * self.ijconst + ii_sum2 * self.squareconst
        self.e_single_test = -self.i_sum * self.ijconst + self.squareconst
        self.ion_ion = self.ewald_ion()

        # XC correction not used, so we can compare to other codes
        # rs = lambda ne: (3 / (4 * np.pi) / (ne * cellvolume)) ** (1 / 3)
        # cexc = 0.36
        # xc_correction = lambda ne: cexc / rs(ne)

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
            dist = pyqmc.distance.MinimalImageDistance(self.latvec)
            ion_distances, ion_inds = dist.dist_matrix(self.atom_coords[np.newaxis])
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
        ion_ion_rec = gpu.cp.dot(self.gweight, gpu.cp.abs(self.ion_exp) ** 2)

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
        r"""
        Compute the Ewald sum for e-e and e-ion

        Note: We ignore the constant term :math:`\frac{N_e}{2} C_{\rm self\ image}` in the real-space e-e sum corresponding to the interaction of an electron with its own image in other cells.

        Real space e-e:

        .. math:: E_{\rm real\ space}^{ee} = \sum_{\vec{n}} \sum_{i<j}^{N_e} \frac{{\rm erfc}(\alpha r_{ijn})}{r_{ijn}}

        Real space e-i:

        .. math:: E_{\rm real\ space}^{e\text{-ion}} = {\sum_{\vec{n}}} \sum_{i=1}^{N_e} \sum_{I=1}^{N_{ion}} -Z_I \frac{{\rm erfc}(\alpha r_{iIn})}{r_{iIn}}

        Reciprocal space e-e:

        .. math:: E_{\rm reciprocal\ space}^{ee} = \sum_{\vec{G}>0} W_G \left| \sum_{i=1}^{N_e} e^{-i\vec{k}\cdot\vec{x}_i} \right|^2

        Reciprocal space e-i:

        .. math:: E_{\rm reciprocal\ space}^{e\text{-ion}} = \sum_{\vec{G}>0} W_G {\rm Re} \left[ 2 \sum_{i=1}^{N_e} \sum_{I=1}^{N_{ion}} -Z_I e^{-i\vec{k}\cdot\vec{x}_i} e^{i\vec{k}\cdot\vec{x}_I} \right]

        :parameter configs: electron positions (walkers)
        :type configs: (nconf, nelec, 3) PeriodicConfigs object
        :returns:
            * ee: electron-electron part
            * ei: electron-ion part
        :rtype: float, float
        """
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

            for ((i, j), val) in zip(ee_inds, ee_cij.T):
                ee_real_separated[:, i] += val
                ee_real_separated[:, j] += val
            ee_real_separated /= 2

        ee_recip, ei_recip = self.reciprocal_space_electron(configs.configs)
        ee = ee_real_separated.sum(axis=1) + ee_recip
        ei = ei_real_separated.sum(axis=1) + ei_recip
        return ee, ei

    def reciprocal_space_electron(self, configs):
        # Reciprocal space electron-electron part
        e_GdotR = gpu.cp.einsum("hik,jk->hij", gpu.cp.asarray(configs), self.gpoints)
        sum_e_sin = gpu.cp.sin(e_GdotR).sum(axis=1)
        sum_e_cos = gpu.cp.cos(e_GdotR).sum(axis=1)
        ee_recip = gpu.cp.dot(sum_e_sin**2 + sum_e_cos**2, self.gweight)
        ## Reciprocal space electron-ion part
        coscos_sinsin = -self.ion_exp.real * sum_e_cos - self.ion_exp.imag * sum_e_sin
        ei_recip = 2 * gpu.cp.dot(coscos_sinsin, self.gweight)
        return ee_recip, ei_recip

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


def select_big(gpts, cellvolume, recvec, alpha):
    gpoints = gpu.cp.einsum(
        "j...,jk->...k", gpu.cp.asarray(gpts), gpu.cp.asarray(recvec) * 2 * np.pi
    )
    gsquared = gpu.cp.einsum("...k,...k->...", gpoints, gpoints)
    gweight = 4 * np.pi * gpu.cp.exp(-gsquared / (4 * alpha**2))
    gweight /= cellvolume * gsquared
    bigweight = gweight > 1e-10
    return gpoints[bigweight], gweight[bigweight]
