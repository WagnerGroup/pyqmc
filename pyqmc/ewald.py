import numpy as np
import pyqmc
from scipy.special import erfc


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

    def __init__(self, cell, ewald_gmax=200, nlatvec=2):
        """
        Inputs:
            cell: pyscf Cell object (simulation cell)
            ewald_gmax: int, how far to take reciprocal sum; probably never needs to be changed.
            nlatvec: int, how far to take real-space sum; probably never needs to be changed.
        """
        self.nelec = np.array(cell.nelec)
        self.atom_coords, self.atom_charges = cell.atom_coords(), cell.atom_charges()
        self.latvec = cell.lattice_vectors()
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        """
        Generates list of lattice-vector displacements to add together for real-space sum, going from `-nlatvec` to `nlatvec` in each lattice direction.
        """
        XYZ = np.meshgrid(*[np.arange(-nlatvec, nlatvec + 1)] * 3, indexing="ij")
        xyz = np.stack(XYZ, axis=-1).reshape((-1, 3))
        self.lattice_displacements = np.dot(xyz, self.latvec)

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        r"""
        Determine parameters for Ewald sums. 

        :math:`\alpha` determines the partitioning of the real and reciprocal-space parts.

        We define a weight `gweight` for the part of the reciprocal-space sums that doesn't depend on the coordinates:
        
        .. math:: W_G = \frac{4\pi}{V |\vec{G}|^2} e^{- \frac{|\vec{G}|^2}{ 4\alpha^2}}

        Inputs:
            latvec: (3, 3) array of lattice vectors; latvec[0] is the first
            ewald_gmax: int, max number of reciprocal lattice vectors to check away from 0
        """
        cellvolume = np.linalg.det(self.latvec)
        recvec = np.linalg.inv(self.latvec)
        crossproduct = recvec.T * cellvolume

        # Determine alpha
        tmpheight_i = np.einsum("ij,ij->i", crossproduct, self.latvec)
        length_i = np.linalg.norm(crossproduct, axis=1)
        smallestheight = np.amin(np.abs(tmpheight_i) / length_i)
        self.alpha = 5.0 / smallestheight
        print("Setting Ewald alpha to ", self.alpha)

        # Determine G points to include in reciprocal Ewald sum
        XYZ = np.meshgrid(*[np.arange(-ewald_gmax, ewald_gmax + 1)] * 3, indexing="ij")
        X, Y, Z = [x.ravel() for x in XYZ]
        positive_octants = X + 1e-6 * Y + 1e-12 * Z > 0  # assume ewald_gmax < 1e5
        gpoints = np.stack((X, Y, Z), axis=-1)[positive_octants]
        gpoints = np.dot(gpoints, recvec) * 2 * np.pi
        gsquared = np.sum(gpoints ** 2, axis=1)
        gweight = 4 * np.pi * np.exp(-gsquared / (4 * self.alpha ** 2))
        gweight /= cellvolume * gsquared
        bigweight = gweight > 1e-10
        self.gpoints = gpoints[bigweight]
        self.gweight = gweight[bigweight]

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

        .. math:: E_{\rm self+charged}^{e\text{-ion}} = N_e \sum_{I=1}^{N_{ion}} Z_I C_{ij}

        .. math:: E_{\rm self+charged}^{\text{ion-ion}} = \sum_{I=1}^{N_{ion}} Z_I^2 C_{\rm square} + \sum_{I<J}^{N_{ion}} Z_I Z_J C_{ij}

        We also compute contributions from a single electron, to separate the Ewald sum by electron.
        
        .. math:: E_{\rm self+charged}^{\rm single} = C_{\rm square} + \frac{N_e-1}{2} C_{ij} - \sum_{I=1}^{N_{ion}} Z_I C_{ij}

        .. math:: E_{\rm self+charged}^{\text{single-test}} = C_{\rm square} - \sum_{I=1}^{N_{ion}} Z_I C_{ij}

        """
        i_sum = np.sum(self.atom_charges)
        ii_sum2 = np.sum(self.atom_charges ** 2)
        ii_sum = (i_sum ** 2 - ii_sum2) / 2

        ijconst = -np.pi / (cellvolume * self.alpha ** 2)
        self.ijconst = ijconst
        squareconst = -self.alpha / np.sqrt(np.pi) + ijconst / 2

        self.ii_const = ii_sum * ijconst + ii_sum2 * squareconst
        self.ee_const = lambda ne: ne * (ne - 1) / 2 * ijconst + ne * squareconst
        self.ei_const = lambda ne: -ne * i_sum * ijconst

        self.e_single = lambda ne: (ne - 1) * ijconst - i_sum * ijconst + squareconst
        self.e_single_test = -i_sum * ijconst + squareconst
        self.ion_ion = self.ewald_ion()

        # XC correction not used, so we can compare to other codes
        rs = lambda ne: (3 / (4 * np.pi) / (ne * cellvolume)) ** (1 / 3)
        cexc = 0.36
        xc_correction = lambda ne: cexc / rs(ne)

    def ewald_ion(self):
        r"""
        Compute ion contribution to Ewald sums.  Since the ions don't move in our calculations, the ion-ion term only needs to be computed once.

        Note: We ignore the constant term :math:`\frac{1}{2} \sum_{I} Z_I^2 C_{\rm self\ image}` in the real-space ion-ion sum corresponding to the interaction of an ion with its own image in other cells.

        The real-space part:

        .. math:: E_{\rm real\ space}^{\text{ion-ion}} = \sum_{\vec{n}} \sum_{I<J}^{N_{ion}} Z_I Z_J \frac{{\rm erfc}(\alpha |\vec{x}_{IJ}+\vec{n}|)}{|\vec{x}_{IJ}+\vec{n}|} 

        The reciprocal-space part:

        .. math:: E_{\rm reciprocal\ space}^{\text{ion-ion}} = \sum_{\vec{G} > 0 } W_G \left| \sum_{I=1}^{N_{ion}} Z_I e^{-i\vec{G}\cdot\vec{x}_I} \right|^2

        Returns:
            ion_ion: float, ion-ion component of Ewald sum
        """
        # Real space part
        if len(self.atom_charges) == 1:
            ion_ion_real = 0
        else:
            dist = pyqmc.distance.MinimalImageDistance(self.latvec)
            ion_distances, ion_inds = dist.dist_matrix(self.atom_coords[np.newaxis])
            rvec = ion_distances[:, :, np.newaxis, :] + self.lattice_displacements
            r = np.linalg.norm(rvec, axis=-1)
            charge_ij = np.prod(self.atom_charges[np.asarray(ion_inds)], axis=1)
            ion_ion_real = np.einsum("j,ijk->", charge_ij, erfc(self.alpha * r) / r)

        # Reciprocal space part
        GdotR = np.dot(self.gpoints, self.atom_coords.T)
        self.ion_exp = np.dot(np.exp(1j * GdotR), self.atom_charges)
        ion_ion_rec = np.dot(self.gweight, np.abs(self.ion_exp) ** 2)

        ion_ion = ion_ion_real + ion_ion_rec
        return ion_ion

    def _real_cij(self, dists):
        r = np.zeros(dists.shape[:-1])
        cij = np.zeros(r.shape)
        for ld in self.lattice_displacements:
            r[:] = np.linalg.norm(dists + ld, axis=-1)
            cij += erfc(self.alpha * r) / r
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

        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
        Returns:
            ee: electron-electron part
            ei: electron-ion part
        """
        nconf, nelec, ndim = configs.configs.shape

        # Real space electron-ion part
        # ei_distances shape (elec, conf, atom, dim)
        ei_distances = configs.dist.dist_i(self.atom_coords, configs.configs)
        ei_cij = self._real_cij(ei_distances)
        ei_real_separated = np.einsum("k,ijk->ji", -self.atom_charges, ei_cij)

        # Real space electron-electron part
        ee_real_separated = np.zeros((nconf, nelec))
        if nelec > 1:
            ee_distances, ee_inds = configs.dist.dist_matrix(configs.configs)
            ee_cij = self._real_cij(ee_distances)

            for ((i, j), val) in zip(ee_inds, ee_cij.T):
                ee_real_separated[:, i] += val
                ee_real_separated[:, j] += val
            ee_real_separated /= 2

        # Reciprocal space electron-electron part
        e_GdotR = np.dot(configs.configs, self.gpoints.T)
        e_expGdotR = np.exp(1j * e_GdotR)
        sum_e_exp = np.sum(e_expGdotR, axis=1, keepdims=True)
        coscos_sinsin = np.real(sum_e_exp.conj() * e_expGdotR)
        ### Don't know why we subtract 0.5 for "separated"
        ee_recip_separated = np.dot(coscos_sinsin - 0.5, self.gweight)

        # Reciprocal space electron-ion part
        coscos_sinsin = np.real(-self.ion_exp.conj() * e_expGdotR)
        ei_recip_separated = np.dot(coscos_sinsin, self.gweight)

        # Combine parts
        self.ei_separated = ei_real_separated + 2 * ei_recip_separated
        self.ee_separated = ee_real_separated + 1 * ee_recip_separated
        self.ewalde_separated = self.ei_separated + self.ee_separated
        nelec = ee_recip_separated.shape[1]
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
        
        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
        Returns: 
            ee: electron-electron part
            ei: electron-ion part
            ii: ion-ion part
        """
        nelec = configs.configs.shape[1]
        ee, ei = self.ewald_electron(configs)
        ee += self.ee_const(nelec)
        ei += self.ei_const(nelec)
        ii = self.ion_ion + self.ii_const
        return ee, ei, ii

    def energy_separated(self, configs):
        """
        Compute Coulomb energy contribution from each electron (does not include ion-ion energy).

        NOTE: energy() needs to be called first to update the separated energy values

        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
        Returns: 
            (nelec,) energies
        """
        nelec = configs.configs.shape[1]
        return self.e_single(nelec) + self.ewalde_separated

    def energy_with_test_pos(self, configs, epos):
        """
        Compute Coulomb energy of an additional test electron with a set of configs

        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
            epos: pyqmc PeriodicConfigs object of shape (nconf, ndim)
        Returns: 
            Vtest: (nconf, nelec+1) array. The first nelec columns are Coulomb energies between the test electron and each electron; the last column is the contribution from all the ions.
        """
        nconf, nelec, ndim = configs.configs.shape
        Vtest = np.zeros((nconf, nelec + 1)) + self.ijconst
        Vtest[:, -1] = self.e_single_test

        # Real space electron-ion part
        # ei_distances shape (conf, atom, dim)
        ei_distances = configs.dist.dist_i(self.atom_coords, epos.configs)
        rvec = ei_distances[:, :, np.newaxis, :] + self.lattice_displacements
        r = np.linalg.norm(rvec, axis=-1)
        Vtest[:, -1] += np.einsum(
            "k,jkl->j", -self.atom_charges, erfc(self.alpha * r) / r
        )

        # Real space electron-electron part
        ee_distances = configs.dist.dist_i(configs.configs, epos.configs)
        rvec = ee_distances[:, :, np.newaxis, :] + self.lattice_displacements
        r = np.linalg.norm(rvec, axis=-1)
        Vtest[:, :-1] += np.sum(erfc(self.alpha * r) / r, axis=-1)

        # Reciprocal space electron-electron part
        e_expGdotR = np.exp(1j * np.dot(configs.configs, self.gpoints.T))
        test_exp = np.exp(1j * np.dot(epos.configs, self.gpoints.T))
        ee_recip_separated = np.dot(np.real(test_exp.conj() * e_expGdotR), self.gweight)
        Vtest[:, :-1] += 2 * ee_recip_separated

        # Reciprocal space electrin-ion part
        coscos_sinsin = np.real(-self.ion_exp.conj() * test_exp)
        ei_recip_separated = np.dot(coscos_sinsin + 0.5, self.gweight)
        Vtest[:, -1] += 2 * ei_recip_separated

        return Vtest
