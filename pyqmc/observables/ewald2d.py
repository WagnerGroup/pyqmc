# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
import pyqmc
from pyqmc.configurations.coord import PeriodicConfigs
import pyqmc.observables.energy
import pyqmc.gpu as gpu
from pyscf.pbc.gto.cell import Cell
from typing import Tuple
import pyqmc.observables.ewald as ewald


class Ewald:
    """
    Evaluate the Ewald summation using the 2D formula
    [Yeh and Berkowitz. J. Chem. Phys. 111, 3155–3162 (1999)]
    https://doi.org/10.1063/1.479595
    """

    def __init__(
        self, cell: Cell, gmax: int = 200, nlatvec: int = 1, alpha_scaling: float = 5.0
    ):
        """
        :parameter pyscf.pbc.gto.cell.Cell cell: PySCF Cell object
        :parameter int gmax: max number of reciprocal lattice vectors to check away from 0
        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        :parameter float alpha_scaling: scaling factor for partitioning the real-space and reciprocal-space parts.
        """
        self.latvec = cell.lattice_vectors()
        self.atom_coords = cell.atom_coords()[np.newaxis]
        self.atom_charges = gpu.cp.asarray(cell.atom_charges())
        self.dist = pyqmc.configurations.distance.MinimalImageDistance(self.latvec)
        self.cell_area = gpu.cp.linalg.det(self.latvec[:2, :2])
        self.recvec = gpu.cp.linalg.inv(self.latvec).T
        self.set_alpha(alpha_scaling)
        self.set_lattice_displacements(nlatvec)
        self.set_gpoints(gmax)
        self.set_ewald_ion_ion()

    def set_alpha(self, alpha_scaling: float):
        """
        Define the partitioning of the real and reciprocal-space parts.
        """
        smallest_height = gpu.cp.amin(
            1 / gpu.cp.linalg.norm(self.recvec[:2, :2], axis=1)
        )
        self.alpha = alpha_scaling / smallest_height

    def set_lattice_displacements(self, nlatvec: int):
        """
        Define a list of lattice-vector displacements to add together for real-space sum.

        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        """
        space = [gpu.cp.arange(-nlatvec, nlatvec + 1)] * 2
        XYZ = gpu.cp.meshgrid(*space, indexing="ij")
        xyz = gpu.cp.stack(XYZ, axis=-1).reshape((-1, 2))
        z_zeros = gpu.cp.zeros((xyz.shape[0], 1))
        xyz = gpu.cp.concatenate([xyz, z_zeros], axis=1)
        self.lattice_displacements = gpu.cp.asarray(gpu.cp.dot(xyz, self.latvec))

    def generate_positive_gpoints(self, gmax: int) -> gpu.cp.ndarray:
        """
        Generate a list of points in the reciprocal space to add together for reciprocal-space sum.

        :parameter gmax: max number of reciprocal lattice vectors to check away from 0
        :return: reciprocal-space points (nk, 3)
        """
        gXpos = gpu.cp.mgrid[1 : gmax + 1, -gmax : gmax + 1, 0:1].reshape(3, -1)
        gX0Ypos = gpu.cp.mgrid[0:1, 1 : gmax + 1, 0:1].reshape(3, -1)
        gpts = gpu.cp.concatenate([gXpos, gX0Ypos], axis=1)
        gpoints = gpu.cp.einsum(
            "ji,jk->ik", gpts, gpu.cp.asarray(self.recvec) * 2 * gpu.cp.pi
        )
        return gpoints

    def set_gpoints(self, gmax: int, tol: float = 1e-10):
        """
        Define reciprocal-lattice points with large contributions according to `tol`.

        :parameter gmax: max number of reciprocal lattice vectors to check away from 0
        :parameter tol: tolerance for the cutoff weight
        """
        candidate_gpoints = self.generate_positive_gpoints(gmax)
        gnorm = gpu.cp.linalg.norm(candidate_gpoints, axis=-1)
        gweight = gpu.cp.pi * gpu.erfc(gnorm / (2 * self.alpha)) * 2
        gweight /= self.cell_area * gnorm
        mask_bigweight = gweight > tol
        self.gpoints = candidate_gpoints[mask_bigweight]
        self.gweight = gweight[mask_bigweight]
        self.gnorm = gnorm[mask_bigweight]
        self.sum_gweight = gpu.cp.sum(self.gweight)

    def set_ewald_ion_ion(self):
        r"""
        real space sum:

        .. math:: E_{\textrm{real,cross}}^{\textrm{ion-ion}} = \sum_{I=1}^{N_{\textrm{ion}}} \sum_{J > I}^{N_{\textrm{ion}}} q_I q_J W_{\textrm{real}}(\mathbf{r}_{IJ}).

        reciprocal-space sum:

        .. math:: E_{\textrm{recip},k > 0}^{\textrm{ion-ion}}
            = \sum_{\mathbf{k} > 0} \sum_{I=1}^{N_{\textrm{ion}}} \sum_{J=1}^{N_{\textrm{ion}}} q_I q_J \mathrm{e}^{i \mathbf{k} \cdot \mathbf{r}_{IJ}} W(k, z_{IJ}).

        sum of the charge terms:

        .. math:: E_{\textrm{recip},k = 0}^{\textrm{ion-ion}}
            = \sum_{I=1}^{N_{\textrm{ion}}} \sum_{J=1}^{N_{\textrm{ion}}} q_I q_J W(z_{IJ}).
        :returns: ion-ion component of Ewald sum
        """
        sum_charges2 = np.sum(self.atom_charges**2)
        if len(self.atom_charges) == 1:
            self.ewald_ion_ion = self.ewald_self(sum_charges2)
            return
        # real cross term
        # input to dist_matrix has the shape (nconf, natoms, ndim)
        ion_ion_dist, ion_ion_idxs = self.dist.dist_matrix(self.atom_coords)
        ion_ion_cij = ewald.real_cij(
            ion_ion_dist, self.lattice_displacements, self.alpha
        )  # (nconf, npairs=choose(natoms, 2))
        pair_charges = self.atom_charges[gpu.cp.asarray(ion_ion_idxs)]
        ion_ion_charge_ij = gpu.cp.prod(pair_charges, axis=1)  # (npairs,)
        ion_ion_real_cross = gpu.cp.einsum(
            "j,ij->i", ion_ion_charge_ij, ion_ion_cij
        )  # (nconf,)

        # reciprocal term
        g_dot_r = gpu.cp.einsum(
            "kd,ijd->ijk", self.gpoints, ion_ion_dist
        )  # (1, npairs, nk)
        gweight = self.ewald_recip_weight(ion_ion_dist)  # (1, npairs, nk)
        ion_ion_recip = (
            2
            * gpu.cp.einsum(
                "j,ijk,ijk->", ion_ion_charge_ij, gpu.cp.exp(1j * g_dot_r), gweight
            ).real
        )

        # charge term
        weight = self.ewald_recip_weight_charge(ion_ion_dist)
        ion_ion_charge = 2 * gpu.cp.einsum("j,ij->", ion_ion_charge_ij, weight)

        self.ewald_ion_ion = (
            ion_ion_real_cross
            + ion_ion_recip
            + ion_ion_charge
            + self.ewald_self(sum_charges2)
        )

    def ewald_elec_ion(self, configs: PeriodicConfigs) -> float:
        r"""
        real space sum:

        .. math:: E_{\textrm{real,cross}}^{\textrm{e-ion}} = \sum_{i=1}^{N_{\textrm{e}}} \sum_{I=1}^{N_{\textrm{ion}}} (-1) q_I W_{\textrm{real}}(\mathbf{r}_{iI}).

        reciprocal-space sum:

        .. math:: E_{\textrm{recip},k > 0}^{\textrm{e-ion}}
            = \sum_{\mathbf{k} > 0} \sum_{i=1}^{N_{\textrm{e}}} \sum_{I=1}^{N_{\textrm{ion}}} (-2 q_I) \mathrm{e}^{i \mathbf{k} \cdot \mathbf{r}_{iI}} W(k, z_{iI}).

        charge terms:

        .. math:: E_{\textrm{recip},k = 0}^{\textrm{e-ion}}
            = \sum_{i=1}^{N_{\textrm{e}}} \sum_{I=1}^{N_{\textrm{ion}}} (-2 q_I) W(z_{iI}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :returns: electron-ion real-space cross-term component of Ewald sum
        """
        # real term
        ei_dist = configs.dist.pairwise(
            self.atom_coords, configs.configs
        )  # (nelec, nconf, natoms, ndim)
        ei_cij = ewald.real_cij(
            ei_dist, self.lattice_displacements, self.alpha
        )  # (nconf, nelec, natoms)
        ei_real = gpu.cp.einsum("k,ijk->i", -self.atom_charges, ei_cij)  # (nconf,)

        # reciprocal term
        g_dot_r = gpu.cp.einsum(
            "kd,cijd->cijk", self.gpoints, ei_dist
        )  # (nconf, natoms, nelec, nk)
        gweight = self.ewald_recip_weight(ei_dist)  # (nconf, natoms, nelec, nk)
        ei_recip = -2 * gpu.cp.einsum(
            "i,cijk,cijk->c", self.atom_charges, gpu.cp.cos(g_dot_r), gweight
        )

        # charge term
        weight = self.ewald_recip_weight_charge(ei_dist)
        ei_charge = -2 * gpu.cp.einsum("i,cij->c", self.atom_charges, weight)

        return ei_real + ei_recip + ei_charge

    def ewald_elec_elec(self, configs: PeriodicConfigs) -> float:
        r"""
        real space sum:

        .. math:: E_{\textrm{real,cross}}^{\textrm{e-e}} = \sum_{i=1}^{N_{\textrm{e}}} \sum_{j > i}^{N_{\textrm{e}}} W_{\textrm{real}}(\mathbf{r}_{ij}).

        reciprocal-space sum:

        .. math:: E_{\textrm{recip},k > 0}^{\textrm{e-e}}
            = \sum_{\mathbf{k} > 0} \sum_{i=1}^{N_{\textrm{e}}} \sum_{j=1}^{N_{\textrm{e}}} \mathrm{e}^{i \mathbf{k} \cdot \mathbf{r}_{ij}} W_{\textrm{recip}}(k, z_{ij}).

        charge terms:

        .. math:: E_{\textrm{recip},k = 0}^{\textrm{e-e}}
            = \sum_{i=1}^{N_{\textrm{e}}} \sum_{j=1}^{N_{\textrm{e}}} W_{\textrm{recip}}(z_{ij}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :returns: electron-electron component of Ewald sum
        """
        nconf, nelec, ndim = configs.configs.shape
        if nelec == 1:
            return self.ewald_self(nelec)
        # real cross term
        ee_dist, ee_idxs = configs.dist.dist_matrix(
            configs.configs
        )  # (nconf, npairs, ndim)
        ee_cij = ewald.real_cij(ee_dist, self.lattice_displacements, self.alpha)
        ee_real = gpu.cp.sum(ee_cij, axis=-1)

        # reciprocal term
        g_dot_r = gpu.cp.einsum(
            "kd,cid->cik", self.gpoints, ee_dist
        )  # (nconf, npairs, nk)
        gweight = self.ewald_recip_weight(ee_dist)  # (nconf, npairs, nk)
        ee_recip = (
            2 * gpu.cp.einsum("cik,cik->c", gpu.cp.exp(1j * g_dot_r), gweight).real
        )

        # charge term
        weight = self.ewald_recip_weight_charge(ee_dist)
        ee_charge = 2 * gpu.cp.sum(weight, axis=1)

        return ee_real + ee_recip + ee_charge + self.ewald_self(nelec)

    def ewald_self(self, sum_charge_squared):
        r"""
        Compute contributions to the ion-ion or electron-electron self terms

        .. math:: E_{\textrm{real,self}}^{\textrm{ion-ion}} = - \frac{\alpha}{\sqrt{\pi}} \sum_{I=1}^{N_{\textrm{ion}}} q_I^2. \\

        :parameter sum_charge_squared: sum of squared atom charges or nelec
        :return: self contribution
        """
        sqrt_pi = gpu.cp.sqrt(gpu.cp.pi)
        real_self = -self.alpha / sqrt_pi
        recip_self = self.sum_gweight
        charge_self = -sqrt_pi / (self.cell_area * self.alpha)
        return (real_self + recip_self + charge_self) * sum_charge_squared

    def ewald_recip_weight(self, dist: gpu.cp.ndarray) -> gpu.cp.ndarray:
        r"""
        Compute the weight for the reciprocal-space sum

        .. math:: W_{\textrm{recip},k>0}(k, z_{mn}) = \frac{\pi}{A k}
                \left[
                    \mathrm{e}^{k z_{mn}} \textrm{erfc} \left(\alpha z_{mn} + \frac{k}{2 \alpha} \right)
                    + \mathrm{e}^{-k z_{mn}} \textrm{erfc} \left(-\alpha z_{mn} + \frac{k}{2 \alpha} \right)
                \right].

        :parameter dist: distance matrix.
            Shape: (num_particles_m, num_particles_n, 3) or (nconf, num_particles_m, num_particles_n, 3)
        :return: weight for reciprocal-space sum when k > 0. Shape: (nk, num_particles_m, num_particles_n) or (nconf, nk, num_particles_m, num_particles_n)
        """
        z = dist[..., 2][..., np.newaxis]
        gnorm = self.gnorm
        alpha = self.alpha
        w1 = gpu.cp.exp(gnorm * z) * gpu.erfc(gnorm / (2 * alpha) + alpha * z)
        w2 = gpu.cp.exp(-gnorm * z) * gpu.erfc(gnorm / (2 * alpha) - alpha * z)
        gweight = gpu.cp.pi / (self.cell_area * gnorm) * (w1 + w2)
        return gweight

    def ewald_recip_weight_charge(self, dist: gpu.cp.ndarray) -> gpu.cp.ndarray:
        r"""
        Compute the weight for the charge terms (k = 0 terms) in reciprocal space sum

        .. math:: W_{\textrm{recip,k=0}}(z_{mn}) = - \frac{\pi}{A} \left[z_{mn} \textrm{erf}(\alpha z_{mn}) + \frac{1}{\alpha \sqrt{\pi}} \exp(-\alpha^2 z_{mn}^2) \right].

        :parameter dist: distance matrix.
            Shape: (num_particles_m, num_particles_n, 3) or (nconf, num_particles_m, num_particles_n, 3)
        :return: weight for reciprocal-space sum when k = 0. Shape: (num_particles_m, num_particles_n) or (nconf, num_particles_m, num_particles_n)
        """
        z = dist[..., 2]  # ([nconf,] npairs)
        w1 = z * gpu.erf(self.alpha * z)
        w2 = (
            1
            / (self.alpha * gpu.cp.sqrt(gpu.cp.pi))
            * gpu.cp.exp(-self.alpha**2 * z**2)
        )
        w = -gpu.cp.pi / self.cell_area * (w1 + w2)
        return w

    def energy(self, configs: PeriodicConfigs) -> Tuple[float, float, float]:
        r"""
        Compute Coulomb energy for a set of configs.

        .. math:: E &= E^{\textrm{e-e}} + E^{\textrm{e-ion}} + E^{\textrm{ion-ion}} \\
            E^{\textrm{e-e}} &= E_{\textrm{real,cross}}^{\textrm{e-e}} + E_{\textrm{real,self}}^{\textrm{e-e}}
                + E_{\textrm{recip},k>0}^{\textrm{e-e}} + E_{\textrm{recip},k=0}^{\textrm{e-e}} \\
            E^{\textrm{e-ion}} &= E_{\textrm{real,cross}}^{\textrm{e-ion}}
                + E_{\textrm{recip},k>0}^{\textrm{e-ion}} + E_{\textrm{recip},k=0}^{\textrm{e-ion}} \\
            E^{\textrm{ion-ion}} &= E_{\textrm{real,cross}}^{\textrm{ion-ion}} + E_{\textrm{real,self}}^{\textrm{ion-ion}}
                + E_{\textrm{recip},k>0}^{\textrm{ion-ion}} + E_{\textrm{recip},k=0}^{\textrm{ion-ion}}

        :parameter configs: Shape: (nconf, nelec, 3)
        :return:
            * ee: electron-electron part
            * ei: electron-ion part
            * ii: ion-ion part
        """
        ii = self.ewald_ion_ion
        ee = self.ewald_elec_elec(configs)
        ei = self.ewald_elec_ion(configs)
        return ee, ei, ii
