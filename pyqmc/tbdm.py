""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import copy, deepcopy
from pyqmc.mc import initial_guess
from pyqmc.obdm import sample_onebody
from sys import stdout

class TBDMAccumulator:
    """ Returns the different spin sectors of the tbdm[s1,s2] as an array (norb_s1,norb_s1,norb_s2,norb_s2) with indices (using pySCF's 
    convention): tbdm[s1,s2][i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s1,j} c_{s2,l} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s1,i} \phi_{s2,k}.

    Args:

      mol (Mole): PySCF Mole object.

      orb_coeff (array): coefficients with size (2,nbasis,norb) relating mol basis to basis
        of the 2-RDM.

      nsweeps (int): number of sweeps over all the electron pairs.

      tstep (float): width of the Gaussian to update a walker position for each extra coordinate.

      warmup (int): number of warmup steps for single-particle local orbital sampling.

      naux (int): number of auxiliary configurations for extra moves sampling the local
         orbitals.  

      spin (array): with size (num_spin_sectors,2) contains spin sectors to be computed;
         can include [0,0], [0,1], [1,0] and [1,1] for up-up, up-down, down-up or down-down. 
         Defaults to all four sectors.

      ijkl (array): contains M tbdm matrix elements to calculate with dim (M,4). 
    """

    def __init__(
        self,
        mol,
        orb_coeff,
        nsweeps=1,
        tstep=0.50,
        warmup=200,
        naux=500,
        spin_sectors=None,
        ijkl=None,
    ):
        assert (
            len(orb_coeff.shape) == 3
        ), "orb_coeff should be a list of orbital coefficients with size (2,num_mobasis,num_orb)."

        self._mol = mol
        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._nsweeps = nsweeps

        if not (spin_sectors is None):
            self._spin_sectors = np.array(spin_sectors)
            for i,sector in enumerate(spin_sectors):
                electrons_a = np.arange(sector[0]*mol.nelec[0], mol.nelec[0]+sector[0]*mol.nelec[1])
                electrons_b = np.arange(sector[1]*mol.nelec[0], mol.nelec[0]+sector[1]*mol.nelec[1])
                if i==0:
                    self._pairs = np.array(np.meshgrid( electrons_a, electrons_b)).T.reshape(-1,2)
                else:
                    self._pairs = np.concatenate([ self._pairs, np.array(np.meshgrid( electrons_a, electrons_b)).T.reshape(-1,2) ])                
            self._pairs = self._pairs[self._pairs[:,0]!=self._pairs[:,1]] # Removes repeated electron pairs
        else:
            self._spin_sectors = np.stack(np.indices((2,2)).reshape(2,-1).T)
            self._pairs = np.array(np.meshgrid( np.arange(0, np.sum(mol.nelec)), np.arange(0, np.sum(mol.nelec)) )).T.reshape(-1,2)
            self._pairs = self._pairs[self._pairs[:,0]!=self._pairs[:,1]] # Removes repeated electron pairs

        # Initialization and warmup of aux_configs_up
        if 0 in self._spin_sectors:
            self._aux_configs_up = initial_guess(mol, int(naux/sum(self._mol.nelec))).configs.reshape(-1, 3)
            for i in range(warmup):
                accept_up, self._aux_configs_up = sample_onebody(
                    mol, orb_coeff[0], self._aux_configs_up, tstep
                )
        # Initialization and warmup of aux_configs_down
        if 1 in self._spin_sectors:
            self._aux_configs_down = initial_guess(mol, int(naux/sum(self._mol.nelec))).configs.reshape(-1, 3)
            for i in range(warmup):
                accept_down, self._aux_configs_down = sample_onebody(
                    mol, orb_coeff[1], self._aux_configs_down, tstep
                )


            
    def __call__(self, configs, wf, extra_configs=None, auxassignments=None):
        """Gathers quantities from equation (10) of DOI:10.1063/1.4793531."""

        # Constructs results dictionary
        spin_dic={0:'up',1:'down'}
        nconf = configs.configs.shape[0]
        results={}
        for sector in self._spin_sectors:
            orb_a_size = self._orb_coeff[sector[0]].shape[1]
            orb_b_size = self._orb_coeff[sector[1]].shape[1]
            results["value_%s%s"%(spin_dic[sector[0]],spin_dic[sector[1]])] = np.zeros(
                (nconf, orb_a_size, orb_a_size, orb_b_size, orb_b_size)
            )
        for s in np.unique(self._spin_sectors):
            results["norm_%s"%spin_dic[s]] = np.zeros((nconf, self._orb_coeff[s].shape[1]))
            results["acceptance_%s"%spin_dic[s]] = np.zeros(nconf)
  
        if extra_configs is None:
            # Generates aux_configs_up
            aux_configs_up = []
            if 0 in self._spin_sectors:
                for step in range(self._nsweeps * len(self._pairs)):
                    aux_configs_up.append(np.copy(self._aux_configs_up))
                    accept, self._aux_configs_up = sample_onebody(
                        self._mol, self._orb_coeff[0], self._aux_configs_up, tstep=self._tstep
                    )
                    results["acceptance_up"] += np.mean(accept)
                results["acceptance_up"] /= ( self._nsweeps * len(self._pairs) )
                aux_configs_up = np.concatenate(aux_configs_up,axis=0)
            # Generates aux_configs_down
            aux_configs_down = []
            if 1 in self._spin_sectors:
                for step in range(self._nsweeps * len(self._pairs)):
                    aux_configs_down.append(np.copy(self._aux_configs_down))
                    accept, self._aux_configs_down = sample_onebody(
                        self._mol, self._orb_coeff[1], self._aux_configs_down, tstep=self._tstep
                    )
                    results["acceptance_down"] += np.mean(accept)
                results["acceptance_down"] /= ( self._nsweeps * len(self._pairs) )
                aux_configs_down = np.concatenate(aux_configs_down,axis=0)

            # Generates random choice of aux_config_up and aux_config_down for moving electron_a and electron_b
            naux_up = self._aux_configs_up.shape[0]
            naux_down = self._aux_configs_down.shape[0]
            auxassignments_a = np.array([ np.random.randint(0, int(naux_up/2), size=(self._nsweeps*len(self._pairs), nconf)),
                                 np.random.randint(0, int(naux_down/2), size=(self._nsweeps*len(self._pairs), nconf)) ])
            auxassignments_b = np.array([ np.random.randint(int(naux_up/2), naux_up, size=(self._nsweeps*len(self._pairs), nconf)),
                                 np.random.randint(int(naux_down/2), naux_down, size=(self._nsweeps*len(self._pairs), nconf)) ])
        else:   
            assert auxassignments is not None
            aux_configs_up = extra_configs[0]
            aux_configs_dn = extra_configs[1]
            auxassignments_a = auxassignments[0]
            auxassignments_b = auxassignments[1]
            
        # Sweeps over electron pairs        
        for sweep in range(self._nsweeps):
            for i,pair in enumerate(self._pairs):
                naux = [ naux_up, naux_down]
                aux_configs = [ aux_configs_up[i*naux_up:(i+1)*naux_up], aux_configs_down[i*naux_down:(i+1)*naux_down] ]
                spin_a = int(pair[0]>=self._mol.nelec[0]) # electron_a's spin in this pair
                spin_b = int(pair[1]>=self._mol.nelec[0]) # electron_a's spin in this pair
                
                # Orbital evaluations at all coordinates
                points_a = np.concatenate([aux_configs[spin_a], configs.configs[:, pair[0], :]])
                points_b = np.concatenate([aux_configs[spin_b], configs.configs[:, pair[1], :]])
                ao_a = self._mol.eval_gto("GTOval_sph", points_a)
                ao_b = self._mol.eval_gto("GTOval_sph", points_b)
                orb_a = ao_a.dot(self._orb_coeff[spin_a])
                orb_b = ao_b.dot(self._orb_coeff[spin_b])
                
                # Constructs aux_orbitals, fsum and norm for electron_a and electron_b
                orb_a_aux = orb_a[0:naux[spin_a], :]
                orb_b_aux = orb_b[0:naux[spin_b], :]
                fsum_a = np.sum(orb_a_aux * orb_a_aux, axis=1)
                fsum_b = np.sum(orb_b_aux * orb_b_aux, axis=1)
                norm_a = orb_a_aux * orb_a_aux / fsum_a[:, np.newaxis]
                norm_b = orb_b_aux * orb_b_aux / fsum_b[:, np.newaxis]
                orb_a_configs = orb_a[naux[spin_a]:, :]
                orb_b_configs = orb_b[naux[spin_b]:, :]
           
                # Calculation of wf ratio (no McMillan trick yet)
                epos_a = configs.make_irreducible(
                    pair[0], aux_configs[spin_a][auxassignments_a[spin_a,i+sweep*len(self._pairs)]]
                )
                epos_a_orig = configs.electron(pair[0])
                epos_b = configs.make_irreducible(
                    pair[1], aux_configs[spin_b][auxassignments_b[spin_b,i+sweep*len(self._pairs)]]
                )
                wfratio_a = wf.testvalue(pair[0], epos_a)
                wf.updateinternals(pair[0], epos_a)
                wfratio_b = wf.testvalue(pair[1], epos_b)
                wfratio = wfratio_a * wfratio_b
                wf.updateinternals(pair[0], epos_a_orig)
            
                # We use pySCF's index convention (while Eq. 10 in DOI:10.1063/1.4793531 uses QWalk's)
                # QWalk -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,j} c_{s1,k} c_{s2,l} > = \phi*_{s1,k} \phi*_{s2,l} \phi_{s1,i} \phi_{s2,j}
                # pySCF -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s1,j} c_{s2,l} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s1,i} \phi_{s2,k}
                orbratio = np.einsum(
                    "mj,ml,mi,mk->mijkl",
                    orb_a_aux[auxassignments_a[spin_a,i+sweep*len(self._pairs)], :] / fsum_a[auxassignments_a[spin_a,i+sweep*len(self._pairs)], np.newaxis],
                    orb_b_aux[auxassignments_b[spin_b,i+sweep*len(self._pairs)], :] / fsum_b[auxassignments_b[spin_b,i+sweep*len(self._pairs)], np.newaxis],
                    orb_a_configs, orb_b_configs,
                )

                # Adding to results
                results["value_%s%s"%(spin_dic[spin_a],spin_dic[spin_b])] += np.einsum("i,ijklm->ijklm", wfratio, orbratio)
                results["norm_%s"%spin_dic[spin_a]] += norm_a[auxassignments_a[spin_a,i+sweep*len(self._pairs)]]
                results["norm_%s"%spin_dic[spin_b]] += norm_b[auxassignments_b[spin_b,i+sweep*len(self._pairs)]]

        # Average over sweeps and pairs
        for sector in self._spin_sectors:
            results["value_%s%s"%(spin_dic[sector[0]],spin_dic[sector[1]])] /= self._nsweeps 
        for s in np.unique(self._spin_sectors):
            # Correct number of spin_up and spin_down moves (necessary when totalspin!=0)
            results["norm_%s"%spin_dic[s]] /= ( self._nsweeps * np.sum( np.abs(-s + (self._pairs<self._mol.nelec[0]) ) ) )
            
        return results

    def avg(self, configs, wf):
        d = self(configs, wf)
        davg = {}
        for k, v in d.items():
            # print(k, v.shape)
            davg[k] = np.mean(v, axis=0)
        return davg

    def get_extra_configs(self, configs):
        """ Returns an nstep length array of configurations
        starting from self._extra_config """
        # Generates aux_configs_up
        aux_configs_up = []
        if 0 in self._spin_sectors:
            for step in range(self._nsweeps * len(self._pairs)):
                aux_configs_up.append(np.copy(self._aux_configs_up))
                accept, self._aux_configs_up = sample_onebody(
                    self._mol, self._orb_coeff[0], self._aux_configs_up, tstep=self._tstep
                )
            aux_configs_up = np.concatenate(aux_configs_up,axis=0)
        # Generates aux_configs_down
        aux_configs_down = []
        if 1 in self._spin_sectors:
            for step in range(self._nsweeps * len(self._pairs)):
                aux_configs_down.append(np.copy(self._aux_configs_down))
                accept, self._aux_configs_down = sample_onebody(
                    self._mol, self._orb_coeff[1], self._aux_configs_down, tstep=self._tstep
                )
            aux_configs_down = np.concatenate(aux_configs_down,axis=0)

        # Generates random choice of aux_config_up and aux_config_down for moving electron_a and electron_b
        naux_up = self._aux_configs_up.shape[0]
        naux_down = self._aux_configs_down.shape[0]
        auxassignments_a = np.array([ np.random.randint(0, int(naux_up/2), size=(self._nsweeps*len(self._pairs), nconf)),
                             np.random.randint(0, int(naux_down/2), size=(self._nsweeps*len(self._pairs), nconf)) ])
        auxassignments_b = np.array([ np.random.randint(int(naux_up/2), naux_up, size=(self._nsweeps*len(self._pairs), nconf)),
                             np.random.randint(int(naux_down/2), naux_down, size=(self._nsweeps*len(self._pairs), nconf)) ])
        return [aux_configs_up, aux_configs_down], [auxassignments_a, auxassignments_b]

def normalize_tbdm(tbdm, norm_a, norm_b):
    '''Returns tbdm by taking the ratio of the averages in Eq. (10) of DOI:10.1063/1.4793531.'''
    # We are using pySCF's notation:
    #  tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s1,j} c_{s2,l} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s1,i} \phi_{s2,k}
    return tbdm / np.einsum('i,j,k,l->ijkl',norm_a,norm_a,norm_b,norm_b) ** 0.5 
