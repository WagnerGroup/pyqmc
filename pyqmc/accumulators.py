import numpy as np 
from pyqmc.energy import energy

class EnergyAccumulator:
  """returns energy of each configuration in a dictionary. 
  Keys and their meanings can be found in energy.energy """
  def __init__(self, mol):
      self.mol = mol

  def __call__(self,configs, wf):
      return energy(self.mol, configs, wf)

  def avg(self,configs,wf):
      d={}
      for k,it in self(configs,wf).items():
          d[k]=np.mean(it,axis=0)
      return d



class LinearTransform:
    def __init__(self,parameters,to_opt=None):
        if to_opt is None:
            self.to_opt=list(parameters.keys())
        else:
            self.to_opt=to_opt

        self.shapes=np.array([parameters[k].shape for k in self.to_opt])
        self.slices=np.array([ np.prod(s) for s in self.shapes ])
        
    def serialize_parameters(self,parameters):
        """Convert the dictionary to a linear list
        of gradients
        """
        params=[]
        for k in self.to_opt:
            params.append(parameters[k].flatten())
        return np.concatenate(params)


    def serialize_gradients(self,pgrad):
        """Convert the dictionary to a linear list
        of gradients
        """
        grads=[]
        for k in self.to_opt:
            grads.append(pgrad[k].reshape((pgrad[k].shape[0],-1)))
        return np.concatenate(grads,axis=1)

        
    def deserialize(self,parameters):
        """Convert serialized parameters to dictionary
        """
        n=0
        d={}
        for i,k in enumerate(self.to_opt):
            np=self.slices[i]
            d[k]=parameters[n:n+np].reshape(self.shapes[i])
            n+=np
        return d
    


class PGradTransform:
    """   """
    def __init__(self,enacc,transform,nodal_cutoff=1e-6):
        self.enacc=enacc
        self.transform=transform
        self.nodal_cutoff=nodal_cutoff

    def _node_cut(self,configs,wf):
        """ Return true if a given configuration is within nodal_cutoff 
        of the node """
        ne=configs.shape[1]
        d2=0.0
        for e in range(ne):
            d2+=np.sum(wf.gradient(e,configs[:,e,:])**2,axis=0)
        r=1./(d2*ne*ne)
        return r < self.nodal_cutoff**2


    def __call__(self,configs,wf):
        pgrad=wf.pgradient()
        d=self.enacc(configs,wf)
        energy = d['total']
        dp=self.transform.serialize_gradients(pgrad)
        node_cut=self._node_cut(configs,wf)
        dp[node_cut,:]=0.0
        #print('number cut off',np.sum(node_cut))
        
        d['dpH'] = np.einsum('i,ij->ij',energy,dp)
        d['dppsi'] = dp
        d['dpidpj'] = np.einsum('ij,ik->ijk',dp,dp)
        return d

    def avg(self,configs,wf):
        nconf=configs.shape[0]
        pgrad=wf.pgradient()
        den=self.enacc(configs,wf)
        energy = den['total']
        dp=self.transform.serialize_gradients(pgrad)

        node_cut=self._node_cut(configs,wf)
        dp[node_cut,:]=0.0
        print('number cut off',np.sum(node_cut))

        d={}
        for k,it in den.items():
            d[k]=np.mean(it,axis=0)
        d['dpH'] = np.einsum('i,ij->j',energy,dp)/nconf
        d['dppsi'] = np.mean(dp,axis=0)
        d['dpidpj'] = np.einsum('ij,ik->jk',dp,dp)/nconf

        return d


def test_transform():
    from pyscf import gto,scf 
    import pyqmc

    r=1.54/.529177
    mol = gto.M(atom='H 0. 0. 0.; H 0. 0. %g'%r, ecp='bfd',basis='bfd_vtz',unit='bohr',verbose=1)
    mf=scf.RHF(mol).run()
    wf=pyqmc.slater_jastrow(mol,mf)
    enacc=pyqmc.EnergyAccumulator(mol)
    print(list(wf.parameters.keys()))
    transform=LinearTransform(wf.parameters)
    x=transform.serialize_parameters(wf.parameters)
    print(x)
    print(transform.deserialize(x))
    configs=pyqmc.initial_guess(mol,10)
    wf.recompute(configs)
    pgrad=wf.pgradient()
    print(transform.serialize_gradients(pgrad))

    

if __name__=="__main__":
    test_transform()
        
        



