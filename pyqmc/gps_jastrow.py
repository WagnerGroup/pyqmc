import numpy as np
from pyqmc import mc

#things to change: figure if there is a way to deal with epos with len(epos.shape)>2 without if statement
#when masking, only do relevant calculations. understand masking better.

class GPSJastrow:

  def __init__(self,mol,start_alpha,start_sigma):
    self.n_training= start_alpha.size
    self._mol = mol
    self.iscomplex=False
    self.parameters = {}
    #self.parameters["Xtraining"] = mc.initial_guess(mol,n_training).configs
    self.parameters["Xtraining"]=np.array([[[0,0,0],[0,0,1.54/0.529177]],[[0,0,1.54/0.529177],[0,0,0]]])
    self.parameters["sigma"] = start_sigma
    self.parameters["alpha"] = start_alpha
  def recompute(self,configs):
    self._configscurrent= configs
    self.n_configs,self.n_electrons = configs.configs.shape[:-1]
    e_partial = np.zeros(shape=(self.n_configs,self.n_training,self.n_electrons))

    #is there faster way than looping over e
    for e in range(self.n_electrons):
      e_partial[:,:,e] = self._compute_partial_e(configs.configs[:,e,:])
    self._sum_over_e = e_partial.sum(axis=-1)
    self._e_partial= e_partial
    return self.value()

  def _compute_partial_e(self,epos):
    y=epos[..., np.newaxis, np.newaxis, :] - self.parameters['Xtraining']
    partial_e_2= (y*y).sum(axis=(-2,-1))
    return partial_e_2

  def _compute_value_from_sum_e(self,sum_e):
    means = np.sum(self.parameters['alpha']*np.exp(-sum_e/(2.0*self.parameters["sigma"])),axis=-1)
    return means
  
  def updateinternals(self,e,epos,configs,mask=None):
    if mask is None:
      mask = [True] * epos.configs.shape[0]
    prior_partial_e = self._e_partial[:,:,e]
    prior_sum = self._sum_over_e
    partial_e = self._compute_partial_e(epos.configs)
    new_sum=prior_sum+partial_e-prior_partial_e
    self._sum_over_e[mask]=new_sum[mask]
    self._e_partial[:,:,e][mask] = partial_e[mask]
    self._configscurrent.move(e, epos, mask)

#return phase and log(value) tuple
  def value(self):
    return (np.ones(self.n_configs),self._compute_value_from_sum_e(self._sum_over_e))

  def testvalue(self,e,epos,mask=None):
    if mask is None:
      mask = [True] * epos.configs.shape[0]

    old_means = self.value()[1]
    new_partial_e = self._compute_partial_e(epos.configs)
    prior_e_sum=self._sum_over_e
    if len(epos.configs.shape)>2:
      e_sum2 = prior_e_sum[:,np.newaxis,:]+new_partial_e-self._e_partial[:,np.newaxis,:,e]
      means2=self._compute_value_from_sum_e(e_sum2)
      return np.exp(means2-old_means[:,np.newaxis])[mask]
    else:
      e_sum2 = prior_e_sum+new_partial_e-self._e_partial[:,:,e]
      means2=self._compute_value_from_sum_e(e_sum2)
      return np.exp(means2-old_means)[mask]
      

    # if len(epos.configs.shape)>2:
    #   e_sum = np.zeros(shape=(self.n_configs,epos.configs.shape[1],self.n_training))
    #   for nconf in range(self.n_configs):
    #     for pos in range(epos.configs.shape[1]):
    #       for nt in range(self.n_training):
    #         e_sum[nconf,pos,nt] = prior_e_sum[nconf,nt]+new_partial_e[nconf,pos,nt]-self._e_partial[nconf,nt,e] 
    # else:  
    #   e_sum = prior_e_sum+new_partial_e-self._e_partial[:,:,e]
    # means = self._compute_value_from_sum_e(e_sum)
    # if len(epos.configs.shape)>2:
    #   diff = np.zeros(shape=(self.n_configs,epos.configs.shape[1]))
    #   for nconf in range(self.n_configs):
    #     for npos in range(epos.configs.shape[1]):
    #       diff[nconf,npos] = means[nconf,npos]-old_means[nconf]
    #   print(means.shape)
    #   print(means-means2)
    #   return np.exp(diff)[mask]
    # else:
    #   print(means.shape)
    #   print(means-means2)
    #   return np.exp(means-old_means)[mask]

  def gradient(self,e,epos):
    prior_partial_e = np.copy(self._e_partial[:,:,e])
    partial_e = self._compute_partial_e(epos.configs)
    gradsum=np.sum((self.parameters['Xtraining'][np.newaxis,:,:,:] - epos.configs[:,np.newaxis,np.newaxis,:]),axis=-2)
    gradsum=np.transpose(gradsum,axes=(2,0,1))
    term1 = gradsum/self.parameters['sigma']
    e_sum = self._sum_over_e+partial_e-prior_partial_e
    grads = np.sum(self.parameters["alpha"]*term1*np.exp(-e_sum/(2.0*self.parameters['sigma'])),axis=-1)
    return grads

#first simplify the math, then the function
  def laplacian(self,e,epos):
    prior_partial_e = self._e_partial[:,:,e]
    gradsum = np.zeros(shape=(self.n_configs,self.n_training,3))
    partial_e = self._compute_partial_e(epos.configs)
    for nc in range(self.n_configs):
      for nt in range(self.n_training):
        sum=0
        for ne in range(self.n_electrons):
          sum+=self.parameters['Xtraining'][nt,ne]-epos.configs[nc]
        gradsum[nc,nt]=sum
    prior_e_sum=self._sum_over_e
    e_sum = prior_e_sum+partial_e-prior_partial_e
    term1 = np.power(gradsum/self.parameters['sigma'],2)
    term1 = np.sum(term1,axis=-1)
    term2 = -3*self.n_electrons/self.parameters['sigma']

    laps=np.zeros(shape=(self.n_configs))
    gradient = self.gradient(e,epos)
    for nc in range(self.n_configs):
      sum=0
      for nt in range(self.n_training):
        sum+=self.parameters['alpha'][nt]*(term1[nc,nt]+term2)*np.exp(-e_sum[nc,nt]/(2*self.parameters['sigma']))
      laps[nc] = sum
    laps += np.sum(np.power(gradient.T,2),axis=-1)
    return laps
  

  def gradient_laplacian(self,e,epos):
    laplacian = self.laplacian(e,epos)
    grad = self.gradient(e,epos)
    return grad,laplacian

  def pgradient(self):
    configs = self._configscurrent.configs
    alphader = np.exp(-0.5*self._sum_over_e/self.parameters["sigma"])
    Xder = np.zeros(shape= (self.n_configs,self.n_training,self.n_electrons,3))
    #print(self._sum_over_e.shape,"sumovere")
    #ds= self.parameters["Xtraining"][np.newaxis,:,:,:]-configs[:,np.newaxis,:,:]
    #nc,ns,ne,3
    #Xder2 = self.parameters["alpha"]*ds*np.exp(-0.5*self._sum_over_e/self.parameters["sigma"])/self.parameters["sigma"]
    for nc in range(self.n_configs):
      dersum  = np.zeros(shape=(self.n_training,self.n_electrons,3))
      for m in range(self.n_electrons):
        dersum += self.parameters["Xtraining"]-configs[nc,m]
      for nt in range(self.n_training):
        Xder[nc,nt] = self.parameters["alpha"][nt]*dersum[nt]*np.exp(-0.5*self._sum_over_e[nc,nt]/self.parameters["sigma"])/self.parameters["sigma"]
    
    #print('diff',Xder-Xder2)

    sigmader = np.zeros(self.n_configs)
    for nc in range(self.n_configs):
      sum=0
      for nt in range(self.n_training):
        sum+= self.parameters["alpha"][nt]*np.exp(-0.5*self._sum_over_e[nc,nt]/self.parameters["sigma"])*self._sum_over_e[nc,nt]/(2*self.parameters["sigma"]*self.parameters["sigma"])
      sigmader[nc] = sum
    return {"alpha":alphader,"Xtraining":Xder,"sigma":sigmader}

  def gradient_value(self,e,epos):
    return self.gradient(e,epos),self.testvalue(e,epos)





  