import numpy as np
""" 
Collection of 3d function objects. Each has a dictionary parameters, which corresponds
to any variational parameters the funtion has.

They should implement the following functions, all of which take input value x.
x should be of dimension (nconf,3).

value(x):
    returns f(x)
gradient(x)
    returns grad f(x) (nconf,3)
laplacian(x)
    returns diagonals of Hessian (nconf,3)
pgradient(x)
    returns dp f(x) as a dictionary corresponding to the keys of self.parameters
"""

class GaussianFunction:
    r"""A representation of a Gaussian: 
    :math: `\exp(-\alpha r^2)`
    where :math: `\alpha` can be accessed through parameters['exponent']

    """
    def __init__(self,exponent):
        self.parameters={}
        self.parameters['exponent']=exponent

    def value(self,x): 
        """Returns function exp(-exponent*r^2).
        Parameters:
          x: (nconfig,3) vector
        Returns:
          func: (nconfig,) vector
        """
        r2=np.sum(x**2,axis=1)
        return np.exp(-self.parameters['exponent']*r2)
        
    def gradient(self,x):
        """Returns gradient of function.
        Parameters:
          x: (nconfig,3) vector
        Returns:
          grad: (nconfig,3) vector
        """
        v=self.value(x)
        return -2*self.parameters['exponent']*x*v[:,np.newaxis]

    def laplacian(self,x):
        """Returns laplacian of function.
        Parameters:
          x: (nconfig,3) vector
        Returns:
          grad: (nconfig,3) vector (components of laplacian d^2/dx_i^2 separately)
        """
        v=self.value(x)
        alpha=self.parameters['exponent']
        return (4*alpha*alpha*x*x-2*alpha)*v[:,np.newaxis]

    def pgradient(self,x):
        """Returns parameters gradient.
        Parameters:
          x: (nconfig,3) vector
        Returns:
          pgrad: dictionary {'exponent':d/dexponent}
        """
        r2=np.sum(x**2,axis=1)
        return {'exponent':-r2*np.exp(-self.parameters['exponent']*r2)}
        
        
class PadeFunction:
    """
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    alpha_k = alpha/2^k, k starting at 0
    """
    def __init__(self, alphak):
        self.parameters={}
        self.parameters['alphak'] = alphak

    def value(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3 (number of inner dimensions doesn't matter)
        Return:
          func: same dimensions as rvec, but the last one removed 
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = self.parameters['alphak']* r
        return (a/(1+a))**2

    def gradient(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3, displacement between particles
            For example, nconf x n_elec_pairs x 3, where n_elec_pairs could be all pairs of electrons or just the pairs that include electron e for the purpose of updating one electron.
            Or it could be nconf x nelec x natom x 3 for electron-ion displacements
        Return:
          grad: same dimensions as rvec
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        a = self.parameters['alphak']* r
        grad = 2* self.parameters['alphak']**2/(1+a)**3 *rvec
        return grad
        
    def laplacian(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3
        Return:
          lap: same dimensions as rvec, d2/dx2, d2/dy2, d2/dz2 separately
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        a = self.parameters['alphak']* r
        #lap = 6*self.parameters['alphak']**2 * (1+a)**(-4) #scalar formula
        lap = 2*self.parameters['alphak']**2 * (1+a)**(-3) \
              *(1 - 3*a/(1+a)*(rvec/r)**2)
        return lap

    def pgradient(self, rvec):
        """ Return gradient of value with respect to parameter alphak
        Parameters:
          rvec: nconf x ... x 3
        Return:
          pgrad: dictionary {'alphak':d/dalphak} with akderiv dimensions (config,)
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = self.parameters['alphak']* r
        akderiv = 2*a/(1+a)**3 * r
        return {'alphak':akderiv}

    
class ExpCuspFunction:
    """
    :math:`b(r) = -\frac{p(r/r_{cut})}{1+\gamma*p(r/r_{cut})} + \frac{1}{3+\gamma}` 
    where 
    :math:`p(y) = y - y^2 + y^3/3`
    This function is positive at small r, decreasing to zero at r=rcut, being cutoff to 
    zero for r>rcut.
    """
    def __init__(self, gamma, rcut):
        self.parameters={}
        self.parameters['gamma'] = gamma
        self.parameters['rcut'] = rcut

    def value(self, rvec):
        """Returns 
        Parameters:
          rvec: (nconf,3) vector
        Returns:
          func: p(r/rcut)/(1+gamma*p(r/rcut))
        """
        r = np.linalg.norm(rvec, axis=-1)
        y = r/self.parameters['rcut']
        mask=y<=1
        func=np.zeros(r.shape)
        func[mask]=( - (y-y**2+y**3/3) / ( 1 + self.parameters['gamma'] * (y-y**2+y**3/3) ) + 1/(3+self.parameters['gamma']) )[mask]
        return func
        
    def gradient(self, rvec):
        """
        Parameters:
          rvec: (nconf,3) vector
        Returns:
          grad: has same dimensions as rvec 
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        y = r/self.parameters['rcut']
        mask=(y<=1)*np.ones(rvec.shape).astype(bool)
        func=np.zeros(rvec.shape)
        func[mask]=(- rvec * ( (1-2*y+y**2) / ( 1 + self.parameters['gamma'] * (y-y**2+y**3/3) )**2 / (self.parameters['rcut'] * r) ) )[mask]
        return func

    def laplacian(self, rvec):
        """
        Parameters:
          rvec: (nconf,3) vector
        Returns:
          lapl: has same dimensions as rvec, because returns components of laplacian d^2/dx_i^2 separately
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        y = r/self.parameters['rcut']
        dydr = 1/self.parameters['rcut']
        mask=(y<=1)*np.ones(rvec.shape).astype(bool)
        func=np.zeros(rvec.shape)
        func[mask]=-( ((1-2*y+y**2) / r / (1+self.parameters['gamma']*(y-y**2+y**3/3))**2 / self.parameters['rcut']) + (( (-2/self.parameters['rcut']+2*r/self.parameters['rcut']**2)/r**2 - (1-2*y+y**2)/r**3 ) / self.parameters['rcut'] / (1+self.parameters['gamma']*(y-y**2+y**3/3))**2 + ((1-2*y+y**2)/self.parameters['rcut'])**2 * (-2*self.parameters['gamma']/(1+self.parameters['gamma']*(y-y**2+y**3/3))**3) / r**2) * (rvec**2) )[mask]
        return func
    
    def pgradient(self, rvec):
        """ Returns gradient of self.value with respect all parameters
        Parameters:
          rvec: (nconf,3) vector
        Returns:
          paramderivs: dictionary {'rcut':d/drcut,'gamma':d/dgamma}
        """
        r = np.linalg.norm(rvec, axis=-1)
        y = r/self.parameters['rcut']
        mask=y<=1
        func={'rcut':np.zeros(r.shape),'gamma':np.zeros(r.shape)}
        func['rcut'][mask]=-( -r/self.parameters['rcut']**2 * (1-2*y+y**2) / (1+self.parameters['gamma']*(y-y**2+y**3/3))**2 )[mask]
        func['gamma'][mask]=-( -(y-y**2+y**3/3)**2/(1+self.parameters['gamma']*(y-y**2+y**3/3))**2 - 1/(3+self.parameters['gamma'])**2 )[mask]
        return func


    
def test_func3d_gradient(bf, delta=1e-5):
    rvec = np.random.randn(150,3)
    grad = bf.gradient(rvec)
    numeric = np.zeros(rvec.shape)
    for d in range(3):
      pos = rvec.copy()
      pos[:,d] += delta
      plusval = bf.value(pos)
      pos[:,d] -= 2*delta
      minuval = bf.value(pos)
      numeric[:,d] = (plusval - minuval)/(2*delta)
    maxerror = np.amax(np.abs(grad-numeric))
    normerror = np.linalg.norm(grad-numeric)
    return (maxerror,normerror)

def test_func3d_laplacian(bf, delta=1e-5):
    rvec = np.random.randn(150,3)
    lap = bf.laplacian(rvec)
    numeric = np.zeros(rvec.shape)
    for d in range(3):
      pos = rvec.copy()
      pos[:,d] += delta
      plusval = bf.gradient(pos)[:,d]
      pos[:,d] -= 2*delta
      minuval = bf.gradient(pos)[:,d]
      numeric[:,d] = (plusval - minuval)/(2*delta)
    maxerror = np.amax(np.abs(lap-numeric))
    normerror = np.linalg.norm(lap-numeric)
    return (maxerror,normerror)

def test(): 
    test_functions = {'Pade':PadeFunction(0.2), 'Gaussian':GaussianFunction(0.4), 'ExpCusp':ExpCuspFunction(0.4,0.6)}
    for name, func in test_functions.items():
        for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
            print(name, 'delta', delta, "Testing gradient", test_func3d_gradient(func,delta=delta))
            print(name, 'delta', delta, "Testing laplacian", test_func3d_laplacian(func,delta=delta))

            
if __name__=="__main__":
    test()

