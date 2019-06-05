import numpy as np
from numpy import polyfit,linspace,inf


def line_minimization(func, params0, line=None, line_min_steps=5, max_step=1, **func_kwargs):
    """
    Args:
        func: a function that takes in parameters (p0) and returns (value, grad)

        params0: a point in the domain to start the minimization

        line: vector in the line to minimize along

        line_min_steps: number of steps to take along line minimization

        func_kwargs: kwargs to pass to func
    
    Returns:
        min: the final point in the domain that minimized func along line

        fitted_minval: the minimized value of func
    """

    assert line_min_steps>3, "cubic line minimization needs at least 4 points"
    val, grad = func.value(params0)
    if line is None:
        line=grad/np.linalg.norm(grad)
    normalization = np.sum(line**2)

    line_steps = linspace(0.0,max_step,line_min_steps-1) 
    line_steps[0] = -max_step/line_min_steps # already have current point; need one behind
    line_points = params0 + line*line_steps[:,np.newaxis]
    line_data = [ func(p, **func_kwargs) for pidx,p in enumerate(line_points) ]
    line_data.insert(1,(val,grad))
    line_deriv = [data[1]@line/normalization for data in line_data]
    line_steps.insert(1,0)
    line_min, fitted_minval = fit_line_minimum(line_steps,line_deriv)

    params = params0 + line_min*line
    return params, fitted_minval
      
def fit_line_minimum(xvals,yderivs,yderiv_err=None):
    ''' Fit a cubic function using its derivatives and return its minimum.
    This means fitting a quadratic to the derivatives and using its roots.
  
    Assumes first step is decreasing (which would happen if you are following the negative derivative).
    This is to simplify monotonic cases.
  
    Args:
      xvals (array-like): places where data is along the axis. Assumed to be sorted and first value is zero.
      yderivs (array-like): derivatives at xvals.
      yderiv_err (array-like): (Optional) errors on yderivs.
    Returns:
      float: between xvals[0] and xvals[-1].
    '''
    print("Line minimizaion.")
    if yderiv_err is not None: raise NotImplementedError("Should be trivial to add this.")
  
    coefs = polyfit(xvals,yderivs,deg=2)
  
    descriminant = coefs[1]**2 - 4*coefs[0]*coefs[2]
  
    if abs(coefs[0])<1e-10: # very close to quadratic.
        print("Nearly quadratic")
        if coefs[1]>0: xmin = -coefs[2]/coefs[1]
        else: xmin = inf
    elif descriminant >= 0:
        print("Has local minima")
        xmin = (-coefs[1] + descriminant**0.5)/2/coefs[0]
    else: # monotonic.
        print("Monotonic")
        if coefs[2] > 0: xmin = inf
        if coefs[2] < 0: xmin = -inf
  
    print("xmin found:",xmin)
    xfinal = max(0.5*(xvals[0]+xvals[1]),min(xmin,xvals[-1]))
  
    return xfinal, np.dot(coefs, xfinal**np.array([3,2,1,0]))
