3-dimensional functions
**********************************

3d basis functions :math:`f(r)`, where :math:`r` is the magnitude of a 3d vector.

=================================
func3d objects
=================================

All func3d objects have the following interface, and the input arrays `rvec` and `r` are the same for all functions

.. py:function:: __init__()

    Initialize the parameters of the function in the dictionary `self.parameters`


.. py:function:: value(rvec, r)

    Evaluate the function f(r).

    :parameter rvec: (nconfig,...,3) 
    :parameter r: (nconfig,...) 
    :returns: function value :math:`f(r)`
    :rtype: ``(nconfig,...)`` array
    
.. py:function:: gradient(rvec, r)

    Evaluate the gradient :math:`\nabla f(r)`

    :returns: function gradient  :math:`(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z})`
    :rtype: ``(nconfig, ..., 3)`` array
    
.. py:function:: gradient_value(rvec, r)

    Evaluate the gradient and value together :math:`\nabla f(r), f(r)`

    :returns: function gradient and value
    :rtype: tuple of ``(nconfig, ...), (nconfig, ..., 3)`` arrays
    
.. py:function:: laplacian(rvec, r)

    Evaluate the laplacian :math:`\nabla^2 f(r)`

    :returns: laplacian as :math:`(\frac{\partial^2 f}{\partial x^2}, \frac{\partial^2 f}{\partial y^2}, \frac{\partial^2 f}{\partial z^2})`
    :rtype: ``(nconfig, ..., 3)`` array
    
.. py:function:: gradient_laplacian(rvec, r)

    Evaluate the gradient an laplacian together :math:`\nabla f(r), \nabla^2f(r)`

    :returns: gradient and laplacian
    :rtype: two ``(nconfig, ..., 3)`` arrays
    
.. py:function:: pgradient(rvec, r)

    Evaluate the gradient with respect to parameter(s)

    :returns: parameter gradient  {'pname': :math:`\frac{\partial f}{\partial p}`}
    :rtype: dictionary, values are ``(nconfig, ...)`` arrays
    


.. automodule:: pyqmc.func3d
   :members:
   
