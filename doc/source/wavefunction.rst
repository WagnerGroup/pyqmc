=================================
Wave function objects
=================================

.. py:function:: __init__()

    Initialize the wave function parameters. 
    This can be just about anything, but do not allocate walker-specific memory here. 


.. py:function:: recompute(configs)

Initialize any walker-based storage and compute the value of the wave function. 

.. py:function:: updateinternals(e,epos, mask=None)

Update any internals given that electron e moved to epos. mask is a Boolean array which allows us to update only certain walkers

.. py:function:: value()

Return logarithm of the wave function as noted in recompute()

.. py:function:: testvalue(e,epos)

Return the ratio between the current wave function and the wave function if electron e's position is replaced by epos

.. py:function:: gradient(e,epos)

Return the gradient of the log wave function.
Note that this can be called even if the internals have not been updated for electron e, if epos differs from the current position of electron e.

.. py:function:: laplacian(e,epos)

Return the laplacian Psi/ Psi. Conditions similar to gradient()

.. py:function:: pgradient()

Return the parameter gradient of Psi. 
Returns d_p \Psi/\Psi as a dictionary of numpy arrays, which correspond to the parameter dictionary.

----------------------
Slater determinant
----------------------

.. automodule:: pyqmc.slater
   :members:
   

----------------------
Jastrow factor
----------------------

.. automodule:: pyqmc.jastrowspin
   :members:
 
--------------------------------------------
Multiple Slater determinant
--------------------------------------------


.. automodule:: pyqmc.multislater
   :members:

----------------------
Multipy wave function
----------------------

.. automodule:: pyqmc.multiplywf
   :members:
