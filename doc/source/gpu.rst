GPU implementation using CuPy
*****************************

The current implementation focuses on using the GPU for many internal wave function operations, so that

- Wave function parameters are stored on the GPU
    - This allows the calculations to be done on the GPU without copying the parameters over each time.
    - When assigning wf parameters, arrays need to be moved to GPU
    - During optimization, parameters serialize to CPU and deserialize back to GPU

- Wave function methods return values on the CPU
    - This allows functions like vmc and dmc to remain unchanged, and operate only on CPU.

The Slater determinant helper functions use the GPU:

- Orbitals object returns AOs and MOs on GPU.
- `determinant_tools.compute_value` takes in GPU arrays from the Slater object and outputs CPU arrays.

In addition, some calculations outside of wf evaluation are also carried out on GPU

- RDM: since the RDM accumulators evaluate orbitals directly, the orbital arrays and internal calculations are on GPU. 
  All return values are on CPU. 

  (Note: wf.testvalue_many returns an array on CPU. Since it is only used in RDMs, which move the arrays back to GPU anyway, it might make sense for testvalue_many to return GPU arrays instead.)
- Ewald: reciprocal_space_electron() uses GPU internally. Its inputs and outpus are CPU arrays.


