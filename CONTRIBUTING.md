


## How to contribute a feature

First off, THANK YOU! Before working on it, do make sure to raise an issue and/or get in touch with the developers by email. They may have thoughts on how to create the feature, or useful advice to save you some time.

Much of this is based on the excellent guide [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962). 

### Create a fork

Click the "Fork" button on the `pyqmc` GitHub page. You can then clone it to your computer by doing

```
git clone git@github.com:USERNAME/pyqmc.git
```


### Do your work

Make sure that you branch **from master**, not from another feature you may be working on.

```
git checkout master

git checkout -b newfeature
```

### Keeping your fork up to date

You should always update to master before doing a pull request, and occasionally throughout development. 

Do this once:

```
git remote add upstream https://github.com/wagnergroup/pyqmc.git

# verify
git remote -v
```

Each time 
```
git fetch upstream

git branch -va
```

Update your master copy.
```
git checkout master
git merge upstream/master
```

(If warranted) Update your feature with the master changes
```
git checkout newfeature
git rebase master
```
Note the rebase here; it prevents the commit history from getting too messed up.

### Coding style

1. Try to use [functional](https://en.wikipedia.org/wiki/Functional_programming) styles as much as possible. There has to be a very good reason to create a new class. 
2. Keep functions small and with one responsibility. You may need to revise your first implementation to do this; it's worth it. That is, each function should do one thing. [Related](https://en.wikipedia.org/wiki/Single-responsibility_principle)
3. As much as possible, avoid looping in Python. `numpy` calls should, as much as possible, be called on all walkers in the ensemble to avoid Python's poor performance.
4. As much as possible, avoid repeatedly calling functions in a loop, again for performance reasons. Prefer having functions operate in batches.
5. Try to avoid using libraries that are not in `requirements.txt`. If it is absolutely necessary, discuss with the maintainers. 


### Documentation requirements

Not all of the current codebase satisfies this unfortunately. 

1. Use type annotations.
2. All multidimensional arrays should have comments indicating their dimensions. For example, `A = np.zeros((200,6,3)) # nconfig x electrons x dimension`
3. Each function should have a [docstring](https://peps.python.org/pep-0257/) that explains all inputs and outputs.

### Writing tests

There are a few strategies to create tests for numerical methods. 
This is roughly in order of preference. 

1. Compare the numerical solution to an exact one. (or if you are implementing say analytic gradients, compare analytic gradients to numerical ones)
2. Compare two independent numerical solutions. (for example, comparing Hartree-Fock energies from pyscf to the energy of the same Slater determinant in pyqmc, or comparing two methods of computing the kinetic energy)
3. Checking that it actually runs and produces consistent results (for example, just running the function with reasonable inputs and checking that the results did not change from some pre-run references)
4. Sanity check (for example, checking that an optimized Jastrow factor does indeed lower the energy)


### Submitting your work

First make sure that you have gone through the checklist:

1. Any new functions are documented as above.
2. You have written a test for the new feature. 
3. Your code runs all tests. Run `pytest` in your root directory and all tests should pass.
4. Write a short explanation of the new feature. If it is a bugfix, give an example of the bug and show that your implementation fixes it. This could be part of a test.

With that in hand, you should be able to hit the Pull Request button on GitHub to initiate the request. 

