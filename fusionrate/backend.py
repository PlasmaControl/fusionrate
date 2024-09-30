# This 'backend' functionality was copied from DESC (and then pared down)
# From other files, we can import jnp from this file.
# i.e. from backend import jnp.
# It'll automatically be either regular numpy (if set that way via environment
# variable, or if jax is not found) or jax (if it's found and permitted).
# np is always regular numpy.
#

# MIT License
#
# Copyright (c) 2020 Daniel Dudt, Rory Conlin, Dario Panici, Egemen Kolemen
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
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import warnings

import numpy as np

from fusionrate.constants import PROJECT

PROJECT_ALLCAPS = PROJECT.upper()

if os.environ.get(f"{PROJECT_ALLCAPS}_BACKEND") == "numpy":
    jnp = np
    use_jax = False
    print(
        f"{PROJECT}"+" using numpy backend, version={}, dtype={}".format(
            np.__version__, np.linspace(0, 1).dtype
        )
    )
else:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jax
            import jax.numpy as jnp
            import jaxlib
            from jax import config as jax_config

            jax_config.update("jax_enable_x64", True)
            x = jnp.linspace(0, 1)
        use_jax = True
        print(
            f"{PROJECT} "
            + f"using JAX backend, jax version={jax.__version__}, "
            + f"jaxlib version={jaxlib.__version__}, dtype={x.dtype}"
        )
        del x
    except ModuleNotFoundError:
        jnp = np
        x = jnp.linspace(0, 1)
        use_jax = False
        warnings.warn("Failed to load JAX")
        print(
            "{} using NumPy backend, version={}, dtype={}".format(
                PROJECT, np.__version__, x.dtype
            )
        )


if use_jax:  # noqa: C901 - FIXME: simplify this, define globally and then assign?
    jit = jax.jit
    fori_loop = jax.lax.fori_loop
    cond = jax.lax.cond
    switch = jax.lax.switch
    while_loop = jax.lax.while_loop

    # see docstring below
    def put(arr, inds, vals):
        return jnp.asarray(arr).at[inds].set(vals)


    # see docstring below
    def sign(x):
        x = jnp.atleast_1d(x)
        y = jnp.where(x == 0, 1, jnp.sign(x))
        return y

else:
    jit = lambda func, *args, **kwargs: func

    # see docstring below
    def put(arr, inds, vals):
        arr[inds] = vals
        return arr


    # see docstring below
    def sign(x):
        x = np.atleast_1d(x)
        y = np.where(x == 0, 1, np.sign(x))
        return y


    def fori_loop(lower, upper, body_fun, init_val):
        """Loop from lower to upper, applying body_fun to init_val.

        This version is for the numpy backend, for jax backend see jax.lax.fori_loop

        Parameters
        ----------
        lower : int
            an integer representing the loop index lower bound (inclusive)
        upper : int
            an integer representing the loop index upper bound (exclusive)
        body_fun : callable
            function of type ``(int, a) -> a``.
        init_val : array-like or container
            initial loop carry value of type ``a``

        Returns
        -------
        final_val: array-like or container
            Loop value from the final iteration, of type ``a``.

        """
        val = init_val
        for i in np.arange(lower, upper):
            val = body_fun(i, val)
        return val

    def cond(pred, true_fun, false_fun, operand):
        """Conditionally apply true_fun or false_fun.

        This version is for the numpy backend, for jax backend see jax.lax.cond

        Parameters
        ----------
        pred: bool
            which branch function to apply.
        true_fun: callable
            Function (A -> B), to be applied if pred is True.
        false_fun: callable
            Function (A -> B), to be applied if pred is False.
        operand: any
            input to either branch depending on pred. The type can be a scalar, array,
            or any pytree (nested Python tuple/list/dict) thereof.

        Returns
        -------
        value: any
            value of either true_fun(operand) or false_fun(operand), depending on the
            value of pred. The type can be a scalar, array, or any pytree (nested
            Python tuple/list/dict) thereof.

        """
        if pred:
            return true_fun(operand)
        else:
            return false_fun(operand)

    def switch(index, branches, operand):
        """Apply exactly one of branches given by index.

        If index is out of bounds, it is clamped to within bounds.

        Parameters
        ----------
        index: int
            which branch function to apply.
        branches: Sequence[Callable]
            sequence of functions (A -> B) to be applied based on index.
        operand: any
            input to whichever branch is applied.

        Returns
        -------
        value: any
            output of branches[index](operand)

        """
        index = np.clip(index, 0, len(branches) - 1)
        return branches[index](operand)

    def while_loop(cond_fun, body_fun, init_val):
        """Call body_fun repeatedly in a loop while cond_fun is True.

        Parameters
        ----------
        cond_fun: callable
            function of type a -> bool.
        body_fun: callable
            function of type a -> a.
        init_val: any
            value of type a, a type that can be a scalar, array, or any pytree (nested
            Python tuple/list/dict) thereof, representing the initial loop carry value.

        Returns
        -------
        value: any
            The output from the final iteration of body_fun, of type a.

        """
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

# add common docstrings to two of these functions
_put_doc = """Functional interface for array "fancy indexing".

Provides a way to do arr[inds] = vals in a way that works with JAX.

Parameters
----------
arr : array-like
    Array to populate
inds : array-like of int
    Indices to populate
vals : array-like
    Values to insert

Returns
-------
arr : array-like
    Input array with vals inserted at inds.

"""

_sign_doc = """Sign function, but returns 1 for x==0.

Parameters
----------
x : array-like
    array of input values

Returns
-------
y : array-like
    1 where x>=0, -1 where x<0

"""

put.__doc__ = _put_doc
sign.__doc__ = _sign_doc
