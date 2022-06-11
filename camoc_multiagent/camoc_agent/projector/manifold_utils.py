from functools import partial

# from jax import jit
import numpy as np


def halfinterval2slack(x, a):
    y = np.zeros((x.shape[0], x.shape[1] * 3))
    y[:, : x.shape[1]] = x
    y[:, x.shape[1] : 2 * x.shape[1]] = np.log(a - x + 1)
    y[:, 2 * x.shape[1] :] = np.log(x + 1)

    return y


"""
#@partial(jit, static_argnums=(1,))
def halfinterval2slack(x, a):
    y = np.zeros((x.shape[0], x.shape[1] * 3))

    y.at[:, : x.shape[1]].set(x)
    y.at[:, x.shape[1] : 2 * x.shape[1]].set(np.log(a - x + 1))
    y.at[:, 2 * x.shape[1] :].set(np.log(x + 1))

    return y


"""


def fullinterval2slack(x: np.ndarray, a):
    y = np.zeros(x.size * 3)

    y[0 : x.size] = x
    y[x.size : x.size * 2] = np.log(a - x + 1)
    y[x.size * 2 :] = np.log(a + x + 1)


"""
#@jit
def fullinterval2slack(x, a):
    y = np.zeros(x.size * 3)
    y.at[0 : x.size].set(x)
    y.at[x.size : x.size * 2].set(np.log(a - x + 1))
    y.at[x.size * 2 :].set(np.log(a + x + 1))

    return y

"""


def identity_factory():
    def identity(x):
        return x

    return identity


def angle2cart_factory():
    """
    This function returns a lifting function that takes variables of the form

        theta in [-pi, pi]

    to coordinates of the lifted manifold

        (x, y),

        x = cos(theta)
        y = sin(theta)
    """

    def angle2cart(theta):
        return np.array([np.cos(theta), np.sin(theta)]).flatten()

    return angle2cart


def halfinterval2slack_factory(a):
    """
    This function returns a lifting function that takes variables of the form

        x in [0, a]

    to coordinates of the lifted manifold

        (x, alpha, beta),

        alpha = sqrt(a - x)
        beta = sqrt(x)

    such that the slack variable equality constraints are equivalent to the
    inequailty constraints 0 <= x <= a
    """

    return lambda x: halfinterval2slack(x, a)


def fullinterval2slack_factory(a):
    """
    This function returns a lifting function that takes variables of the form

        x in [-a, a]

    to coordinates of the lifted manifold

        (x, alpha, beta),

        alpha = sqrt(a - x)
        beta = sqrt(a + x)

    such that the slack variable equality constraints are equivalent to the
    inequailty constraints -a <= x <= a
    """

    return lambda x: fullinterval2slack(x, a)


"""
def obs_native2mfd(native, schema):
    mfd = np.zeros(sum([s["mfd_size"] for s in schema.values()]))

    idx_n = 0
    idx_m = 0
    for field in schema.keys():
        mfd[idx_m : idx_m + schema[field]["mfd_size"]] = schema[field][
            "conversion_lambda"
        ](native[idx_n : idx_n + schema[field]["native_size"]])
        idx_n += schema[field]["native_size"]
        idx_m += schema[field]["mfd_size"]

    return mfd


"""
