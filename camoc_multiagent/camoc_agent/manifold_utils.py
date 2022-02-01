import numpy as np


def pos_sign(x):
    y = np.sign(x)
    y[x == 0] = 1
    return y


def slack_up(x):
    return pos_sign(x) * np.log(np.power(x + pos_sign(x), 10))


def slack_down(y):
    sgn = np.sign(y)
    return np.power(np.exp(sgn * y), 1 / 10) - sgn


def halfinterval2slack(x, a):
    y = np.zeros(x.size * 3)
    y[0 : x.size] = x
    y[x.size : x.size * 2] = slack_up(a - x)
    y[x.size * 2 :] = slack_up(x)
    return y


def fullinterval2slack(x, a):
    y = np.zeros(x.size * 3)
    y[0 : x.size] = x
    y[x.size : x.size * 2] = slack_up(a - x)
    y[x.size * 2 :] = slack_up(a + x)
    return y


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
