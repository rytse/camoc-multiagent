import numpy as np

def identity_factory():
    def identity(x):
        return x
    return identity

def angle2cart_factory():
    '''
    This function returns a lifting function that takes variables of the form

        theta in [-pi, pi]

    to coordinates of the lifted manifold

        (x, y),

        x = cos(theta)
        y = sin(theta)
    '''

    def angle2cart(theta):
        return np.array([np.cos(theta), np.sin(theta)])

    return angle2cart

def halfinterval2slack_factory(a):
    '''
    This function returns a lifting function that takes variables of the form

        x in [0, a]

    to coordinates of the lifted manifold

        (x, alpha, beta),

        alpha = sqrt(1 - x)
        beta = sqrt(x)

    such that the slack variable equality constraints are equivalent to the
    inequailty constraints 0 <= x <= a
    '''

    def halfinterval2slack(x):
        return np.array([x, np.sqrt(a - x), np.sqrt(x)])

    return halfinterval2slack

def fullinterval2slack_factory(a):
    '''
    This function returns a lifting function that takes variables of the form

        x in [-a, a]

    to coordinates of the lifted manifold

        (x, alpha, beta),

        alpha = sqrt(a - x)
        beta = sqrt(a + x)

    such that the slack variable equality constraints are equivalent to the
    inequailty constraints -a <= x <= a
    '''

    def fullinterval2slack(x):
        return np.array([x, np.sqrt(a - x), np.sqrt(a + x)])

    return fullinterval2slack

def obs_native2mfd(native, schema):
    mfd = np.zeros(sum([s["mfd_size"] for s in schema.values()]))

    idx_n = 0
    idx_m = 0
    for field in schema.keys():
        mfd[idx_m : idx_m + schema[field]["mfd_size"]] = schema[field]["conversion_lambda"](native[idx_n : idx_n + schema[field]["native_size"]])
        idx_n += schema[field]["native_size"]
        idx_m += schema[field]["mfd_size"]

    return mfd


