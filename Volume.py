import numpy as np


# volume of tetraeder by coordinates
def volume_scalar(r_a, r_b, r_c, r_d):
    ab = r_b - r_a
    ac = r_c - r_a
    ad = r_d - r_a
    v = np.dot(ab, np.cross(ac, ad))
    return v


volume = np.vectorize(volume_scalar,signature='(n),(n),(n),(n)->()')
