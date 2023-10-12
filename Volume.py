import numpy as np


# volume of tetraeder by coordinates
def volume_scalar(r_a, r_b, r_c, r_d):
    ab = r_b - r_a
    ac = r_c - r_a
    ad = r_d - r_a
    v = np.dot(ab, np.cross(ac, ad)) / 6.0
    return v


volume = np.vectorize(volume_scalar, signature='(n),(n),(n),(n)->()')


def Volume_r_short(tetras, radiusvector):
    r_0, r_1, r_2, r_3 = radiusvector[tetras[:, 0]], radiusvector[tetras[:, 1]], radiusvector[tetras[:, 2]], \
                         radiusvector[tetras[:, 3]]
    return volume(r_0, r_1, r_2, r_3)


class VolumeCalculator:
    def __init__(self, tetras, pool, n_nuc):
        self.tetras = tetras
        self.pool = pool
        self.n_nuc = n_nuc
        self.volume = np.zeros(len(tetras))

    def Volume(self, radiusvector):
        r_0_splited = np.array_split(radiusvector[self.tetras[:, 0]], self.n_nuc)
        r_1_splited = np.array_split(radiusvector[self.tetras[:, 1]], self.n_nuc)
        r_2_splited = np.array_split(radiusvector[self.tetras[:, 2]], self.n_nuc)
        r_3_splited = np.array_split(radiusvector[self.tetras[:, 3]], self.n_nuc)
        proc_list = [
            self.pool.apply_async(volume, args=(r_0_splited[i], r_1_splited[i], r_2_splited[i], r_3_splited[i])) for
            i
            in range(self.n_nuc)]
        position = 0
        for proc in proc_list:
            temp = proc.get()
            l = len(temp)
            self.volume[position:position + l] = temp
            position += l
        return self.volume

