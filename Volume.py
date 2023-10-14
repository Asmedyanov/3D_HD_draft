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
        self.index_tetra_list = np.array_split(np.arange(len(tetras)), n_nuc)
        self.tertas_list_0 = np.array_split(self.tetras[:, 0], n_nuc)
        self.tertas_list_1 = np.array_split(self.tetras[:, 1], n_nuc)
        self.tertas_list_2 = np.array_split(self.tetras[:, 2], n_nuc)
        self.tertas_list_3 = np.array_split(self.tetras[:, 3], n_nuc)

    def Volume(self, r):
        proc_list = [
            self.pool.apply_async(volume, args=(
                r[self.tertas_list_0[i]],
                r[self.tertas_list_1[i]],
                r[self.tertas_list_2[i]],
                r[self.tertas_list_3[i]],
                ))
            for i in range(self.n_nuc)]
        for i in range(self.n_nuc):
            self.volume[self.index_tetra_list[i]] = proc_list[i].get()
        return self.volume

