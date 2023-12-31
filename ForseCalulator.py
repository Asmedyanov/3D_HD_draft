import numpy as np
from Force import *


def movement(v0, r0, f, m, dt):
    a = f / m
    v = v0 + a * dt
    r = r0 + v * dt
    return v, r


movement_vect = np.vectorize(movement, signature='(n),(n),(n),(),()->(n),(n)')

class ForseCalulator:
    def __init__(self, dt, tetras, points_tetras_0, points_tetras_1, points_tetras_2, points_tetras_3,
                 sector_surface_XY,
                 sector_surface_YZ, sector_surface_XZ, border_outer, point_mass, pool, n_nuc):
        Npoint = len(points_tetras_0)
        self.dt = dt * np.ones(Npoint)
        self.tetras = tetras
        self.points_tetras_0 = points_tetras_0
        self.points_tetras_1 = points_tetras_1
        self.points_tetras_2 = points_tetras_2
        self.points_tetras_3 = points_tetras_3
        self.sector_surface_XY = sector_surface_XY
        self.sector_surface_YZ = sector_surface_YZ
        self.sector_surface_XZ = sector_surface_XZ
        self.border_outer = border_outer
        self.point_mass = point_mass
        self.pool = pool
        self.n_nuc = n_nuc
        self.mass_array = np.array_split(point_mass, n_nuc)
        self.dt_array = np.array_split(self.dt, n_nuc)
        self.Ntr = len(tetras)
        self.tetras_force_0 = np.zeros((self.Ntr, 3))
        self.tetras_force_1 = np.zeros((self.Ntr, 3))
        self.tetras_force_2 = np.zeros((self.Ntr, 3))
        self.tetras_force_3 = np.zeros((self.Ntr, 3))
        self.index_point_list = np.array_split(np.arange(Npoint), n_nuc)
        self.index_tetra_list = np.array_split(np.arange(self.Ntr), n_nuc)
        self.points_tetras_0_splited = np.array_split(points_tetras_0, n_nuc)
        self.points_tetras_1_splited = np.array_split(points_tetras_1, n_nuc)
        self.points_tetras_2_splited = np.array_split(points_tetras_2, n_nuc)
        self.points_tetras_3_splited = np.array_split(points_tetras_3, n_nuc)
        self.tertas_list_0 = np.array_split(self.tetras[:, 0], n_nuc)
        self.tertas_list_1 = np.array_split(self.tetras[:, 1], n_nuc)
        self.tertas_list_2 = np.array_split(self.tetras[:, 2], n_nuc)
        self.tertas_list_3 = np.array_split(self.tetras[:, 3], n_nuc)

        self.points_force_0 = np.zeros((Npoint, 3))
        self.points_force_1 = np.zeros((Npoint, 3))
        self.points_force_2 = np.zeros((Npoint, 3))
        self.points_force_3 = np.zeros((Npoint, 3))
        self.v = np.zeros((Npoint, 3))
        self.r = np.zeros((Npoint, 3))

    def point_acceleration(self, points, pressure):
        tetras_force = self.tetra_force_pool(points, pressure)
        point_force = self.point_force_list_pool(tetras_force)
        point_force[self.sector_surface_XY, 2] *= 0
        point_force[self.sector_surface_YZ, 0] *= 0
        point_force[self.sector_surface_XZ, 1] *= 0
        point_force[self.border_outer] *= 0
        point_acceleration = point_force / self.point_mass[:, np.newaxis]
        return point_acceleration

    def movement_pool(self, v0, r0, f):
        proc_array = [self.pool.apply_async(movement_vect, args=(
            v0[self.index_point_list[i]],
            r0[self.index_point_list[i]],
            f[self.index_point_list[i]],
            self.mass_array[i],
            self.dt_array[i]
        )) for i in range(self.n_nuc)]
        for i in range(self.n_nuc):
            temp_v, temp_r = proc_array[i].get()
            self.v[self.index_point_list[i]] = temp_v
            self.r[self.index_point_list[i]] = temp_r
        return self.v, self.r

    def get_new_vectors(self, points, velocity, pressure):
        tetras_force = self.tetra_force_pool(points, pressure)
        point_force = self.point_force_list_pool(tetras_force)
        point_force[self.sector_surface_XY, 2] *= 0
        point_force[self.sector_surface_YZ, 0] *= 0
        point_force[self.sector_surface_XZ, 1] *= 0
        point_force[self.border_outer] *= 0
        v, r = self.movement_pool(velocity, points, point_force)
        return v, r

    def tetra_force_pool(self, points, pressure):
        proc_array = [self.pool.apply_async(tetra_force_function, args=(
            points[self.tertas_list_0[i]],
            points[self.tertas_list_1[i]],
            points[self.tertas_list_2[i]],
            points[self.tertas_list_3[i]],
            pressure[self.index_tetra_list[i]])) \
                      for i in range(self.n_nuc)]
        for i in range(self.n_nuc):
            temp_0, temp_1, temp_2, temp_3 = proc_array[i].get()
            self.tetras_force_0[self.index_tetra_list[i]] = temp_0
            self.tetras_force_1[self.index_tetra_list[i]] = temp_1
            self.tetras_force_2[self.index_tetra_list[i]] = temp_2
            self.tetras_force_3[self.index_tetra_list[i]] = temp_3
        return np.array([self.tetras_force_0, self.tetras_force_1, self.tetras_force_2, self.tetras_force_3])

    def point_force_list_pool(self, tetras_force):

        proc_list_0 = [self.pool.apply_async(point_force_func, args=(tetras_force[0], points_tetras)) for
                       points_tetras
                       in self.points_tetras_0_splited]
        proc_list_1 = [self.pool.apply_async(point_force_func, args=(tetras_force[1], points_tetras)) for
                       points_tetras
                       in self.points_tetras_1_splited]
        proc_list_2 = [self.pool.apply_async(point_force_func, args=(tetras_force[2], points_tetras)) for
                       points_tetras
                       in self.points_tetras_2_splited]
        proc_list_3 = [self.pool.apply_async(point_force_func, args=(tetras_force[3], points_tetras)) for
                       points_tetras
                       in self.points_tetras_3_splited]
        for i in range(self.n_nuc):
            self.points_force_0[self.index_point_list[i]] = proc_list_0[i].get()
            self.points_force_1[self.index_point_list[i]] = proc_list_1[i].get()
            self.points_force_2[self.index_point_list[i]] = proc_list_2[i].get()
            self.points_force_3[self.index_point_list[i]] = proc_list_3[i].get()
        points_force = self.points_force_0 + self.points_force_1 + self.points_force_2 + self.points_force_3
        return points_force
