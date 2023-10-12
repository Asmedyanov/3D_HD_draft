import numpy as np
from Force import *


class ForseCalulator:
    def __init__(self, tetras, points_tetras_0, points_tetras_1, points_tetras_2, points_tetras_3, sector_surface_XY,
                 sector_surface_YZ, sector_surface_XZ, border_outer, point_mass, pool, n_nuc):
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
        self.Ntr = len(tetras)
        self.tetras_force_0 = np.zeros((self.Ntr, 3))
        self.tetras_force_1 = np.zeros((self.Ntr, 3))
        self.tetras_force_2 = np.zeros((self.Ntr, 3))
        self.tetras_force_3 = np.zeros((self.Ntr, 3))
        self.points_tetras_0_splited = np.array_split(points_tetras_0, n_nuc)
        self.points_tetras_1_splited = np.array_split(points_tetras_1, n_nuc)
        self.points_tetras_2_splited = np.array_split(points_tetras_2, n_nuc)
        self.points_tetras_3_splited = np.array_split(points_tetras_3, n_nuc)
        Npoint = len(self.points_tetras_0)
        self.points_force_0 = np.zeros((Npoint, 3))
        self.points_force_1 = np.zeros((Npoint, 3))
        self.points_force_2 = np.zeros((Npoint, 3))
        self.points_force_3 = np.zeros((Npoint, 3))

    def point_acceleration(self, points, pressure):
        tetras_force = self.tetra_force_pool(points, pressure)
        point_force = self.point_force_list_pool(tetras_force)
        point_force[self.sector_surface_XY, 2] *= 0
        point_force[self.sector_surface_YZ, 0] *= 0
        point_force[self.sector_surface_XZ, 1] *= 0
        point_force[self.border_outer] *= 0
        point_acceleration = point_force / self.point_mass[:, np.newaxis]
        return point_acceleration

    def tetra_force_pool(self, points, pressure):
        r_0_list = np.array_split(points[self.tetras[:, 0]], self.n_nuc)
        r_1_list = np.array_split(points[self.tetras[:, 1]], self.n_nuc)
        r_2_list = np.array_split(points[self.tetras[:, 2]], self.n_nuc)
        r_3_list = np.array_split(points[self.tetras[:, 3]], self.n_nuc)
        P_list = np.array_split(pressure, self.n_nuc)
        proc_array = [self.pool.apply_async(tetra_force_function, args=(r_0_e, r_1_e, r_2_e, r_3_e, P_e)) \
                      for r_0_e, r_1_e, r_2_e, r_3_e, P_e in zip(r_0_list, r_1_list, r_2_list, r_3_list, P_list)]
        position = 0
        for proc in proc_array:
            temp_0, temp_1, temp_2, temp_3 = np.array(proc.get())
            l = len(temp_0)
            self.tetras_force_0[position:position + l] = temp_0
            self.tetras_force_1[position:position + l] = temp_1
            self.tetras_force_2[position:position + l] = temp_2
            self.tetras_force_3[position:position + l] = temp_3
            position += l
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
        position = 0
        for proc_0, proc_1, proc_2, proc_3 in zip(proc_list_0, proc_list_1, proc_list_2, proc_list_3):
            f_0 = proc_0.get()
            f_1 = proc_1.get()
            f_2 = proc_2.get()
            f_3 = proc_3.get()
            l = len(f_0)
            self.points_force_0[position:position + l] = f_0
            self.points_force_1[position:position + l] = f_1
            self.points_force_2[position:position + l] = f_2
            self.points_force_3[position:position + l] = f_3
            position += l
        points_force = self.points_force_0 + self.points_force_1 + self.points_force_2 + self.points_force_3
        return points_force
