import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def proj(a, b):
    return b * np.dot(a, b) / np.dot(b, b)


def tetra_force_function_scalar(r_0, r_1, r_2, r_3, P):
    # vectors of sides
    r_01 = r_1 - r_0

    r_02 = r_2 - r_0

    r_03 = r_3 - r_0
    r_12 = r_2 - r_1
    r_23 = r_3 - r_2
    r_31 = r_1 - r_3

    # length of the side
    '''S_01 = norm(r_01)
    S_02 = norm(r_02)
    S_03 = norm(r_03)
    S_12 = norm(r_12)
    S_23 = norm(r_23)
    S_31 = norm(r_31)'''
    # surface area vectors
    S_012 = np.cross(r_01, r_02)
    S_013 = np.cross(r_01, r_03)
    S_023 = np.cross(r_02, r_03)
    S_123 = np.cross(r_12, r_23)
    # surface norms
    '''n_012 = norm(S_012)
    n_013 = norm(S_013)
    n_023 = norm(S_023)
    n_123 = norm(S_123)'''
    # surface directions
    '''d_012 = S_012 / n_012
    d_013 = S_013 / n_013
    d_023 = S_023 / n_023
    d_123 = S_123 / n_123'''
    # surface projections
    proj_012_013 = proj(S_012, S_013)
    proj_012_023 = proj(S_012, S_023)
    proj_012_123 = proj(S_012, S_123)
    proj_013_012 = proj(S_013, S_012)
    proj_013_023 = proj(S_013, S_023)
    proj_013_123 = proj(S_013, S_123)
    proj_023_012 = proj(S_023, S_012)
    proj_023_013 = proj(S_023, S_013)
    proj_023_123 = proj(S_023, S_123)
    proj_123_012 = proj(S_123, S_012)
    proj_123_013 = proj(S_123, S_013)
    proj_123_023 = proj(S_123, S_023)

    # point vector
    v_0 = proj_012_013 + proj_012_023 + proj_013_012 + proj_013_023 + proj_023_012 + proj_023_013
    v_1 = proj_012_013 + proj_012_123 + proj_013_012 + proj_013_123 + proj_123_012 + proj_123_013
    v_2 = proj_012_023 + proj_012_123 + proj_023_012 + proj_023_123 + proj_123_012 + proj_123_023
    v_3 = proj_013_123 + proj_013_023 + proj_023_013 + proj_023_123 + proj_123_023 + proj_123_013
    # point force
    f_0 = P * v_0
    f_1 = P * v_1
    f_2 = P * v_2
    f_3 = P * v_3

    #tetra_force = np.array([f_0, f_1, f_2, f_3])
    return f_0, f_1, f_2, f_3


tetra_force_function = np.vectorize(tetra_force_function_scalar, signature='(n),(n),(n),(n),()->(n),(n),(n),(n)')


def tetra_force_pool(tetras, radiusvector, Pressure, pool, n_proc):
    r_0_list = np.array_split(radiusvector[tetras[:, 0]], n_proc)
    r_1_list = np.array_split(radiusvector[tetras[:, 1]], n_proc)
    r_2_list = np.array_split(radiusvector[tetras[:, 2]], n_proc)
    r_3_list = np.array_split(radiusvector[tetras[:, 3]], n_proc)
    P_list = np.array_split(Pressure, n_proc)
    proc_array = [pool.apply_async(tetra_force_function, args=(r_0_e, r_1_e, r_2_e, r_3_e, P_e)) \
                  for r_0_e, r_1_e, r_2_e, r_3_e, P_e in zip(r_0_list, r_1_list, r_2_list, r_3_list, P_list)]
    tetras_force = np.array(proc_array[0].get())
    tetras_force = tetras_force.swapaxes(0, 1)
    for proc in proc_array[1:]:
        tempa = np.array(proc.get())
        tempa = tempa.swapaxes(0,1)
        tetras_force = np.concatenate([tetras_force, tempa])
    return tetras_force
    pass


def point_force_func(a: np.array, points_tetras: np.array):
    point_force = np.zeros((len(points_tetras), 3))
    for i, trs in enumerate(points_tetras):
        point_force[i] += a[trs].sum(axis=0)
    return point_force


def point_force_list_pool(points_tetras_0, points_tetras_1, points_tetras_2, points_tetras_3,
                          tetras_force: np.array, pool,
                          N_nuc: int):
    points_tetras_0_splited = np.array_split(points_tetras_0, N_nuc)
    points_tetras_1_splited = np.array_split(points_tetras_1, N_nuc)
    points_tetras_2_splited = np.array_split(points_tetras_2, N_nuc)
    points_tetras_3_splited = np.array_split(points_tetras_3, N_nuc)
    proc_list_0 = [pool.apply_async(point_force_func, args=(tetras_force[:, 0], points_tetras)) for
                   points_tetras
                   in points_tetras_0_splited]
    proc_list_1 = [pool.apply_async(point_force_func, args=(tetras_force[:, 1], points_tetras)) for
                   points_tetras
                   in points_tetras_1_splited]
    proc_list_2 = [pool.apply_async(point_force_func, args=(tetras_force[:, 2], points_tetras)) for
                   points_tetras
                   in points_tetras_2_splited]
    proc_list_3 = [pool.apply_async(point_force_func, args=(tetras_force[:, 3], points_tetras)) for
                   points_tetras
                   in points_tetras_3_splited]
    points_force_0 = proc_list_0[0].get()
    points_force_1 = proc_list_1[0].get()
    points_force_2 = proc_list_2[0].get()
    points_force_3 = proc_list_3[0].get()
    for proc_0, proc_1, proc_2, proc_3 in zip(proc_list_0[1:], proc_list_1[1:], proc_list_2[1:], proc_list_3[1:]):
        f_0 = proc_0.get()
        f_1 = proc_1.get()
        f_2 = proc_2.get()
        f_3 = proc_3.get()
        points_force_0 = np.concatenate([points_force_0, f_0])
        points_force_1 = np.concatenate([points_force_1, f_1])
        points_force_2 = np.concatenate([points_force_2, f_2])
        points_force_3 = np.concatenate([points_force_3, f_3])
    points_force = points_force_0 + points_force_1 + points_force_2 + points_force_3
    return points_force
