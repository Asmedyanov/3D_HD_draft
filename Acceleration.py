import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def triangles_acceleration_function(r_0, r_1, r_2, neighbor_triangle_01_mass, neighbor_triangle_02_mass,
                                    neighbor_triangle_12_mass, P):
    #plt.clf()
    #plt.axis('equal')
    # vectors of sides
    r_01 = r_1 - r_0
    #plt.arrow(r_0[0, 0], r_0[0, 1], r_01[0, 0], r_01[0, 1], length_includes_head=True, width=5.0e-5, color='0')
    r_02 = r_2 - r_0
    #plt.arrow(r_0[0, 0], r_0[0, 1], r_02[0, 0], r_02[0, 1], length_includes_head=True, width=5.0e-5, color='0')
    r_12 = r_2 - r_1
    #plt.arrow(r_1[0, 0], r_1[0, 1], r_12[0, 0], r_12[0, 1], length_includes_head=True, width=5.0e-5, color='0')
    # length of the side
    # S_01 = sqrt((r_01 * r_01).sum(axis=1))
    S_01 = norm(r_01, axis=1)
    S_02 = norm(r_02, axis=1)
    S_12 = norm(r_12, axis=1)
    # projection of 01 side of triangle to 12 side of triangle
    proj_01_12 = r_12 * ((r_01 * r_12).sum(axis=1) / S_12 ** 2)[:, np.newaxis]
    # height of triangle on the side 12
    h_12 = r_01 - proj_01_12
    #plt.arrow(r_1[0, 0], r_1[0, 1], h_12[0, 0], h_12[0, 1], length_includes_head=True, width=5.0e-5,color='0.1')
    #plt.arrow(r_2[0, 0], r_2[0, 1], h_12[0, 0], h_12[0, 1], length_includes_head=True, width=5.0e-5,color='0.1')
    # direction of the force on side 12
    h_norm_12 = h_12 / norm(h_12, axis=1)[:, np.newaxis]

    proj_02_01 = r_01 * ((r_01 * r_02).sum(axis=1) / S_01 ** 2)[:, np.newaxis]
    h_01 = -r_02 + proj_02_01

    #plt.arrow(r_1[0, 0], r_1[0, 1], h_01[0, 0], h_01[0, 1], length_includes_head=True, width=5.0e-5, color='0.2')
    #plt.arrow(r_0[0, 0], r_0[0, 1], h_01[0, 0], h_01[0, 1], length_includes_head=True, width=5.0e-5, color='0.2')

    h_norm_01 = h_01 / norm(h_01, axis=1)[:, np.newaxis]

    proj_12_02 = r_02 * ((r_02 * r_12).sum(axis=1) / S_02 ** 2)[:, np.newaxis]
    h_02 = r_12 - proj_12_02

    #plt.arrow(r_2[0, 0], r_2[0, 1], h_02[0, 0], h_02[0, 1], length_includes_head=True, width=5.0e-5, color='0.3')
    #plt.arrow(r_0[0, 0], r_0[0, 1], h_02[0, 0], h_02[0, 1], length_includes_head=True, width=5.0e-5, color='0.3')

    h_norm_02 = h_02 / norm(h_02, axis=1)[:, np.newaxis]

    # forces on the walls of the triangle
    f_01 = h_norm_01 * (P * S_01)[:, np.newaxis]
    f_02 = h_norm_02 * (P * S_02)[:, np.newaxis]
    f_12 = h_norm_12 * (P * S_12)[:, np.newaxis]

    a_01 = f_01 / np.abs(neighbor_triangle_01mass)[:, np.newaxis]
    a_02 = f_02 / np.abs(neighbor_triangle_02_mass)[:, np.newaxis]
    a_12 = f_12 / np.abs(neighbor_triangle_12_mass)[:, np.newaxis]
    # forces on the points of triangle
    proj_a_01_r_12 = r_12 * ((a_01 * r_12).sum(axis=1) / S_12 ** 2)[:, np.newaxis]
    proj_a_12_r_01 = r_01 * ((a_12 * r_01).sum(axis=1) / S_01 ** 2)[:, np.newaxis]
    a_1=proj_a_01_r_12+proj_a_12_r_01

    proj_a_01_r_02 = r_02 * ((a_01 * r_02).sum(axis=1) / S_02 ** 2)[:, np.newaxis]
    proj_a_02_r_01 = r_01 * ((a_02 * r_01).sum(axis=1) / S_01 ** 2)[:, np.newaxis]
    a_0=proj_a_01_r_02+proj_a_02_r_01

    proj_a_02_r_12 = r_12 * ((a_02 * r_12).sum(axis=1) / S_12 ** 2)[:, np.newaxis]
    proj_a_12_r_02 = r_02 * ((a_12 * r_02).sum(axis=1) / S_02 ** 2)[:, np.newaxis]
    a_2 = proj_a_02_r_12 + proj_a_12_r_02
    #a_0 = (a_01 + a_02)
    #a_1 = (a_01 + a_12)
    #a_2 = (a_02 + a_12)
    #a_0_norm = norm(a_0, axis=1)
    #a_1_norm = norm(a_1, axis=1)
    #a_2_norm = norm(a_2, axis=1)
    #a_mean = (a_0_norm + a_1_norm + a_2_norm) / 3.0
    #s_mean = (S_01 + S_02 + S_12) / 3.0

    #plt.arrow(r_0[0, 0], r_0[0, 1], a_0[0, 0]*s_mean[0]/a_mean[0], a_0[0, 1]*s_mean[0]/a_mean[0], length_includes_head=True, width=5.0e-5, color='r')
    #plt.arrow(r_1[0, 0], r_1[0, 1], a_1[0, 0]*s_mean[0]/a_mean[0], a_1[0, 1]*s_mean[0]/a_mean[0], length_includes_head=True, width=5.0e-5, color='r')
    #plt.arrow(r_2[0, 0], r_2[0, 1], a_2[0, 0]*s_mean[0]/a_mean[0], a_2[0, 1]*s_mean[0]/a_mean[0], length_includes_head=True, width=5.0e-5, color='r')
    triangles_acceleration = np.array([a_0, a_1, a_2])
    #plt.show()
    return triangles_acceleration


def triangles_acceleration_pool(triangles, radiusvector,
                                neighbor_triangle_01_mass,
                                neighbor_triangle_02_mass,
                                neighbor_triangle_12_mass,
                                Pressure, pool, n_proc):
    r_0_list = np.array_split(radiusvector[triangles[:, 0]], n_proc)
    r_1_list = np.array_split(radiusvector[triangles[:, 1]], n_proc)
    r_2_list = np.array_split(radiusvector[triangles[:, 2]], n_proc)
    neighbor_triangle_01_mass_list = np.array_split(neighbor_triangle_01_mass, n_proc)
    neighbor_triangle_02_mass_list = np.array_split(neighbor_triangle_02_mass, n_proc)
    neighbor_triangle_12_mass_list = np.array_split(neighbor_triangle_12_mass, n_proc)
    P_list = np.array_split(Pressure, n_proc)
    proc_array = [pool.apply_async(triangles_acceleration_function, args=(
        r_0_e, r_1_e, r_2_e, neighbor_triangle_01_mass_e, neighbor_triangle_02_mass_e, neighbor_triangle_12_mass_e,
        P_e)) \
                  for
                  r_0_e, r_1_e, r_2_e, neighbor_triangle_01_mass_e, neighbor_triangle_02_mass_e, neighbor_triangle_12_mass_e, P_e
                  in
                  zip(r_0_list, r_1_list, r_2_list, neighbor_triangle_01_mass_list, neighbor_triangle_02_mass_list,
                      neighbor_triangle_12_mass_list, P_list)]
    triangles_acceleration = proc_array[0].get().swapaxes(0, 1)
    for proc in proc_array[1:]:
        tempa = proc.get().swapaxes(0, 1)
        triangles_acceleration = np.concatenate([triangles_acceleration, tempa])
    return triangles_acceleration
    pass


def point_acceleration_func(a: np.array, points_triangles: np.array):
    point_acceleration = np.zeros((len(points_triangles), 2))
    for i, trs in enumerate(points_triangles):
        point_acceleration[i] += a[trs].sum(axis=0)
    return point_acceleration


def point_acceleration_list_pool(points_triangles_0, points_triangles_1, points_triangles_2,
                                 triangles_acceleration: np.array, pool,
                                 N_nuc: int):
    # print('splt')
    points_triangles_0_splited = np.array_split(points_triangles_0, N_nuc)
    points_triangles_1_splited = np.array_split(points_triangles_1, N_nuc)
    points_triangles_2_splited = np.array_split(points_triangles_2, N_nuc)
    proc_list_0 = [pool.apply_async(point_acceleration_func, args=(triangles_acceleration[:, 0], points_triangles)) for
                   points_triangles
                   in points_triangles_0_splited]
    proc_list_1 = [pool.apply_async(point_acceleration_func, args=(triangles_acceleration[:, 1], points_triangles)) for
                   points_triangles
                   in points_triangles_1_splited]
    proc_list_2 = [pool.apply_async(point_acceleration_func, args=(triangles_acceleration[:, 2], points_triangles)) for
                   points_triangles
                   in points_triangles_2_splited]
    points_acceleration_0 = proc_list_0[0].get()
    points_acceleration_1 = proc_list_1[0].get()
    points_acceleration_2 = proc_list_2[0].get()
    for proc_0, proc_1, proc_2 in zip(proc_list_0[1:], proc_list_1[1:], proc_list_2[1:]):
        f_0 = proc_0.get()
        f_1 = proc_1.get()
        f_2 = proc_2.get()
        points_acceleration_0 = np.concatenate([points_acceleration_0, f_0])
        points_acceleration_1 = np.concatenate([points_acceleration_1, f_1])
        points_acceleration_2 = np.concatenate([points_acceleration_2, f_2])
    points_acceleration = points_acceleration_0 + points_acceleration_1 + points_acceleration_2
    return points_acceleration
