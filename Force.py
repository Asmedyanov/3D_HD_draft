import numpy as np


# from numpy.linalg import norm
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as plt3D
# import cupy as cp


def proj(a, b):
    return -b * np.dot(a, b) / np.dot(b, b)


def tetra_force_function_scalar(r_0, r_1, r_2, r_3, P):
    # vectors of sides
    r_01 = r_1 - r_0

    r_02 = r_2 - r_0

    r_03 = r_3 - r_0
    r_12 = r_2 - r_1
    r_23 = r_3 - r_2
    # r_31 = r_1 - r_3
    '''ax = plt.figure().add_subplot(projection='3d')
    x = np.array([r_0[0], r_0[0], r_0[0],r_1[0],r_2[0],r_3[0]])
    y = np.array([r_0[1], r_0[1], r_0[1],r_1[1],r_2[1],r_3[1]])
    z = np.array([r_0[2], r_0[2], r_0[2],r_1[2],r_2[2],r_3[2]])
    dx = np.array([r_01[0], r_02[0], r_03[0],r_12[0],r_23[0],r_31[0]])
    dy = np.array([r_01[1], r_02[1], r_03[1],r_12[1],r_23[1],r_31[1]])
    dz = np.array([r_01[2], r_02[2], r_03[2],r_12[2],r_23[2],r_31[2]])
    ax.quiver(x, y, z, dx, dy, dz)
    #plt.show()'''
    # length of the side
    '''S_01 = norm(r_01)
    S_02 = norm(r_02)
    S_03 = norm(r_03)
    S_12 = norm(r_12)
    S_23 = norm(r_23)
    S_31 = norm(r_31)'''
    # surface area vectors
    S_012 = -np.cross(r_01, r_02) / 2
    S_013 = np.cross(r_01, r_03) / 2
    S_023 = -np.cross(r_02, r_03) / 2
    S_123 = np.cross(r_12, r_23) / 2

    # center
    '''M_012 = (r_0 + r_1 + r_2) / 3.0
    M_013 = (r_0 + r_1 + r_3) / 3.0
    M_023 = (r_0 + r_2 + r_3) / 3.0
    M_123 = (r_1 + r_2 + r_3) / 3.0'''
    '''# surface norms
    n_012 = norm(S_012)
    n_013 = norm(S_013)
    n_023 = norm(S_023)
    n_123 = norm(S_123)
    # surface directions
    d_012 = S_012 / np.sqrt(n_012)
    d_013 = S_013 / np.sqrt(n_013)
    d_023 = S_023 / np.sqrt(n_023)
    d_123 = S_123 / np.sqrt(n_123)

    x = np.array([M_012[0],M_013[0],M_023[0],M_123[0]])
    y = np.array([M_012[1],M_013[1],M_023[1],M_123[1]])
    z = np.array([M_012[2],M_013[2],M_023[2],M_123[2]])
    dx = np.array([d_012[0],d_013[0],d_023[0],d_123[0]])
    dy = np.array([d_012[1],d_013[1],d_023[1],d_123[1]])
    dz = np.array([d_012[2],d_013[2],d_023[2],d_123[2]])

    ax.quiver(x, y, z, dx, dy, dz)
    #plt.show()'''

    # surface projections
    '''proj_012_013 = proj(S_012, S_013)
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
    proj_123_023 = proj(S_123, S_023)'''

    # point vector
    v_0 = S_012 + S_013 + S_023  # proj_012_013 + proj_012_023 + proj_013_012 + proj_013_023 + proj_023_012 + proj_023_013
    v_1 = S_012 + S_013 + S_123  # proj_012_013 + proj_012_123 + proj_013_012 + proj_013_123 + proj_123_012 + proj_123_013
    v_2 = S_012 + S_023 + S_123  # proj_012_023 + proj_012_123 + proj_023_012 + proj_023_123 + proj_123_012 + proj_123_023
    v_3 = S_013 + S_023 + S_123  # proj_013_123 + proj_013_023 + proj_023_013 + proj_023_123 + proj_123_023 + proj_123_013

    # force directions
    '''d_0 = v_0/np.sqrt(n_012)
    d_1 = v_1/np.sqrt(n_013)
    d_2 = v_2/np.sqrt(n_023)
    d_3 = v_3/np.sqrt(n_123)
    x = np.array([r_0[0], r_1[0], r_2[0], r_3[0]])
    y = np.array([r_0[1], r_1[1], r_2[1], r_3[1]])
    z = np.array([r_0[2], r_1[2], r_2[2], r_3[2]])
    dx = np.array([d_0[0], d_1[0], d_2[0], d_3[0]])
    dy = np.array([d_0[1], d_1[1], d_2[1], d_3[1]])
    dz = np.array([d_0[2], d_1[2], d_2[2], d_3[2]])

    ax.quiver(x, y, z, dx, dy, dz)
    plt.show()'''
    # point force
    f_0 = P * v_0
    f_1 = P * v_1
    f_2 = P * v_2
    f_3 = P * v_3

    # tetra_force = np.array([f_0, f_1, f_2, f_3])
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
    Ntr = len(tetras)
    tetras_force_0 = np.zeros((Ntr, 3))
    tetras_force_1 = np.zeros((Ntr, 3))
    tetras_force_2 = np.zeros((Ntr, 3))
    tetras_force_3 = np.zeros((Ntr, 3))
    position = 0
    for proc in proc_array:
        temp_0, temp_1, temp_2, temp_3 = np.array(proc.get())
        l = len(temp_0)
        tetras_force_0[position:position + l] = temp_0
        tetras_force_1[position:position + l] = temp_1
        tetras_force_2[position:position + l] = temp_2
        tetras_force_3[position:position + l] = temp_3
        position += l
    return np.array([tetras_force_0, tetras_force_1, tetras_force_2, tetras_force_3])


def point_force_func(a: np.array, points_tetras: np.array):
    point_force = np.zeros((len(points_tetras), 3))
    for i, trs in enumerate(points_tetras):
        point_force[i] += a[trs].sum(axis=0)
    return point_force


points_tetras = []


def point_force_list_pool(points_tetras_0, points_tetras_1, points_tetras_2, points_tetras_3,
                          tetras_force: np.array, pool,
                          N_nuc: int):
    points_tetras_0_splited = np.array_split(points_tetras_0, N_nuc)
    points_tetras_1_splited = np.array_split(points_tetras_1, N_nuc)
    points_tetras_2_splited = np.array_split(points_tetras_2, N_nuc)
    points_tetras_3_splited = np.array_split(points_tetras_3, N_nuc)
    proc_list_0 = [pool.apply_async(point_force_func, args=(tetras_force[0], points_tetras)) for
                   points_tetras
                   in points_tetras_0_splited]
    proc_list_1 = [pool.apply_async(point_force_func, args=(tetras_force[1], points_tetras)) for
                   points_tetras
                   in points_tetras_1_splited]
    proc_list_2 = [pool.apply_async(point_force_func, args=(tetras_force[2], points_tetras)) for
                   points_tetras
                   in points_tetras_2_splited]
    proc_list_3 = [pool.apply_async(point_force_func, args=(tetras_force[3], points_tetras)) for
                   points_tetras
                   in points_tetras_3_splited]
    Npoint = len(points_tetras_0)
    points_force_0 = np.zeros((Npoint, 3))
    points_force_1 = np.zeros((Npoint, 3))
    points_force_2 = np.zeros((Npoint, 3))
    points_force_3 = np.zeros((Npoint, 3))
    position = 0
    for proc_0, proc_1, proc_2, proc_3 in zip(proc_list_0, proc_list_1, proc_list_2, proc_list_3):
        f_0 = proc_0.get()
        f_1 = proc_1.get()
        f_2 = proc_2.get()
        f_3 = proc_3.get()
        l = len(f_0)
        points_force_0[position:position + l] = f_0
        points_force_1[position:position + l] = f_1
        points_force_2[position:position + l] = f_2
        points_force_3[position:position + l] = f_3
        position += l
    points_force = points_force_0 + points_force_1 + points_force_2 + points_force_3
    return points_force
