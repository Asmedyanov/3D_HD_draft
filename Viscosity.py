from Area import *
from numpy import ones, sqrt, square, abs, where,dot
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def Viscosity(x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3, ax1, ax2, ax3, ay1, ay2, ay3, SS, RO):
    xc = (x1 + x2 + x3) / 3
    yc = (y1 + y2 + y3) / 3
    # lc = sqrt(square(xc) + square(yc))
    axc = (ax1 + ax2 + ax3) / 3
    ayc = (ay1 + ay2 + ay3) / 3
    abs_acs = sqrt(square(axc) + square(ayc))
    abs_acs = abs_acs + ones(abs_acs.size) * (abs_acs < 1.e-7)
    cosThetta = axc / abs_acs
    sinThetta = ayc / abs_acs
    DVXDX = -2 * Area((vx2 + vx3), (vx1 + vx3), (vx2 + vx1), y1, y2, y3)
    DVYDY = 2 * Area((vy2 + vy3), (vy1 + vy3), (vy2 + vy1), x1, x2, x3)
    DVXDY = 2 * Area((vx2 + vx3), (vx1 + vx3), (vx2 + vx1), x1, x2, x3)
    DVYDX = -2 * Area((vy2 + vy3), (vy1 + vy3), (vy2 + vy1), y1, y2, y3)
    DSDT = DVXDX * square(cosThetta) + DVYDY * square(sinThetta) + (DVXDY + DVYDX) * cosThetta * sinThetta
    SUM = abs((y1 - yc) * cosThetta - (x1 - xc) * sinThetta)
    SUM = SUM + abs((y2 - yc) * cosThetta - (x2 - xc) * sinThetta)
    SUM = SUM + abs((y3 - yc) * cosThetta - (x3 - xc) * sinThetta)
    SUM = SUM + ones(SUM.size) * (abs(SUM) < 1.e-7)
    v = (DSDT < 0) * (SS * RO * abs(DSDT / SUM))
    return v


def Viscosity_short(triangles, radiusvector, velocity, acceleration, SS, RO):
    x1, x2, x3 = radiusvector[triangles[:, 0], 0], radiusvector[triangles[:, 1], 0], radiusvector[triangles[:, 2], 0]
    y1, y2, y3 = radiusvector[triangles[:, 0], 1], radiusvector[triangles[:, 1], 1], radiusvector[triangles[:, 2], 1]
    vx1, vx2, vx3 = velocity[triangles[:, 0], 0], velocity[triangles[:, 1], 0], velocity[triangles[:, 2], 0]
    vy1, vy2, vy3 = velocity[triangles[:, 0], 1], velocity[triangles[:, 1], 1], velocity[triangles[:, 2], 1]
    ax1, ax2, ax3 = acceleration[triangles[:, 0], 0], acceleration[triangles[:, 1], 0], acceleration[triangles[:, 2], 0]
    ay1, ay2, ay3 = acceleration[triangles[:, 0], 1], acceleration[triangles[:, 1], 1], acceleration[triangles[:, 2], 1]
    xc = (x1 + x2 + x3) / 3.0
    yc = (y1 + y2 + y3) / 3.0
    # lc = sqrt(square(xc) + square(yc))
    axc = (ax1 + ax2 + ax3) / 3
    ayc = (ay1 + ay2 + ay3) / 3
    abs_acs = sqrt(square(axc) + square(ayc))
    abs_acs = abs_acs + ones(abs_acs.size) * (abs_acs < 1.e-7)
    cosThetta = axc / abs_acs
    sinThetta = ayc / abs_acs
    DVXDX = -2 * Area((vx2 + vx3), (vx1 + vx3), (vx2 + vx1), y1, y2, y3)
    DVYDY = 2 * Area((vy2 + vy3), (vy1 + vy3), (vy2 + vy1), x1, x2, x3)
    DVXDY = 2 * Area((vx2 + vx3), (vx1 + vx3), (vx2 + vx1), x1, x2, x3)
    DVYDX = -2 * Area((vy2 + vy3), (vy1 + vy3), (vy2 + vy1), y1, y2, y3)
    DSDT = DVXDX * square(cosThetta) + DVYDY * square(sinThetta) + (DVXDY + DVYDX) * cosThetta * sinThetta
    SUM = abs((y1 - yc) * cosThetta - (x1 - xc) * sinThetta)
    SUM = SUM + abs((y2 - yc) * cosThetta - (x2 - xc) * sinThetta)
    SUM = SUM + abs((y3 - yc) * cosThetta - (x3 - xc) * sinThetta)
    SUM = SUM + ones(SUM.size) * (abs(SUM) < 1.e-7)
    v = (DSDT < 0) * (SS * RO * abs(DSDT / SUM))
    return v


def Viscosity_mu_r(r_0, r_1, r_2, r_3, v_0, v_1, v_2, v_3):
    A0 = Area_r_vect(r_1, r_2, r_3)
    A1 = Area_r_vect(r_0, r_2, r_3)
    A2 = Area_r_vect(r_0, r_1, r_3)
    A3 = Area_r_vect(r_0, r_1, r_2)
    u_0 = v_0 - (v_1 + v_2 + v_3) / 3.0
    u_1 = v_1 - (v_0 + v_2 + v_3) / 3.0
    u_2 = v_2 - (v_0 + v_1 + v_3) / 3.0
    u_3 = v_3 - (v_0 + v_1 + v_2) / 3.0
    u_0_perp = dot(u_0, A0) / norm(A0)
    u_1_perp = dot(u_1, A1) / norm(A1)
    u_2_perp = dot(u_2, A2) / norm(A2)
    u_3_perp = dot(u_3, A3) / norm(A3)
    SUM = 0
    SUM += dot(A0, A0) * (dot(u_0, u_0) - dot(u_0_perp, u_0_perp))
    SUM += dot(A1, A1) * (dot(u_1, u_1) - dot(u_1_perp, u_1_perp))
    SUM += dot(A2, A2) * (dot(u_2, u_2) - dot(u_2_perp, u_2_perp))
    SUM += dot(A3, A3) * (dot(u_3, u_3) - dot(u_3_perp, u_3_perp))
    return SUM


Viscosity_mu_Vector = np.vectorize(Viscosity_mu_r, signature='(n),(n),(n),(n),(n),(n),(n),(n)->()')


def Viscosity_r(r_0, r_1, r_2, r_3, v_0, v_1, v_2, v_3, a_0, a_1, a_2, a_3, SS, RO):
    rc = (r_0 + r_1 + r_2 + r_3) / 4.0
    # lc = sqrt(square(xc) + square(yc))
    ac = (a_0 + a_1 + a_2 + a_3) / 4.0
    abs_acs = norm(ac)
    abs_acs = where(abs_acs < 1.0e-7, abs_acs + 1.0e-7, abs_acs)
    # abs_acs = abs_acs + ones(abs_acs.size) * (abs_acs < 1.e-7)
    n_ac = ac / abs_acs
    v_no_0 = v_1 + v_2 + v_3
    v_no_1 = v_0 + v_2 + v_3
    v_no_2 = v_0 + v_1 + v_3
    v_no_3 = v_0 + v_1 + v_2
    # DVXDX = -2 * Area((v_1[0] + v_2[0]), (v_0[0] + v_2[0]), (v_1[0] + v_0[0]), r_0[1], r_1[1], r_2[1])
    # DVYDY = 2 * Area((v_1[1] + v_2[1]), (v_0[1] + v_2[1]), (v_1[1] + v_0[1]), r_0[0], r_1[0], r_2[0])

    # DVXDY = 2 * Area((v_1[0] + v_2[0]), (v_0[0] + v_2[0]), (v_1[0] + v_0[0]), r_0[0], r_1[0], r_2[0])

    # DVYDX = -2 * Area((v_1[1] + v_2[1]), (v_0[1] + v_2[1]), (v_1[1] + v_0[1]), r_0[1], r_1[1], r_2[1])
    DVXDX = -Area_r([v_no_0[0], r_0[1]], [v_no_1[0], r_1[1]], [v_no_2[0], r_2[1]])
    DVYDY = Area_r([v_no_0[1], r_0[0]], [v_no_1[1], r_1[0]], [v_no_2[1], r_2[0]])
    DVZDZ = Area_r([v_no_0[2], r_0[0]], [v_no_1[2], r_1[0]], [v_no_2[2], r_2[0]])

    DVYDX = -Area_r([v_no_0[1], r_0[1]], [v_no_1[1], r_1[1]], [v_no_2[1], r_2[1]])
    DVXDY = Area_r([v_no_0[0], r_0[0]], [v_no_1[0], r_1[0]], [v_no_2[0], r_2[0]])
    DVXDZ = Area_r([v_no_0[0], r_0[2]], [v_no_1[0], r_1[2]], [v_no_2[0], r_2[2]])
    DVYDZ = Area_r([v_no_0[1], r_0[2]], [v_no_1[1], r_1[2]], [v_no_2[1], r_2[2]])
    DVZDX = -Area_r([v_no_0[2], r_0[1]], [v_no_1[2], r_1[1]], [v_no_2[2], r_2[1]])
    DVZDY = Area_r([v_no_0[2], r_0[0]], [v_no_1[2], r_1[0]], [v_no_2[2], r_2[0]])
    DSDT = DVXDX * square(n_ac[0]) + \
           DVYDY * square(n_ac[1]) + \
           DVZDZ * square(n_ac[2]) + \
           (DVXDY + DVYDX) * n_ac[0] * n_ac[1] + \
           (DVXDZ + DVZDX) * n_ac[0] * n_ac[2] + \
           (DVYDZ + DVZDY) * n_ac[1] * n_ac[2]
    DSDT *= 2.0
    d_0 = r_0 - rc
    d_1 = r_1 - rc
    d_2 = r_2 - rc
    d_3 = r_3 - rc
    SUM = abs(d_0[1] * n_ac[0] - d_0[0] * n_ac[1])
    SUM += abs(d_0[2] * n_ac[0] - d_0[0] * n_ac[2])
    SUM += abs(d_0[2] * n_ac[1] - d_0[1] * n_ac[2])

    SUM += abs(d_1[1] * n_ac[0] - d_1[0] * n_ac[1])
    SUM += abs(d_1[2] * n_ac[0] - d_1[0] * n_ac[2])
    SUM += abs(d_1[2] * n_ac[1] - d_1[1] * n_ac[2])

    SUM += abs(d_2[1] * n_ac[0] - d_2[0] * n_ac[1])
    SUM += abs(d_2[2] * n_ac[0] - d_2[0] * n_ac[2])
    SUM += abs(d_2[2] * n_ac[1] - d_2[1] * n_ac[2])

    SUM += abs(d_3[1] * n_ac[0] - d_3[0] * n_ac[1])
    SUM += abs(d_3[2] * n_ac[0] - d_3[0] * n_ac[2])
    SUM += abs(d_3[2] * n_ac[1] - d_3[1] * n_ac[2])

    SUM = where(SUM < 1.0e-7, SUM + 1.0e-7, SUM)
    v = (DSDT < 0) * (SS * RO * abs(DSDT / SUM))
    return v


Viscosity_vector = np.vectorize(Viscosity_r, signature='(n),(n),(n),(n),(n),(n),(n),(n),(n),(n),(n),(n),(),()->()')


def Viscosity_short_new(tetras, r, r_dot, r_dot_dot, SS, RO):
    r_0 = r[tetras[:, 0]]
    r_1 = r[tetras[:, 1]]
    r_2 = r[tetras[:, 2]]
    r_3 = r[tetras[:, 3]]
    r_dot_0 = r_dot[tetras[:, 0]]
    r_dot_1 = r_dot[tetras[:, 1]]
    r_dot_2 = r_dot[tetras[:, 2]]
    r_dot_3 = r_dot[tetras[:, 3]]
    r_dot_dot_0 = r_dot_dot[tetras[:, 0]]
    r_dot_dot_1 = r_dot_dot[tetras[:, 1]]
    r_dot_dot_2 = r_dot_dot[tetras[:, 2]]
    r_dot_dot_3 = r_dot_dot[tetras[:, 3]]
    visc = Viscosity_vector(
        r_0,
        r_1,
        r_2,
        r_3,
        r_dot_0,
        r_dot_1,
        r_dot_2,
        r_dot_3,
        r_dot_dot_0,
        r_dot_dot_1,
        r_dot_dot_2,
        r_dot_dot_3,
        SS,
        RO
    )
    return visc


def Viscosity_short_mu(tetras, r, r_dot):
    r_0 = r[tetras[:, 0]]
    r_1 = r[tetras[:, 1]]
    r_2 = r[tetras[:, 2]]
    r_3 = r[tetras[:, 3]]
    r_dot_0 = r_dot[tetras[:, 0]]
    r_dot_1 = r_dot[tetras[:, 1]]
    r_dot_2 = r_dot[tetras[:, 2]]
    r_dot_3 = r_dot[tetras[:, 3]]
    visc = Viscosity_mu_Vector(
        r_0,
        r_1,
        r_2,
        r_3,
        r_dot_0,
        r_dot_1,
        r_dot_2,
        r_dot_3
    )
    return visc


def Viscosity_short_mu_pool(tetras, r, r_dot, pool, n_nuc):
    r_0 = np.array_split(r[tetras[:, 0]], n_nuc)
    r_1 = np.array_split(r[tetras[:, 1]], n_nuc)
    r_2 = np.array_split(r[tetras[:, 2]], n_nuc)
    r_3 = np.array_split(r[tetras[:, 3]], n_nuc)
    r_dot_0 = np.array_split(r_dot[tetras[:, 0]], n_nuc)
    r_dot_1 = np.array_split(r_dot[tetras[:, 1]], n_nuc)
    r_dot_2 = np.array_split(r_dot[tetras[:, 2]], n_nuc)
    r_dot_3 = np.array_split(r_dot[tetras[:, 3]], n_nuc)
    proc_list = [pool.apply_async(Viscosity_mu_Vector, args=(
        r_0[i],
        r_1[i],
        r_2[i],
        r_3[i],
        r_dot_0[i],
        r_dot_1[i],
        r_dot_2[i],
        r_dot_3[i]
    )) for i in range(n_nuc)]
    visc = np.ones(len(tetras))
    position = 0
    for proc in proc_list:
        temp = proc.get()
        l = len(temp)
        visc[position:position + l] = temp
        position += l
    # plt.clf()
    # plt.plot(visc)
    # plt.show()
    return visc
