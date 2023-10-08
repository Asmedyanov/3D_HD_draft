from Area import *
from numpy import ones, sqrt, square, abs, where
import numpy as np
from numpy.linalg import norm


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
    #DVXDX = -2 * Area((v_1[0] + v_2[0]), (v_0[0] + v_2[0]), (v_1[0] + v_0[0]), r_0[1], r_1[1], r_2[1])
    #DVYDY = 2 * Area((v_1[1] + v_2[1]), (v_0[1] + v_2[1]), (v_1[1] + v_0[1]), r_0[0], r_1[0], r_2[0])

    #DVXDY = 2 * Area((v_1[0] + v_2[0]), (v_0[0] + v_2[0]), (v_1[0] + v_0[0]), r_0[0], r_1[0], r_2[0])

    #DVYDX = -2 * Area((v_1[1] + v_2[1]), (v_0[1] + v_2[1]), (v_1[1] + v_0[1]), r_0[1], r_1[1], r_2[1])
    DVXDX = -2 * Area_r([v_no_0[0], r_0[1]], [v_no_1[0], r_1[1]], [v_no_2[0], r_2[1]])
    DVYDY = 2 * Area_r([v_no_0[1], r_0[0]], [v_no_1[1], r_1[0]], [v_no_2[1], r_2[0]])
    DVZDZ = 2 * Area_r([v_no_0[2], r_0[0]], [v_no_1[2], r_1[0]], [v_no_2[2], r_2[0]])
    DVYDX = -2 * Area_r([v_no_0[1], r_0[1]], [v_no_1[1], r_1[1]], [v_no_2[1], r_2[1]])
    DVXDY = 2 * Area_r([v_no_0[0], r_0[0]], [v_no_1[0], r_1[0]], [v_no_2[0], r_2[0]])
    DSDT = DVXDX * square(n_ac[:, 0]) + DVYDY * square(n_ac[:, 1]) + (DVXDY + DVYDX) * n_ac[:, 0] * n_ac[:, 1]
    d_0 = r_0 - rc
    d_1 = r_1 - rc
    d_2 = r_2 - rc
    d_3 = r_3 - rc
    SUM = abs(d_0[:, 1] * n_ac[:, 0] - d_0[:, 0] * n_ac[:, 1])
    SUM += abs(d_0[:, 2] * n_ac[:, 0] - d_0[:, 0] * n_ac[:, 2])
    SUM += abs(d_0[:, 2] * n_ac[:, 1] - d_0[:, 1] * n_ac[:, 2])

    SUM += abs(d_1[:, 1] * n_ac[:, 0] - d_1[:, 0] * n_ac[:, 1])
    SUM += abs(d_1[:, 2] * n_ac[:, 0] - d_1[:, 0] * n_ac[:, 2])
    SUM += abs(d_1[:, 2] * n_ac[:, 1] - d_1[:, 1] * n_ac[:, 2])

    SUM += abs(d_2[:, 1] * n_ac[:, 0] - d_2[:, 0] * n_ac[:, 1])
    SUM += abs(d_2[:, 2] * n_ac[:, 0] - d_2[:, 0] * n_ac[:, 2])
    SUM += abs(d_2[:, 2] * n_ac[:, 1] - d_2[:, 1] * n_ac[:, 2])

    SUM += abs(d_3[:, 1] * n_ac[:, 0] - d_3[:, 0] * n_ac[:, 1])
    SUM += abs(d_3[:, 2] * n_ac[:, 0] - d_3[:, 0] * n_ac[:, 2])
    SUM += abs(d_3[:, 2] * n_ac[:, 1] - d_3[:, 1] * n_ac[:, 2])

    SUM = where(SUM < 1.0e-7, SUM + 1.0e-7, SUM)
    v = (DSDT < 0) * (SS * RO * abs(DSDT / SUM))
    return v
