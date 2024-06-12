import jax.numpy as jnp
from jax import jit, grad
import numpy as np
from numpy import array 
import time

@jit
def DubinsVehicle_Dynamics(x, u, barry, R, goal):
    r = 0.2
    d = 0.1

    xd0 = r * jnp.cos(x[2]) * (u[0] + u[1])/2
    xd1 = r * jnp.sin(x[2]) * (u[0] + u[1])/2
    xd2 = r / (2 * d) * (u[0] - u[1])

    xd = []
    for i in range(len(R)):
        h = (goal[0] - barry[2 * i]) ** 2 + (goal[1] - barry[2 * i + 1]) ** 2 - R[i] ** 2
        xd.append(1 / h)
    xd = jnp.hstack((xd0, xd1, xd2, xd))
    return xd


def Barry_Dynamic(barry, Ts, t):
    # 1
    barry[0] = barry[0] - Ts
    barry[1] = barry[1] + Ts
    # 2
    barry[2] = barry[2] - Ts / 10
    barry[3] = barry[3] + Ts / 10
    # 3
    barry[4] = barry[4] + Ts
    barry[5] = barry[5] - Ts
    # 4
    barry[6] = barry[6] + 2 * (np.sin(np.pi * Ts/10 * t) - np.sin(np.pi * Ts/10 * (t - 1)))
    barry[7] = barry[7]
    # 5
    barry[8] = barry[8]
    barry[9] = barry[9] + 2 * (np.sin(np.pi * Ts / 10 * t) - np.sin(np.pi * Ts / 10 * (t - 1)))
    #6
    barry[10] = barry[10]
    barry[11] = barry[11] + 2 * (np.sin(np.pi * Ts / 5 * t) - np.sin(np.pi * Ts / 5 * (t - 1)))
    #7
    barry[12] = barry[12] - Ts
    barry[13] = barry[13] + Ts
    #8
    barry[14] = barry[14] + 2 * (np.sin(np.pi * Ts/10 * t) - np.sin(np.pi * Ts/10 * (t - 1)))
    barry[15] = barry[15]
    #9
    barry[16] = barry[16] + Ts/10
    barry[17] = barry[17] - Ts/10
    #10
    barry[18] = barry[18] - Ts
    barry[19] = barry[19] - Ts
    res = np.array(barry).reshape(-1, 1)

    return res


def Barry_State(robot, barry, r):
    '''
    如果 temp0 的值为正，那么点在圆外；如果 temp0 的值为零，那么点在圆上；如果 temp0 的值为负，那么点在圆内。这是一个常见的用于判断点和圆的位置关系的计算方法。
    '''
    b = []
    for i in range(len(r)):
        temp0 = (robot[0] - barry[2 * i]) ** 2 + (robot[1] - barry[2 * i + 1]) ** 2 - r[i] ** 2
        b.append(1 / temp0)
    return b



