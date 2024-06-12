# -*- coding: utf-8 -*-
import random

import numpy as np
import math
import sympy as sym
from ddp import DDPOptimizer

class Barriers(object):
    def __init__(self):
        self.a_list = [10]
        self.b_list = [10]
        self.c_list = [10]
        self.r_list = [1]
        self.distance_min = None


    def add_barriers(self, a, b, c, r):
        self.a_list.append(a)
        self.b_list.append(b)
        self.c_list.append(c)
        self.r_list.append(r)


    def get_barriers(self, x, y, z):
        distance_list = []
        for i, a1_ in enumerate(self.a_list):

            d1 = euclidean_distance(x1=x, y1=y, z1=z, x2=a1_, y2=self.b_list[i], z2=self.c_list[i]) - self.r_list[i]
            distance_list.append(d1)

        d_min_index = distance_list.index(min(distance_list))
        distance_min = distance_list[d_min_index]
        self.distance_min = distance_min

        return self.a_list[d_min_index], self.b_list[d_min_index], self.c_list[d_min_index], self.r_list[d_min_index], self.distance_min



def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


# 安全区域
def h_count(a, b, c, r, x, y, z):
    result = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r ** 2
    return result

def h0_count(a, b, c, r):
    result = (0 - a) ** 2 + (0 - b) ** 2 + (0 - c) ** 2 - r ** 2
    return result


def hx_count(a, b, c, x, y, z):
    r1 = 2 * (x - a)
    r2 = 2 * (y - b)
    r3 = 2 * (z - c)
    result = np.array([r1, r2, r3, 0])
    return result

# 辅助
def beta_count(a, b, c, r, x, y, z):
    result = 1 / h_count(a, b, c, r, x, y, z)
    return result


def beta0_count(a, b, c, r):
    result = 1 / h_count(a, b, c, r, 0, 0, 0)
    return result

def dynamic_circle(a, b, t, f=1/60, bias=0, length=2):
    # 动态障碍物, 水平运动
    a = a + length * math.sin(2 * math.pi * f * t + bias)
    b = b

    return a, b


def dynamic_circle_2(a, b, t, f=1/60, bias=0, length=2):
    # 动态障碍物, 竖直运动
    a = a
    b = b + length * math.sin(2 * math.pi * f * t + bias)

    return a, b


class System(object):
    def __init__(self):
        self.t = 0
        self.t_max = 400
        self.x_list = []
        self.y_list = []


        self.A = np.zeros((4, 4))
        self.B = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])


        self.x = 1
        self.y = 1
        self.z = 1
        self.betax = 0

        self.x_ = 0
        self.y_ = 0
        self.z_ = 0



        # 主要障碍物
        self.barriers_a = 0
        self.barriers_b = 0
        self.barriers_c = 0
        self.barriers_r = 0



    def f(self, x, u, constrain=True):
        # dt = 0.001
        dt = 0.0008

        X_ = np.dot(self.A, x) + np.dot(self.B, u)
        x_new = x[0] + X_[0] * dt
        y_new = x[1] + X_[1] * dt
        z_new = x[2] + X_[2] * dt
        betax_new = beta_count(a=self.barriers_a, b=self.barriers_b, c=self.barriers_c, r=self.barriers_r,
                           x=x_new, y=y_new, z=z_new)

        return sym.Matrix(
            [
                x_new,
                y_new,
                z_new,
                betax_new,
            ]
        )

    def g(self, x, u, x_goal):
        error = x - x_goal
        # Q = 1 * np.eye(len(x))
        Q = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        Q = np.asmatrix(Q)
        R = 1 * np.eye(len(u))
        return error.T @ Q @ error + u.T @ R @ u

    def h(self, x, x_goal):
        error = x - x_goal
        Q = 1 * np.eye(len(x))
        return error.T @ Q @ error

    def length_(self, a1, a2, a3, b1, b2, b3):
        return math.sqrt((a1 - b1) ** 2 + (a2 - b2) ** 2 + (a3 - b3) ** 2)



    def run(self, x1_o=17., x2_o=15., x3_o=1., goal_x=0., goal_y=0., goal_z=0.,
            a_=10, b_=10, c_=10, r_=1, distance_min=10, return_list=False):
        self.x = x1_o
        self.y = x2_o
        self.z = x3_o

        # 缓冲区范围大小，a 与 b 之间的距离
        buffer_ = 0.09

        # 可入侵范围大小
        buffer_safe = 0.5

        # ddp 参数
        N = 20  # 轨迹点
        Nx = 4  # 状态维度
        Nu = 3  # 控制维度

        x1_list = []
        x2_list = []
        x3_list = []
        x1_list.append(x1_o)
        x2_list.append(x2_o)
        x3_list.append(x3_o)



        self.barriers_a = a_
        self.barriers_b = b_
        self.barriers_c = c_
        self.barriers_r = r_


        # for t in range(self.t_max):
        # for t in range(1):
        while self.length_(a1=x1_list[-1], a2=x2_list[-1], a3=x3_list[-1],
                      b1=goal_x, b2=goal_y, b3=goal_z) > 0.05:

            # 计算X
            if len(x1_list) == 0:
                X0 = np.array([x1_o, x2_o, x3_o])
                x1_ = X0[0]
                x2_ = X0[1]
                x3_ = X0[2]

                x1_list.append(x1_)
                x2_list.append(x2_)
                x3_list.append(x3_)
                self.x = x1_o
                self.y = x2_o
                self.z = x3_o
            else:
                x1_ = x1_list[-1]
                x2_ = x2_list[-1]
                x3_ = x3_list[-1]

            # beta
            beta_0 = beta0_count(a=self.barriers_a, b=self.barriers_b, c=self.barriers_c, r=self.barriers_r)

            beta_x = beta_count(a=self.barriers_a, b=self.barriers_b, c=self.barriers_c, r=self.barriers_r,
                                x=x1_, y=x2_, z=x3_)

            # DDP
            X0 = np.array([x1_, x2_, x3_, beta_x])
            # x_goal = np.array([0.0, 0.0, 0.0, beta_0])
            x_goal = np.array([goal_x, goal_y, goal_z, beta_0])


            distance_min = euclidean_distance(x1=X0[0], y1=X0[1], z1=X0[2],
                                              x2=self.barriers_a, y2=self.barriers_b, z2=self.barriers_c) - self.barriers_r

            if distance_min > buffer_:
                print(f'distance_min = {distance_min}, 缓存区外')
                # 缓冲区外
                # 前进速度
                v = 0.05
                # 计算方向
                vector = [x_goal[i] - X0[i] for i in range(3)]
                # 模
                length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                # 标准化
                if length > v:
                    unit_vector = [vector[i] / length for i in range(3)]
                else:
                    unit_vector = vector


                x1_list.append(x1_list[-1] + unit_vector[0] * v)
                x2_list.append(x2_list[-1] + unit_vector[1] * v)
                x3_list.append(x3_list[-1] + unit_vector[2] * v)

                # X0 = np.array([x1_list[-1], x2_list[-1], x3_list[-1], beta_x])

            else:


                print('缓冲区内！！！')
                # ddp
                # ddp = DDPOptimizer(Nx, Nu, self.f, self.g, self.h)
                # Xout, U, X_hist, U_hist, J_hist = ddp.optimize(X0, x_goal, N=N, full_output=True)
                #
                # x1_list_ddp = Xout[:, 0]
                # x2_list_ddp = Xout[:, 1]
                # x3_list_ddp = Xout[:, 2]

                v = 0.07
                # 计算方向
                ba_p= [self.barriers_a, self.barriers_b, self.barriers_c]
                vector = [X0[i] - ba_p[i] for i in range(3)]
                # 模
                length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                # 标准化
                if length > v:
                    unit_vector = [vector[i] / length for i in range(3)]
                else:
                    unit_vector = vector

                add_z = 0
                if unit_vector[2] > 0 and random.random() < 0.7:
                    add_z = 0.02

                add_x = 0
                if unit_vector[0] < 0 and random.random() < 0.7:
                    add_x = -0.02


                # for i in range(1, 2):
                # for i in range(len(x1_list_ddp)):
                x1_list.append(x1_list[-1] + unit_vector[0] * v + add_x)
                x2_list.append(x2_list[-1] + unit_vector[1] * v)
                x3_list.append(x3_list[-1] + unit_vector[2] * v + add_z)


        self.x = x1_list[-1]
        self.y = x2_list[-1]
        self.z = x3_list[-1]

        if return_list:
            return x1_list[1:], x2_list[1:], x3_list[1:]
        else:
            return self.x, self.y, self.z


if __name__ == '__main__':
    # 障碍物初始状态
    a1 = 6
    b1 = 5
    c1 = 0
    r1 = 2

    a2 = 5
    b2 = 10
    c2 = 0
    r2 = 2

    a3 = 13
    b3 = 13
    c3 = 0
    r3 = 2

    # 末端端点当前位置
    x_current = 0.5
    y_current = 0.5
    z_current = 0.5

    # 末端端点目标位置
    goal_x = 0.03
    goal_y = 0.0
    goal_z = 1.15

    barriers = Barriers()
    barriers.add_barriers(a=a1, b=b1, c=c1, r=r1)

    a_, b_, c_, r_, distance_min = barriers.get_barriers(x=x_current, y=y_current, z=z_current)

    system = System()
    x, y, z = system.run(x1_o=x_current, x2_o=y_current, x3_o=z_current,
                        goal_x=goal_x, goal_y=goal_y, goal_z=goal_z,
                         a_=a_, b_=b_, c_=c_, r_=r_, distance_min=distance_min)







