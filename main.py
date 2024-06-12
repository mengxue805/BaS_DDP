import matplotlib.pyplot as plt
import numpy as np
from jax import config
import jax.numpy as jnp
from model import Model
from ddp import DDP
from dynamical_sys import DubinsVehicle_Dynamics, Barry_Dynamic, Barry_State
from plotter import plot_trajMul
import time

config.update("jax_enable_x64", True)  # enable float64 types for accuracy


def DubinsVehicle()->tuple:
    """
    Dubins Vehicle model
    ; n: number of states
    ; m: number of control
    ; x0: initial state
    ; x_goal: final state
    ; Ts: interval
    ; Q_in: cost matrix for in buffer entry
    ; Qf_in: final cost matrix for in buffer entry
    ; Q_out: cost matrix for out buffer entry
    ; Qf_out: final cost matrix for out buffer entry
    ; Qc: control cost matrix
    ; r: radius of barry
    ; r_in: radius of in buffer entry
    ; barry: barry state
    ; Switch: switch the control
    ; robot_state: robot initial and final state
    ; DubinsVehicle_Dynamics: dynamics of Dubins Vehicle

    """
    Ts = 0.01  # interval

    r = [0.5, 0.6, 0.4, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.7]  # radius of barry
    Switch = [0.8, 0.9, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # switch the control
    r_in = np.array(r) - 0.1

    n = 3 + len(r)  # number of states
    m = 2  # number of control

    barry = [9., 7., 3., 3., 5., 5., 4.8, 6, 3, 5, 0, 1, 2, -2, -2, -2, -1, 4, 6, 2]
    barry = np.array(barry).reshape(-1, 1)

    robot_state = np.array([[9, -4], [9, -4], [5 * np.pi / 4, 5 * np.pi / 4]])  # robot intial and final state 3x2 matrix first column is x ,y, r

    # In Buffer Entry
    Q_in = np.array([1, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    Q_in = jnp.diag(Q_in)
    Qf_in = np.array([1, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    Qf_in = jnp.diag(Qf_in)

    # Out Buffer Entry
    Q_out = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Q_out = jnp.diag(Q_out)
    Qf_out = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Qf_out = jnp.diag(Qf_out)

    Qc = 0 * jnp.eye(m)

    b0 = Barry_State(robot_state[:, 0], barry, Switch)
    b_goal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    x0 = np.hstack((robot_state[:, 0], np.array(b0).reshape(-1)))  # inital state first column is x ,y, r, second column is barry state
    x0 = jnp.array(x0)
    x_goal = np.hstack((robot_state[:, 1], np.array(b_goal).reshape(-1)))  # final state
    x_goal = jnp.array(x_goal)
    return n, m, x0, x_goal, Ts, Q_in, Qf_in, Q_out, Qf_out, Qc, r, r_in, barry, Switch, robot_state, DubinsVehicle_Dynamics


def main():
    """
    Non-linear trajectory optimization via iLQR/DDP
    """
    # Select system
    dyn_sys = "DubinsVehicle"

    # Load simulation parameters
    if dyn_sys == "DubinsVehicle":
        n, m, x0, x_goal, Ts, Q_in, Qf_in, Q_out, Qf_out, Qc, r, r_in, barry0, Switch, robot_state, DubinsVehicle_Dynamics = DubinsVehicle()
    else:
        print("Dynamical system unknown.")
        exit()

    # choose the algorithm
    Type = 'DDP'
    #Type = 'ILQR'

    model1 = Model(n, m, x_goal, Ts, DubinsVehicle_Dynamics, Q_out, Qf_out, Qc, robot_state)
    model2 = Model(n, m, x_goal, Ts, DubinsVehicle_Dynamics, Q_in, Qf_in, Qc, robot_state)
    ddp_controller1 = DDP(*model1.return_ddp_args())
    ddp_controller2 = DDP(*model2.return_ddp_args())

    x_traj = x0.reshape(n, -1)  # reshape to 13 x 1 matrix
    u_traj = 40 * np.ones((m, 1))  # 2 x 1 matrix
    J_traj = np.array([])
    J_time = np.array([])
    barry = barry0
    N_roll = 10 if Type == 'DDP' else 5
    t = 0
    flag = 0

    start_time = time.time()
    ddp_run_time = 0
    operate_time = []
    while np.linalg.norm(x_traj[:2, t] - robot_state[:2, -1]) > 0.1 and t < 1000: 
        # 机器人当前状态与预期轨迹之间的距离大于 0.1，并且 t 小于 1000，就继续执行循环。
        x_start = x_traj[:, t]

        # 如果 flag 等于0，u_start 的值将基于 u_traj 的最后一列；否则，u_start 的值将基于一个包含两个元素40.0的列向量。在两种情况下，u_start 的值都是一个由列向量复制得到的二维数组。
        if flag == 0:
            u_start = np.tile(u_traj[:, -1].reshape(-1, 1), (1, N_roll - 1))
        else:
            u_start = np.tile(np.array([40.0, 40.0]).reshape(-1, 1), (1, N_roll - 1))

        # check the whether in the buffer entry
        flag = 0
        N_roll = 10 if Type == 'DDP' else 5

        # recalucate the Switch by the distance between the robot and the barry
        # for i in range(0, len(Switch)):
        #     if np.linalg.norm(x_start[:2] - barry[2 * i:2 * i + 2]) < Switch[i]:
        #         pass

        for i in range(3, n):
            in_temp = (x_start[0] - barry[(i - 3) * 2, t]) ** 2 + (x_start[1] - barry[(i - 3) * 2 + 1, t]) ** 2 - \
                      Switch[i - 3] ** 2
            if in_temp < 0:
                N_roll = 15 if Type == 'DDP' else 15
                if x_start[0] == barry[(i - 3) * 2, t]:
                    angle0 = np.pi
                else:
                    slope_y = barry[(i - 3) * 2 + 1, t] - x_start[1]
                    slope_x = barry[(i - 3) * 2, t] - x_start[0]
                    angle0 = np.arctan(slope_y / slope_x)

                if angle0 < 0 and slope_x < 0:
                    angle0 += np.pi
                elif angle0 > 0 and slope_x < 0:
                    angle0 += np.pi
                if angle0 < 0:
                    angle0 += 2 * np.pi
                angle1 = x_start[2]
                if angle1 < 0:
                    angle1 += 2 * np.pi
                if angle1 > angle0:
                    u_start = np.tile(np.array([40., 20.]).reshape(-1, 1), (1, N_roll - 1))
                else:
                    u_start = np.tile(np.array([20., 40.]).reshape(-1, 1), (1, N_roll - 1))

                temp = np.sqrt(-in_temp) / (Switch[i - 3] - r[i - 3])
                if temp > flag:
                    flag = temp
        ddp_start_time = time.time()
        # Controller 2
        if flag == 0:
            temp_x_traj, temp_u_out, J, itr = ddp_controller1.run_DDP(x_start, u_start, N_roll, barry[:, t], r, Type)
            u_all = temp_u_out
        else:
            # pass
            temp_x_traj, temp_u_in, J, itr = ddp_controller2.run_DDP(x_start, u_start, N_roll, barry[:, t], r, Type)
            u_all = temp_u_in
        ddp_process_time = time.time() - ddp_start_time
        operate_time.append(ddp_process_time)
        ddp_run_time += ddp_process_time

        x_new = model1.f(x_start, u_all[:, 0].reshape(m, -1), barry[:, t], r)

        x_traj = np.hstack((x_traj, x_new.reshape(n, -1)))  # add new state to x_traj
        u_traj = np.hstack((u_traj, u_all[:, 0].reshape(m, -1)))  # add new control to u_traj
        temp_barry = Barry_Dynamic(barry[:, t], Ts, t)
        barry = np.hstack((barry, temp_barry))
        J_traj = np.append(J_traj, J)
        J_time = np.append(J_time, itr)
        t += 1

    end_time = time.time()
    total_time = end_time - start_time

    print('-' * 40)
    print("* iLQR/DDP Controller *")
    print("Total cost: J = {}".format(J))
    print("Final state error: {}".format(jnp.linalg.norm(x_traj[:2, -1] - x_goal[:2])))
    print('-' * 40)
    print(f'total time = {total_time}')
    print(f'ddp run time = {ddp_run_time}')
    print(f'backward total time = {ddp_controller1.backward_total_time + ddp_controller2.backward_total_time}')
    print(f'temp time = {ddp_controller1.temp_time +ddp_controller2.temp_time}')
    print(f'count = {ddp_controller1.count + ddp_controller2.count}')
    print(x_traj.shape, u_traj.shape)
    print('Operate time average',np.mean(operate_time))
    plot_trajMul(x_traj, u_traj, barry, r, r_in, Switch, t, 'Buffer')

    # plt.figure()
    # plt.plot(np.array(range(u_traj.shape[1]))*Ts, u_traj[0, :], 'r')
    # plt.plot(np.array(range(u_traj.shape[1]))*Ts, u_traj[1, :], 'b')
    # plt.legend(['v_right', 'v_left'])
    # plt.show()

    # plt.figure()
    # for i in range(10):
    #     plt.plot(np.array(range(x_traj.shape[1]))*Ts, x_traj[3 + i, :], label=str(r[i]))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(labels)
    # plt.xlabel('t')
    # plt.ylabel('Barrier State z = 1/h')
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(range(t)) * Ts, J_traj)
    # plt.xlabel('t')
    # plt.ylabel('Cost')
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(range(t)) * Ts, J_time)
    # plt.xlabel('t')
    # plt.ylabel('iteration')
    # plt.show()

    print(f'run time = {t * Ts}')


if __name__ == "__main__":
    main()
