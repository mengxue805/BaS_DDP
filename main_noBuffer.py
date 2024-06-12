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


def DubinsVehicle():
    Ts = 0.01  # interval

    r = [0.5, 0.6, 0.4, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.7]  # radius of barry
    Switch = [0.8, 0.9, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # switch the control
    r_in = np.array(r) - 0.1

    n = 3 + len(r)  # number of states
    m = 2  # number of control

    barry = [9., 7., 3., 3., 5., 5., 4.8, 6, 3, 5, 0, 1, 2, -2, -2, -2, -1, 4, 6, 2]
    barry = np.array(barry).reshape(-1, 1)

    robot_state = np.array([[9, -4], [9, -4], [5 * np.pi / 4, 5 * np.pi / 4]])  # robot intial and final state

    # In Buffer Entry
    Q = np.array([1, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    Q = jnp.diag(Q)
    Qf = np.array([1, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    Qf = jnp.diag(Qf)


    Qc = 0 * jnp.eye(m)

    b0 = Barry_State(robot_state[:, 0], barry, Switch)
    b_goal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    x0 = np.hstack((robot_state[:, 0], np.array(b0).reshape(-1)))  # inital state
    x0 = jnp.array(x0)
    x_goal = np.hstack((robot_state[:, 1], np.array(b_goal).reshape(-1)))  # final state
    x_goal = jnp.array(x_goal)
    return n, m, x0, x_goal, Ts, Q, Qf, Qc, r, r_in, barry, Switch, robot_state, DubinsVehicle_Dynamics


def main():
    """
    Non-linear trajectory optimization via iLQR/DDP
    """
    # Select system
    dyn_sys = "DubinsVehicle"

    # Load simulation parameters
    if dyn_sys == "DubinsVehicle":
        n, m, x0, x_goal, Ts, Q, Qf, Qc, r, r_in, barry0, Switch, robot_state, DubinsVehicle_Dynamics = DubinsVehicle()
    else:
        print("Dynamical system unknown.")
        exit()

    # choose the algorithm
    Type = 'DDP'
    # Type = 'ILQR'

    model = Model(n, m, x_goal, Ts, DubinsVehicle_Dynamics, Q, Qf, Qc, robot_state)
    ddp_controller = DDP(*model.return_ddp_args())

    x_traj = x0.reshape(n, -1)
    u_traj = 40*np.ones((m, 1))
    J_traj = np.array([])
    J_time = np.array([])
    barry = barry0
    N_roll = 15 if Type == 'DDP' else 5
    t = 0


    start_time = time.time()
    while np.linalg.norm(x_traj[:2, t] - robot_state[:2, -1]) > 0.1 and t < 1000:
        x_start = x_traj[:, t]

        u_start = np.tile(u_traj[:, -1].reshape(-1, 1), (1, N_roll - 1))

        # #check the whether in the buffer entry
        # flag = 0
        # N_roll = 10
        # for i in range(3, n):
        #     in_temp = (x_start[0] - barry[(i - 3) * 2, t]) ** 2 + (x_start[1] - barry[(i - 3) * 2 + 1, t]) ** 2 - \
        #               Switch[i - 3] ** 2
        #     if in_temp < 0:
        #         N_roll = 50
        #         if x_start[0] == barry[(i - 3) * 2, t]:
        #             angle0 = np.pi
        #         else:
        #             slope_y = barry[(i - 3) * 2 + 1, t] - x_start[1]
        #             slope_x = barry[(i - 3) * 2, t] - x_start[0]
        #             angle0 = np.arctan(slope_y / slope_x)
        #
        #         if angle0 < 0 and slope_x < 0:
        #             angle0 += np.pi
        #         elif angle0 > 0 and slope_x < 0:
        #             angle0 += np.pi
        #         if angle0 < 0:
        #             angle0 += 2*np.pi
        #         angle1 = x_start[2]
        #         if angle1 < 0:
        #             angle1 += 2 * np.pi
        #         if angle1 > angle0:
        #             u_start = np.tile(np.array([40., 20.]).reshape(-1, 1), (1, N_roll - 1))
        #         else:
        #             u_start = np.tile(np.array([20., 40.]).reshape(-1, 1), (1, N_roll - 1))
        #
        #         temp = np.sqrt(-in_temp) / (Switch[i - 3] - r[i - 3])
        #         if temp > flag:
        #             flag = temp

        temp_x_traj, temp_u_out, J, itr = ddp_controller.run_DDP(x_start, u_start, N_roll, barry[:, t], r, Type)

        x_traj = np.hstack((x_traj, temp_x_traj[:, 1].reshape(n, -1)))
        u_traj = np.hstack((u_traj, temp_u_out[:, 0].reshape(m, -1)))
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

    # plt.figure()
    # plt.plot(np.array(range(u_traj.shape[1])), u_traj[0, :], 'r')
    # plt.plot(np.array(range(u_traj.shape[1])), u_traj[1, :], 'b')
    # plt.legend(['v_right', 'v_left'])
    # plt.show()

    # plt.figure()
    # for i in range(10):
    #     plt.plot(np.array(range(x_traj.shape[1])) * Ts, x_traj[3 + i, :], label=str(r[i]))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(labels)
    # plt.xlabel('t')
    # plt.ylabel('Barrier State z = 1/h')
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(range(t))*Ts, J_traj)
    # plt.xlabel('t')
    # plt.ylabel('Cost')
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(range(t))*Ts, J_time)
    # plt.xlabel('t')
    # plt.ylabel('iteration')
    # plt.show()

    # print(f'run time = {t*Ts}')

    plot_trajMul(x_traj, u_traj, barry, r, r_in, Switch, t, "NoBuffer")




if __name__ == "__main__":
    main()
