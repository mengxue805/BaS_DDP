import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ma


def plot_traj(x, barry, r):
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r')
    line2, = ax.plot([], [], 'b')

    theta = np.arange(0, 2 * np.pi, 0.01)

    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        return line1, line2

    def update(i):
        a = barry[0, i]
        b = barry[1, i]
        xc = a + r * np.cos(theta)
        yc = b + r * np.sin(theta)
        line1.set_data(xc, yc)
        line2.set_data(x[0, :i + 1], x[1, :i + 1])
        return line1, line2

    ani = ma.FuncAnimation(fig, update, frames=500, init_func=init, interval=100, repeat=True)
    ani.save('morlet.gif')
    plt.show()


def plot_trajMul(x, u, barry, r, r_in, Switch, fream, type):
    fig, ax = plt.subplots()

    lines = []
    for i in range(3 * len(r)):
        if (i + 1) % 3 == 0:
            line, = ax.plot([], [], 'red')
        elif (i + 1) % 3 == 2:
            line, = ax.plot([], [], 'red', linestyle='--')
        else:
            line, = ax.plot([], [], 'black')
        lines.append(line)
    line, = ax.plot([], [], 'blue')
    lines.append(line)

    theta = np.arange(0, 2 * np.pi, 0.01)

    def init():
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        return lines

    def update(i):
        for j in range(len(r)):
            a = barry[2 * j, i]
            b = barry[2 * j + 1, i]

            xc = a + r[j] * np.cos(theta)
            yc = b + r[j] * np.sin(theta)

            lines[3 * j].set_data(xc, yc)

            if type == 'Buffer':
                x_in = a + r_in[j] * np.cos(theta)
                y_in = b + r_in[j] * np.sin(theta)

                xw = a + Switch[j] * np.cos(theta)
                yw = b + Switch[j] * np.sin(theta)

                lines[3 * j + 1].set_data(x_in, y_in)
                lines[3 * j + 2].set_data(xw, yw)
        lines[-1].set_data(x[0, :i + 1], x[1, :i + 1])
        return lines

    ani = ma.FuncAnimation(fig, update, frames=fream, init_func=init, repeat=True, interval=0.1)
    ani.save('images/Exp1.gif')

