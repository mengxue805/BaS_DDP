import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd, device_get
from functools import partial
import jax.numpy as jnp
from dynamical_sys import Barry_State
import time

class Model:
    def __init__(self, n, m, x_goal, Ts, cont_dynamics, Q, Qf, Qc, robot_state):
        self.n = n  # Number of states
        self.m = m  # Number of controls
        self.x_goal = x_goal  # Goal state
        self.Ts = Ts  # Discretization step
        self.cont_dynamics = cont_dynamics  # Continuous dynamics
        self.Q = Q  # State cost weight
        self.Qf = Qf  # Final state cost weight
        self.R = Qc  # Controls cost weight
        self.robot_state = robot_state
        self.f = self.rk4  # Discrete-time dynamics

    def Switch(self, Q, Qf):
        self.Q = Q
        self.Qf = Qf

    @partial(jit, static_argnums=(0,))
    def rk4(self, x, u, barry, r):
        """Discrete-time dynamics: Integration with RK4 method."""
        f = self.cont_dynamics
        last_robot_state = self.robot_state[:, -1]
        k1 = self.Ts * f(x, u, barry, r, last_robot_state)
        k2 = self.Ts * f(x + k1 / 2, u, barry, r, last_robot_state)
        k3 = self.Ts * f(x + k2 / 2, u, barry, r, last_robot_state)
        k4 = self.Ts * f(x + k3, u, barry, r, last_robot_state)
        x_next = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        b = Barry_State(x[:3], barry, r)

        x_next = jnp.hstack((x_next[:3], b))

        return x_next

    @partial(jit, static_argnums=(0,))
    def f_x(self, x, u, barry, r):
        """Partial derivative of dynamics wrt x, dfdx."""
        return jacfwd(self.f, 0)(x, u, barry, r)

    @partial(jit, static_argnums=(0,))
    def f_u(self, x, u, barry, r):
        """Partial derivative of dynamics wrt u, dfdu."""
        return jacfwd(self.f, 1)(x, u, barry, r)

    @partial(jit, static_argnums=(0,))
    def l_x(self, x):
        """Partial derivative of stage cost wrt x, dldx."""
        return self.Q @ (x - self.x_goal)

    @partial(jit, static_argnums=(0,))
    def l_u(self, u):
        """Partial derivative of stage cost wrt u, dldu."""
        return self.R @ u

    def l_xx(self):
        """Second partial derivative of stage cost wrt x."""
        return self.Q

    def l_uu(self):
        """Second partial derivative of stage cost wrt u."""
        return self.R

    def l_xu(self):
        """Second mixed partial derivative of stage cost."""
        return jnp.zeros((self.n, self.m))

    def l_ux(self):
        """Second mixed partial derivative of stage cost."""
        return jnp.zeros((self.m, self.n))

    @partial(jit, static_argnums=(0,))
    def lf_x(self, x):
        """Partial derivative of terminal cost wrt x, dlfdx."""
        return self.Qf @ (x - self.x_goal)

    def lf_xx(self):
        """Second partial derivative of terminal cost wrt x."""
        return self.Qf

    @partial(jit, static_argnums=(0,))  # dAdx
    def f_xx(self, x, u, barry, r):
        """Second partial derivative of dynamics wrt x."""
        return jacfwd(lambda xx, uu, barry, r: self.f_x(xx, uu, barry, r), 0)(x, u, barry, r)

    @partial(jit, static_argnums=(0,))
    def f_xu(self, x, u, barry, r):  # dAdu
        """Second mixed partial derivative of dynamics."""
        return jacfwd(lambda xx, uu, barry, r: self.f_x(xx, uu, barry, r), 1)(x, u, barry, r)

    @partial(jit, static_argnums=(0,))
    def f_ux(self, x, u, barry, r):
        """Second mixed partial derivative of dynamics."""
        Bx = jacfwd(self.f_u, 0)(x, u, barry, r)  # dBdx
        return jnp.squeeze(Bx)

    @partial(jit, static_argnums=(0,))
    def f_uu(self, x, u, barry, r):
        """Second partial derivative of dynamics wrt u."""
        Bu = jacfwd(self.f_u, 1)(x, u, barry, r)  # dBdu
        return jnp.squeeze(Bu)

    @partial(jit, static_argnums=(0,))
    def stage_cost(self, x, u):
        return 0.5 * (x - self.x_goal).T @ self.Q @ (x - self.x_goal) + 0.5*u.T @ self.R @ u

    @partial(jit, static_argnums=(0,))
    def terminal_cost(self, x):
        return 0.5 * (x - self.x_goal).T @ self.Q @ (x - self.x_goal)

    @partial(jit, static_argnums=(0,))
    def cost(self, x_hist, u_hist):
        N = x_hist.shape[1]
        J = 0.0
        for k in range(N - 1):
            J += self.stage_cost(x_hist[:, k], u_hist[:, k])
        J += self.terminal_cost(x_hist[:, N - 1])
        return J

    @staticmethod
    def vec(A):
        """Vectorization operator: Converts lxm matrix to (lm,) vector."""
        return A.ravel(order="F")

    def return_ddp_args(self):
        """Returns the required arguments for the DDP algorithm."""
        args = (self.n,
                self.m,
                self.Ts,
                self.f,
                self.f_x,
                self.f_u,
                self.f_xx,
                self.f_uu,
                self.f_xu,
                self.f_ux,
                self.l_x,
                self.l_u,
                self.l_xx,
                self.l_uu,
                self.lf_x,
                self.lf_xx,
                self.cost)
        return args
