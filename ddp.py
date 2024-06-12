import numpy as np
import jax.numpy as jnp
from functools import wraps
import time


class DDP:
    def __init__(self, n, m, Ts, f, f_x, f_u, f_xx, f_uu, f_xu, f_ux, l_x, l_u, l_xx, l_uu, lf_x, lf_xx, cost):
        self.n = n  # Number of states
        self.m = m  # Number of controls
        self.Ts = Ts
        self.f = f  # Discrete dynamics
        self.f_x = f_x
        self.f_u = f_u
        self.f_xx = f_xx
        self.f_uu = f_uu
        self.f_xu = f_xu
        self.f_ux = f_ux
        self.l_x = l_x
        self.l_u = l_u
        self.l_xx = l_xx
        self.l_uu = l_uu
        self.lf_x = lf_x
        self.lf_xx = lf_xx
        self.cost = cost  # Objective function cost

        self.N = None  # Number of timesteps
        self.eps = 1e-3  # Tolerance
        self.max_iter = 50  # Maximum iterations for DDP
        self.max_iter_reg = 10  # Maximum iterations for regularization
        self.backward_total_time = 0
        self.temp_time = 0
        self.count = 0

    def run_DDP(self, x0, u_traj, N, barry, r, Type):
        """
        Runs the iLQR or DDP algorithm.
        Inputs:
          - x0(np.ndarray):     The current state            [nx1]
          - u_traj(np.ndarray): The current control solution [mxN]
          - N(int):             Try path for N times        [int]
        Returns:
          - x_traj(np.ndarray): The states trajectory        [nxN]
          - u_traj(np.ndarray): The controls trajectory      [mxN]
          - J(float):           The optimal cost
        """
        print("DDP beginning..")
        self.N = N

        # Initial Rollout
        x_traj = np.zeros((self.n, self.N))
        x_traj[:, 0] = x0

        for k in range(self.N - 1):
            x_traj[:, k + 1] = self.f(x_traj[:, k], u_traj[:, k], barry, r)
        J = self.cost(x_traj, u_traj)

        # Initialize DDP matrices
        p = np.ones((self.n, self.N))  # p 通常用于存储每个时间步的代价函数关于状态的梯度。
        P = np.zeros((self.n, self.n, self.N))  # P 通常用于存储每个时间步的代价函数关于状态的Hessian矩阵（也就是二阶梯度）。
        d = np.ones((self.N - 1, self.m))  # d 通常用于存储每个时间步的控制更新。
        K = np.zeros((self.m, self.n, self.N - 1))  # K 通常用于存储每个时间步的反馈控制增益。

        itr = 0
        prev_err = np.inf
        err_diff = np.inf
        #while itr < self.max_iter:
        while np.linalg.norm(d, np.inf) > self.eps and itr < self.max_iter and err_diff > 1e-3:
            # Backward Pass
            backward_start_time = time.time()
            DJ, p, P, d, K = self.backward_pass(p, P, d, K, x_traj, u_traj, barry, r, Type)
            self.backward_total_time += time.time() - backward_start_time
            # print("Backward pass took: {} seconds".format(time.time() - backward_start_time))
            # Forward rollout with line search
            x_new, u_new, Jn = self.rollout_with_linesearch(x_traj, u_traj, d, K, J, DJ, barry, r)

            # Update values
            J = Jn
            x_traj = x_new
            u_traj = u_new
            itr += 1
            err_diff = abs(np.linalg.norm(d, np.inf) - prev_err)
            prev_err = np.linalg.norm(d, np.inf)

            if itr % 10 == 0:
                print("Iteration: {}, J = {}".format(itr, J))

        print("\nDDP took: {} iterations".format(itr))
        return x_traj, u_traj, J, itr

    def backward_pass(self, p, P, d, K, x_traj, u_traj, barry, r, Type):
        """Performs a backward pass to update values."""
        DJ = 0.0
        p[:, self.N - 1] = self.lf_x(x_traj[:, self.N - 1].reshape(-1, 1))[:, 0]
        P[:, :, self.N - 1] = self.lf_xx()
        # V_x = self.l_x(x_traj[:, 9])[-1]
        self.count += 1

        A_temp = np.zeros((self.N - 1, self.n, self.n))
        
        # for k in range(self.N - 2, -1, -1):
        #     A = self.f_x(x_traj[:, k], u_traj[:, k], barry, r)
        #     A_temp[k] = A
        

        for k in range(self.N - 2, -1, -1):
            # Compute derivatives
            A = self.f_x(x_traj[:, k], u_traj[:, k], barry, r)
            B = self.f_u(x_traj[:, k], u_traj[:, k], barry, r)  

            gx = self.l_x(x_traj[:, k]) + A.T @ p[:, k + 1]  # (n,)
            gu = self.l_u(u_traj[:, k]) + B.T @ p[:, k + 1]  # (m,)
            

            # # iLQR (Gauss-Newton) version
            # # ------------------------------------------------------------------
            if Type == 'ILQR':
                Gxx = self.l_xx() + np.dot(np.dot(A.T, P[:, :, k + 1]), A)  # nxn
                Guu = self.l_uu() + np.dot(np.dot(B.T, P[:, :, k + 1]), B)  # mxm
                Gxu = np.dot(A.T, np.dot(P[:, :, k + 1], B))  # nxm
                Gux = np.dot(B.T, np.dot(P[:, :, k + 1], A))  # mxn
                

            # DDP (full Newton) version
            # ------------------------------------------------------------------
            if Type == 'DDP':
                Ax = self.f_xx(x_traj[:, k], u_traj[:, k], barry, r)  # nnxn
                Bx = self.f_ux(x_traj[:, k], u_traj[:, k], barry, r)  # nxn
                Au = self.f_xu(x_traj[:, k], u_traj[:, k], barry, r)  # nnxm
                Bu = self.f_uu(x_traj[:, k], u_traj[:, k], barry, r)  # (n,)

                Gxx = self.l_xx() + A.T @ P[:, :, k + 1] @ A + jnp.tensordot(p[:, k + 1], Ax, axes=1)  # nxn
                Guu = self.l_uu() + B.T @ P[:, :, k + 1] @ B + jnp.tensordot(p[:, k + 1], Bu, axes=1)  # mxm
                Gxu = A.T @ P[:, :, k + 1] @ B + jnp.tensordot(p[:, k + 1], Au, axes=1)  # nxm
                Gux = B.T @ P[:, :, k + 1] @ A + jnp.tensordot(p[:, k + 1], Bx, axes=1)  # mxn


                # Regularization
                beta = 0.1
                G = np.block([[Gxx, Gxu],
                              [Gux, Guu]])
                iter_reg = 0
                temp = time.time()
                self.is_pos_def(G)
                self.temp_time += time.time() - temp
                while not self.is_pos_def(G) and iter_reg < self.max_iter_reg:
                    Gxx += beta * A.T @ A
                    Guu += beta * B.T @ B
                    Gxu += beta * A.T @ B
                    Gux += beta * B.T @ A
                    beta = 2 * beta
                    # print("regularizing G")
                    iter_reg += 1
                # print("Regularization time: ", time.time() - temp)
                
            # ------------------------------------------------------------------
            d[k, :], _, _, _ = jnp.linalg.lstsq(Guu, gu, rcond=None)
            K[:, :, k], _, _, _ = jnp.linalg.lstsq(Guu, Gux, rcond=None)
            # p[:, k] = gx - K[:, :, k].T @ gu + (K[:, :, k].T @ Guu @ d[k, :].T).reshape(self.n, ) - (
            #         Gxu @ d[k, :].T).reshape(self.n, )
            # P[:, :, k] = Gxx + K[:, :, k].T @ Guu @ K[:, :, k] - Gxu @ K[:, :, k] - K[:, :, k].T @ Gux

            p[:, k] = gx + (K[:, :, k].T @ Guu @ d[k, :].T).reshape(self.n, ) - K[:, :, k].T @ gu - (
                    Gux.T @ d[k, :]).reshape(self.n, )

            P[:, :, k] = Gxx + K[:, :, k].T @ Guu @ K[:, :, k] - K[:, :, k].T @ Gux - Gux.T @ K[:, :, k]
            # P[:, :, k] = 0.5*(P[:, :, k] + P[:, :, k].T)

            DJ += gu.T @ d[k, :].T

        return DJ, p, P, d, K  # d = k


    @staticmethod
    def is_pos_def(A):
        """Check if matrix A is positive definite.

        If symmetric and has Cholesky decomposition -> p.d.
        """
        if np.allclose(A, A.T, rtol=1e-04, atol=1e-04):  # Ensure it is symmetric
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False


    def rollout(self, x_traj, u_traj, d, K, a, barry, r):
        """Forward rollout."""
        x_new = np.zeros((self.n, self.N))
        u_new = np.zeros((self.m, self.N - 1))
        x_new[:, 0] = x_traj[:, 0]

        for k in range(self.N - 1):
            diff = x_new[:, k] - x_traj[:, k]
            u_new[:, k] = u_traj[:, k] - a * d[k] - K[:, :, k] @ diff
            u_new[:, k] = np.clip(u_new[:, k], -20, 40)
            x_new[:, k + 1] = self.f(x_new[:, k], u_new[:, k], barry, r)

        J_new = self.cost(x_new, u_new)
        return x_new, u_new, J_new

    # @timeit
    def rollout_with_linesearch(self, x_traj, u_traj, d, K, J, DJ, barry, r):
        """Forward rollout with linesearch to find best step size."""
        a = 1.0  # Step size
        b = 1e-8  # Armijo tolerance
        armijo_scale = b * DJ
        x_new, u_new, Jn = self.rollout(x_traj, u_traj, d, K, a, barry, r)

        while Jn > (J -  a * armijo_scale):
            a *= 0.1
            x_new, u_new, Jn = self.rollout(x_traj, u_traj, d, K, a, barry, r)

        return x_new, u_new, Jn


    @staticmethod
    def comm_mat(m, n):
        """Commutation matrix.

        Used to transform the vectorized form of a matrix into the vectorized
        form of its transpose.
        Inputs:
          - m(int): Number of rows
          - n(int): Number of columns
        """
        w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")
        return np.eye(m * n)[w, :]
