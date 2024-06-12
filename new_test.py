import jax
import jax.numpy as jnp


# 定义一个函数，接受两个输入 x 和 u
def my_function(x, u):
    return jnp.array([x[0] ** 2 + u[1], x[2] * u[0], x[1] - u[1]])


# 使用 jax.jacobian 计算雅可比矩阵关于 x 和 u 的偏导数
x_value = [2.0, 1., 3.]
u_value = [3.0, 0.5]
jacobian_matrix = jax.jacobian(my_function, argnums=(0, 1))(x_value, u_value)

mixed_hessian_matrix = jax.hessian(lambda x, u: my_function(x, u)[0], argnums=(0, 1))(x_value, u_value)

# 假设 a 和 b 都是 n*m*m 形状的数组
a = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
b = jnp.array([[[5, 6], [7, 8]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]])

# 使用 tensordot 计算结果
result = jnp.sum(a*b, 0)

print("混合二阶导数关于 x 和 u 的海森矩阵:")
print(mixed_hessian_matrix)
