import numpy as np


def f(x):
    n = len(x)
    sum_term = 0
    for i in range(n - 1):
        sum_term += (x[i] - x[i + 1]) ** 2
    return (1 / 2) * (x[0] ** 2 + sum_term - x[n - 1] ** 2) - x[0]


def gradient(x):
    n = len(x)
    grad = np.zeros_like(x)
    grad[0] = x[0] - 1
    grad[n - 1] = -x[n - 1]
    for i in range(1, n - 1):
        grad[i] = 2 * (x[i] - x[i - 1]) - 2 * (x[i + 1] - x[i])
    return grad


"""
gradient descent
"""


def gradient_descent(x, learning_rate, num_iterations=1000):
    for _ in range(num_iterations):
        grad = gradient(x)
        x -= learning_rate * grad
    return x


"""
accelerated gradient descent
"""


def accelerated_gradient_descent(x_init, learning_rate, momentum, num_iterations=1000):
    # 初始化动量项和变量值
    velocity = np.zeros_like(x_init)
    x = x_init.copy()

    for _ in range(num_iterations):
        grad = gradient(x)
        # 更新动量项
        velocity = momentum * velocity - learning_rate * grad
        # 更新变量值
        x = x + velocity
    return x


"""
Conjugate Gradient
"""


def conjugate_gradient(x, max_iterations=1000, tol=1e-9):
    r = gradient(x)
    p = r
    iterations = 0

    while iterations < max_iterations and np.linalg.norm(r) > tol:
        Ap = gradient(p)
        alpha = np.dot(r.T, r) / np.dot(p.T, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = np.dot(r_new.T, r_new) / np.dot(r.T, r)
        p = r_new + beta * p
        r = r_new
        iterations += 1
    return x


"""
Newton method
"""


def hessian(x):
    n = len(x)
    hess = np.zeros((n, n))
    hess[0, 0] = 1
    hess[n - 1, n - 1] = -1
    for i in range(1, n - 1):
        hess[i, i - 1] = -2
        hess[i, i] = 4
        hess[i, i + 1] = -2
    return hess


def newton_method(x0, max_iterations=1000, tol=1e-6):
    x = x0
    iterations = 0

    while iterations < max_iterations:
        grad = gradient(x)
        hess = hessian(x)
        delta_x = np.linalg.solve(hess, -grad)
        x = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            break
        iterations += 1
    return x


"""
BFGS
"""


def bfgs_method(x0, max_iterations=1000, tol=1e-6):
    n = len(x0)
    x = x0
    H = np.eye(n)

    iterations = 0

    while iterations < max_iterations:
        grad = gradient(x)
        p = -np.dot(H, grad)

        alpha = 1
        c = 0.5
        rho = 0.5

        while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad, p):
            alpha = alpha * rho

        x_new = x + alpha * p
        s = x_new - x
        y = gradient(x_new) - grad

        H = np.dot(
            (np.eye(n) - np.outer(s, y) / np.dot(y, s)),
            np.dot(H, (np.eye(n) - np.outer(y, s) / np.dot(y, s))),
        ) + np.outer(s, s) / np.dot(y, s)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        iterations += 1

    return x


# 初始化变量 x
x = np.zeros(50)

# 设置学习率和迭代次数
learning_rate_gradient_descent = 0.1
learning_rate_accelerated_gradient_descent = 0.01
num_iterations = 1000
momentum = 0.95


# 使用梯度下降法最小化函数 f(x)
minimized_gradient_descent = gradient_descent(
    x, learning_rate_gradient_descent, num_iterations
)

minimized_accelerated_gradient_descent = accelerated_gradient_descent(
    x, learning_rate_accelerated_gradient_descent, momentum, num_iterations
)

minimized_conjugate_gradient = conjugate_gradient(
    minimized_accelerated_gradient_descent, num_iterations
)

minimized_newton_method = newton_method(x, num_iterations)


minimized_bfgs_method = bfgs_method(x, num_iterations)


# 输出最小化后的结果
print("Minimized x:", minimized_gradient_descent)
print("Minimum value of f(x):", f(minimized_gradient_descent))
print(
    "==============================================================================\n"
)
# print("Minimized x:", minimized_accelerated_gradient_descent)
# print("Minimum value of f(x):", f(minimized_accelerated_gradient_descent))
print(
    "==============================================================================\n"
)
# print("Minimized x:", minimized_conjugate_gradient)
# print("Minimum value of f(x):", f(minimized_conjugate_gradient))
print(
    "==============================================================================\n"
)
# print("Minimized x:", minimized_newton_method)
# print("Minimum value of f(x):", f(minimized_newton_method))
print(
    "==============================================================================\n"
)
print("Minimized minimized_bfgs_method:", minimized_bfgs_method)
print("Minimum value of f(x):", f(minimized_bfgs_method))
