import numpy as np

def f(x):
    first_term = (x[0]) ** 2
    sum_term = sum([(x[i] - x[i+1]) ** 2 for i in range(49)])
    last_term = -(x[50]) ** 2
    fx = 0.5 * (first_term + sum_term + last_term) - x[0]
    return fx


def gradient(x):
    grad_fx = np.zeros_like(x)
    
    # 对于第一个元素
    grad_fx[0] = x[0] - 1 - 2 * x[0]

    # 对于中间的48个元素
    for i in range(1, 49):
        grad_fx[i] = 2 * (x[i] - x[i+1])

    # 对于最后一个元素
    grad_fx[49] = -2 * x[50]

    return grad_fx


"""
gradient descent
"""
def gradient_descent(x, learning_rate, num_iterations):
    for _ in range(num_iterations):
        grad = gradient(x)
        x -= learning_rate * grad
    return x


"""
accelerated gradient descent
"""
def accelerated_gradient_descent(x_init, learning_rate, momentum, num_iterations):
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
def conjugate_gradient(x0, tol=1e-3, max_iter=1000):
    x = x0.copy()
    r = -gradient(x)  # 初始残差，等于负梯度
    p = r.copy()  # 初始搜索方向
    rTr = np.dot(r, r)  # 初始残差与自身的内积
    iteration = 0

    while rTr > tol**2 and iteration < max_iter:
        alpha = rTr / np.dot(p, gradient(p))  # 梯度下降步长
        x += alpha * p  # 更新位置
        old_r = r.copy()
        r -= alpha * gradient(p)  # 更新残差
        beta = np.dot(r, r) / rTr  # 共轭梯度公式中的β
        rTr = np.dot(r, r)  # 更新残差与自身的内积
        p = r + beta * p  # 更新搜索方向

        iteration += 1

    return x



# 初始化变量 x
x = np.zeros(100)

# 设置学习率和迭代次数
learning_rate_gradient_descent = 0.1
learning_rate_accelerated_gradient_descent=0.01
num_iterations = 1000
momentum = 0.95
tol=1e-6


# 使用梯度下降法最小化函数 f(x)
minimized_gradient_descent = gradient_descent(x, learning_rate_gradient_descent, num_iterations)

minimized_accelerated_gradient_descent = accelerated_gradient_descent(x, learning_rate_accelerated_gradient_descent, momentum, num_iterations)

minimized_conjugate_gradient = conjugate_gradient(x, tol, num_iterations)




# 输出最小化后的结果
print("Minimized x:", minimized_gradient_descent)
print("Minimum value of f(x):", f(minimized_gradient_descent))
print("==============================================================================\n")
print("Minimized x:", minimized_accelerated_gradient_descent)
print("Minimum value of f(x):", f(minimized_accelerated_gradient_descent))
print("==============================================================================\n")
print("Minimized x:", minimized_conjugate_gradient)
print("Minimum value of f(x):", f(minimized_conjugate_gradient))


