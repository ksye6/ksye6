import numpy as np

# 1.(2)

# 定义函数及其导数
def f(x):
    return 4*x*np.sin(x) - 4*np.sin(x)**2 - x**2

def f_prime(x):
    return 4*np.sin(x) + 4*x*np.cos(x) - 8*np.sin(x)*np.cos(x) - 2*x

# (i)
# 牛顿迭代法求解函数零点
def newton_method(x0, tol=1e-5, max_iter=1000):
    x = x0
    iter_count = 0
    
    while iter_count < max_iter:
        iter_count += 1
        delta_x = f(x) / f_prime(x)
        x -= delta_x
        # print(x)
        
        if np.abs(delta_x) < tol:
            break
    
    if iter_count == max_iter:
        print("达到最大迭代次数但未收敛到解")
    
    return x

# 初始值
x0 = 1.5

# 求解函数的零点
zero_point1 = newton_method(x0)

print("Newton’s method: 函数的零点解为:", zero_point1)

# (ii)
# 割线法求解函数零点
def secant_method(x0, x1, tol=1e-5, max_iter=100):
    x = x1
    x_prev = x0
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1
        delta_x = f(x) * (x - x_prev) / (f(x) - f(x_prev))
        x_prev = x
        x -= delta_x
        # print(x)

        if np.abs(delta_x) < tol:
            break

    if iter_count == max_iter:
        print("达到最大迭代次数但未收敛到解")

    return x

# 初始值
x0 = 1.5
x1 = 2.0

# 求解函数的零点
zero_point2 = secant_method(x0, x1)

print("the secant method: 函数的零点解为:", zero_point2)


# 2.(2)

# 定义函数
def F(x):
    x1, x2 = x
    f1 = 1 + x1**2 - 4*x2**2 + np.exp(x1)*np.cos(2*x2)
    f2 = 4*x1*x2 + np.exp(x1)*np.sin(2*x2)
    return np.array([f1, f2])

# 定义雅可比矩阵
def J(x):
    x1, x2 = x
    df1_dx1 = 2*x1 + np.exp(x1)*np.cos(2*x2)
    df1_dx2 = -8*x2 - 2*np.exp(x1)*np.sin(2*x2)
    df2_dx1 = 4*x2 + np.exp(x1)*np.sin(2*x2)
    df2_dx2 = 4*x1 + 2*np.exp(x1)*np.cos(2*x2)
    return np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])

# 牛顿迭代法求解非线性方程组
def newton_method(x0, max_iter=5):
    x = x0
    
    for i in range(max_iter):
        delta_x = np.linalg.solve(J(x), -F(x))
        x += delta_x

    return x

# 初始值
x0 = np.array([-1, 2], dtype=np.float64)

# 求解非线性方程组
solution = newton_method(x0, max_iter=5)

print("Newton’s method: 迭代结果:", solution)


# 3

def P(x):
    return (-5*x**2+3*x+26)/6

P(1)
P(2)
P(-1)


# 6
import numpy as np

x = np.array([1.0, 1.1, 1.3, 1.5, 1.9, 2.1])
A = np.array([[1.0]*6, x, x**2, x**3]).T
b = np.array([1.84, 1.96, 2.21, 2.45, 2.94, 3.18]).T
U, S, V_T = np.linalg.svd(A)
C = np.dot(U.T,b)
Z = C[:len(S)]/S
coefficients = np.dot(V_T.T,Z)
least_square_error=np.sqrt(np.sum(C[len(S):] ** 2))  # np.linalg.norm(C[len(S):])
print("the least squares polynomial of degree 3: ","y=0.629+1.185x+0.0353x^2-0.010x^3",";least_square_error:",least_square_error)


