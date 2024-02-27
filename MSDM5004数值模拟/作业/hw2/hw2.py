
import numpy as np

# (1) the Gauss-Seidel method

def Gauss_Seidel(A, b, x0, epsilon):
    n = len(A)
    x = np.copy(x0)
    iterations = 0
    error = 100
    
    while error > epsilon:
        x_new = np.copy(x)
        
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        error = np.linalg.norm(x_new - x, ord=np.inf)  # l∞ norm
        x = np.copy(x_new)
        iterations += 1
    
    return x

# 系数矩阵A
A = np.array([[4, 1, 1, -1],
              [1, 4, -1, -1],
              [1, -1, 3, 1],
              [-1, -1, 1, 5]])

# 向量b
b = np.array([-3, -2, 2, 5])

# 初始估计x(0)
x0 = np.array([0, 0, 0, 1], dtype=float)

# 精度
epsilon = 1e-3

solution = Gauss_Seidel(A, b, x0, epsilon)

print("G-S 解:", solution)


# (2) the the SOR method

def SOR_method(A, b, x0, omega, epsilon):
    n = len(A)
    x = np.copy(x0)
    iterations = 0
    error = 100
    
    while error > epsilon:
        x_new = np.copy(x)
        
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - s1 - s2)
        
        error = np.linalg.norm(x_new - x, ord=np.inf)  # l∞ norm
        x = np.copy(x_new)
        iterations += 1
    
    return x

# 系数矩阵A
A = np.array([[4, 1, 1, -1],
              [1, 4, -1, -1],
              [1, -1, 3, 1],
              [-1, -1, 1, 5]])

# 向量b
b = np.array([-3, -2, 2, 5])

# 初始估计x(0)
x0 = np.array([0, 0, 0, 1], dtype=float)

# 精度
epsilon = 1e-3

# 松弛参数
omega = 1.2

solution2 = SOR_method(A, b, x0, omega, epsilon)

print("SOR 解:", solution2)










