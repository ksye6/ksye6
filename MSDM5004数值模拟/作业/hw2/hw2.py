
import numpy as np

# 2
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
        
        error = np.linalg.norm(x_new - x, ord=np.inf)  # l�� norm
        x = np.copy(x_new)
        iterations += 1
    
    return x

# ϵ������A
A = np.array([[4, 1, 1, -1],
              [1, 4, -1, -1],
              [1, -1, 3, 1],
              [-1, -1, 1, 5]])

# ����b
b = np.array([-3, -2, 2, 5])

# ��ʼ����x(0)
x0 = np.array([0, 0, 0, 1], dtype=float)

# ����
epsilon = 1e-3

solution = Gauss_Seidel(A, b, x0, epsilon)

print("G-S ��:", solution)


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
        
        error = np.linalg.norm(x_new - x, ord=np.inf)  # l�� norm
        x = np.copy(x_new)
        iterations += 1
    
    return x

# ϵ������A
A = np.array([[4, 1, 1, -1],
              [1, 4, -1, -1],
              [1, -1, 3, 1],
              [-1, -1, 1, 5]])

# ����b
b = np.array([-3, -2, 2, 5])

# ��ʼ����x(0)
x0 = np.array([0, 0, 0, 1], dtype=float)

# ����
epsilon = 1e-3

# �ɳڲ���
omega = 1.2

solution2 = SOR_method(A, b, x0, omega, epsilon)

print("SOR ��:", solution2)


# 3
def f(x):
    return np.sin(x) / x**2

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# ���㵼��
x = 3
h_values = np.array([0.08, 0.04, 0.02])
derivatives = [central_difference(f, x, h) for h in h_values]
derivatives # -0.12

# ������������
(derivatives[0]-derivatives[1])/(derivatives[1]-derivatives[2])
# 2^2,  the order of convergence is 2.


# 4

def f(t, y):
    return t**(-2) * (t*y - y**2/2)

def forward_euler(f, t_values, y0, h):
    num_points = len(t_values)
    y_values = np.zeros(num_points)
    y_values[0] = y0

    for i in range(1, num_points):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])

    return y_values

from scipy.optimize import fsolve
def backward_euler(f, t_values, y0, h):
    num_points = len(t_values)
    y_values = np.zeros(num_points)
    y_values[0] = y0

    for i in range(1, num_points):
        func_eq = lambda Xnp : Xnp - y_values[i-1] - h * f(t_values[i],Xnp)
        y_values[i] = fsolve(func_eq, y_values[i-1])
    return y_values

# ����ʱ�䲽����ʱ���
h = 1/128
t_values = np.arange(1, 3+h, h)

# ʹ��ǰ��ŷ�������
y0 = 4
y_forward = forward_euler(f, t_values, y0, h)

y_forward[-1]

# ʹ�ú���ŷ�������
y_backward = backward_euler(f, t_values, y0, h)

y_backward[-1]

# ��ʹ��solve_ivpֱ����΢�ַ��̽�
# from scipy.integrate import solve_ivp
# sol = solve_ivp(f, (1, 3), [4])
# print(sol.t)      # ʱ���
# print(sol.y[0])   # ��Ӧ�Ľ�
