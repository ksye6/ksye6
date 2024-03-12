import numpy as np
import matplotlib.pyplot as plt

################################### 1 ###################################
### (1)
def u0(x):
    return np.where(x<=0.5, 2*x, 2-2*x)

def method(J, dx, dt, tlist):
    # x-J;t-n
    x = np.linspace(0, 1, J+1)
    t = np.arange(0, tlist[-1]+dt, dt)

    # μ
    mu = dt / (dx**2)

    # 初始化
    U = np.zeros((len(t), len(x)))

    # 设置初始条件
    U[0, :] = u0(x)

    # explicit scheme
    for n in range(len(t) - 1):
        for j in range(1, J):
            U[n+1, j] = U[n, j] + mu * (U[n, j+1] - 2 * U[n, j] + U[n, j-1])

    # 绘制数值解
    fig, ax = plt.subplots()
    for i, plot_time in enumerate(tlist):
        n = int(plot_time / dt)
        ax.plot(x, U[n, :], label=f"{plot_time} unit time")

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()
    ax.grid(True)
    plt.show()

# 参数
J = 20
dx = 0.05
dt1 = 0.0012
dt2 = 0.0013
tlist = [0, dt1, 25*dt1, 50*dt1]

# (i)
method(J, dx, dt1, tlist)

# (ii)
tlist2 = [0, dt2, 25*dt2, 50*dt2]
method(J, dx, dt2, tlist2)

### (2)

dt3 = 0.0006
tlist3 = [0, dt3, 25*dt3, 50*dt3]

method(J, dx, dt3, tlist3)


### (3)
from scipy.sparse import diags

def Crank_Nicolson_method(J, dx, dt, tlist):
    # x-J;t-n
    x = np.linspace(0, 1, J+1)
    t = np.arange(0, tlist[-1]+dt, dt)

    # μ
    mu = 0.5*dt/(dx**2)

    # 初始化
    U = np.zeros((len(t), len(x)))
    U[0, :] = u0(x)

    # 系数矩阵
    A = diags([-mu, 1+2*mu, -mu], [-1, 0, 1], shape=(J+1, J+1)).toarray()
    
    # 迭代计算数值解
    for n in range(len(t) - 1):
        U[n+1, :] = np.linalg.solve(A, U[n, :])

    # 绘制数值解
    fig, ax = plt.subplots()
    for i, plot_time in enumerate(tlist):
        n = int(plot_time / dt)
        ax.plot(x, U[n, :], label=f"{plot_time} unit time")

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()
    ax.grid(True)
    plt.show()


# 参数
J = 20
dx = 0.05
dt4 = 0.0012

# (i)
dt4 = 0.0012
tlist4 = [0, dt4, 25*dt4, 50*dt4]
Crank_Nicolson_method(J, dx, dt, tlist4)

# (ii)
dt5 = 0.0013
tlist5 = [0, dt5, 25*dt5, 50*dt5]
Crank_Nicolson_method(J, dx, dt, tlist5)

# (iii)
dt6 = 0.012
tlist6 = [0, dt6, 25*dt6, 50*dt6]
Crank_Nicolson_method(J, dx, dt, tlist6)



