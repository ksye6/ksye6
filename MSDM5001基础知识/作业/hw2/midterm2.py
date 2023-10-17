# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp

nx, ny = 121, 121
num_steps = 240
delta_t = 0.01
delta_x = 0.01
T_fixed = 40
kappa = 0.01


def solve_heat_equation(dt, dx, num_steps, T, T_fixed=40):
    
    for n in range(num_steps):
        # 创建一个用于存储下一个时间步的温度分布的数组
        T_next = np.copy(T)

        # 遍历除了边界点之外的所有内部网格点，根据热传导方程的离散形式更新温度值
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                T_next[i, j] = T[i, j] + kappa * ((T[i+1, j] - 2*T[i, j] + T[i-1, j]) + (T[i, j+1] - 2*T[i, j] + T[i, j-1]))

        # 固定边界上的温度值不变
        T_next[0, :] = T_fixed
        T_next[-1, :] = T_fixed
        T_next[:, 0] = T_fixed

        # 更新温度分布数组
        T = np.copy(T_next)

    return T


if __name__ == '__main__':
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    results = []
    T0 = np.full((nx, ny), 20, float)
    
    for _ in range(num_processes):
        result = pool.apply_async(
            solve_heat_equation,
            args=(delta_t, delta_x, int(num_steps/num_processes), T0, T_fixed)
        )
        results.append(result)
        T0 = result.get()  

    T_parallel = results[-1].get()

    print(T_parallel)

    # 关闭进程池
    pool.close()
    pool.join()

