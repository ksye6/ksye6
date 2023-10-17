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
        # ����һ�����ڴ洢��һ��ʱ�䲽���¶ȷֲ�������
        T_next = np.copy(T)

        # �������˱߽��֮��������ڲ�����㣬�����ȴ������̵���ɢ��ʽ�����¶�ֵ
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                T_next[i, j] = T[i, j] + kappa * ((T[i+1, j] - 2*T[i, j] + T[i-1, j]) + (T[i, j+1] - 2*T[i, j] + T[i, j-1]))

        # �̶��߽��ϵ��¶�ֵ����
        T_next[0, :] = T_fixed
        T_next[-1, :] = T_fixed
        T_next[:, 0] = T_fixed

        # �����¶ȷֲ�����
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

    # �رս��̳�
    pool.close()
    pool.join()

