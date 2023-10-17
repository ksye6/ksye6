# -*- coding: utf-8 -*-
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

def dTdt(T):
    return -T

def solve_differential_eqn(dt, num_steps, T0):
    T = np.zeros(num_steps)
    T[0] = T0

    for i in range(num_steps - 1):
        T[i+1] = T[i] + dt * dTdt(T[i])

    return T

def exact_solution(t):
    return np.exp(-t)

if __name__ == '__main__':
    delta_t = 0.01
    num_steps = int(10 / delta_t) + 1
    num_processes = mp.cpu_count()
    results=[]
    
    pool = mp.Pool(processes=num_processes)

    T0 = 1.0
    
    for i in range(num_processes):
      result = pool.apply_async(solve_differential_eqn, args=(delta_t/num_processes, num_steps, T0))
      results.append(result)
      T0 = result.get()[-1]  # 使用上一个进程的最后一个值作为下一个进程的初始值

    pool.close()
    pool.join()

    T_parallel = np.concatenate([result.get() for result in results])
    t_parallel = np.linspace(0, 10, len(T_parallel))

    t_exact = np.linspace(0, 10, num_steps)
    T_exact = exact_solution(t_exact)

    plt.plot(t_exact, T_exact, label='Exact Solution')
    plt.plot(t_parallel, T_parallel, label='Parallel Solution')
    plt.xlabel('t')
    plt.ylabel('T(t)')
    plt.legend()
    plt.show()


