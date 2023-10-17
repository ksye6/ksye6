# -*- coding: gbk -*-
import time
import numpy as np
from multiprocessing import Process, Array, Lock
import matplotlib.pyplot as plt


def generation(start,end,random_values,lock):
  np.random.seed()
  for i in range(start,end):
    with lock:
      random_values[i]=np.random.normal(0,1)


def plot_histogram(data):

    plt.hist(data, bins=250, density=True)
    plt.title('Simulation Plot of Central Limit Theorem')
    plt.xlabel('Random Values')
    plt.ylabel('Probability')
    plt.show()


if __name__ == '__main__':
  num_processes=8
  total=200000
  random_values=Array('d',total)
  lock=Lock()
  processes=[]

  start_time=time.time()

  for i in range(num_processes):
    start=int(i*total/num_processes)
    end=int((i+1)*total/num_processes)

    p=Process(target=generation,args=(start,end,random_values,lock))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

  mean=np.mean(random_values)

  print(f"随机数均值：{mean}")
  print(f"运行时间：%.2f秒" %(time.time()-start_time))

  # 绘制直方图
  plot_histogram(random_values[:])
