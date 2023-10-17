# -*- coding: gbk -*-
import time
import math
from multiprocessing import Process, Value, Lock


def maincalculation(start,end,result,lock):
  sum_=0
  for i in range(start,end):
    sum_ += 1/math.factorial(i)
  
  with lock:
    result.value+=sum_

if __name__ == '__main__':
  num_processes=4
  num_iterations=1000
  result=Value('d',0.0)
  lock=Lock()
  processes=[]

  start_time=time.time()

  for i in range(num_processes):
    start=int(i*num_iterations/num_processes)
    end=int((i+1)*num_iterations/num_processes)

    p=Process(target=maincalculation,args=(start,end,result,lock))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

  print(f"欧拉数的值为：{result.value}")
  print(f"运行时间：%.2f秒" %(time.time()-start_time))
