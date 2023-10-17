# -*- coding: gbk -*-
import time
from multiprocessing import Process, Value, Lock

def sender(pid,value,lock):
  if pid==0:
    with lock:
      value.value=100
      
  else:
    with lock:
      received_value=value.value
      print(f"进程 {pid} 接收到的整数：{received_value}")
      value.value=received_value

def receiver(pid,value,lock):
  if pid!=0:
    with lock:
      value.value=100**2
      
  else:
    with lock:
      received_value=value.value
      print(f"进程 {pid} 接收到的结果：{received_value}")

if __name__ == "__main__":
  np=4
  value=Value('i',0)
  lock=Lock()
  processes1=[]
  processes2=[]
  
  start_time=time.time()
  
  for pid in range(np):
    p=Process(target=sender,args=(pid,value,lock))
    processes1.append(p)
    p.start()
  
  for p in processes1:
    p.join()
  
  for pid in range(np-1,-1,-1):
    p2=Process(target=receiver,args=(pid,value,lock))
    processes2.append(p2)
    p2.start()

  for p in processes2:
    p2.join()

  print(f"运行时间：%.2f秒" %(time.time()-start_time))
