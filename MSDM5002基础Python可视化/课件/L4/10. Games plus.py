import numpy as np
import time
import random
import math


num_pos=10
nbu=3
pos_take=[0,1,2,3,4]

##1

def func_check(num_test,pos_take):
  num_lose=0
  for nt in range(num_test):
    A=np.zeros(num_pos)
    
    #setup the first bullet
    B1=np.random.randint(0,num_pos)
    A[B1]=1
    
    #setup the second bullet
    B2=B1
    while B2==B1:
        B2=np.random.randint(0,num_pos)
    A[B2]=1
    
    #setup the third bullet
    B3=B1
    while B3==B1 or B3==B2:
        B3=np.random.randint(0,num_pos)
    A[B3]=1

    for n in range(num_pos):
        if A[n]==1:
            if n in pos_take:
                num_lose += 1
            break
    
  return(num_lose/num_test)

##2

def func_check2(num_test,nbu,pos_take):
  num_lose=0
  A=np.zeros(num_pos,int)
  A[pos_take]=1
  
  for nt in range(num_test):
    B1=min(random.sample(range(num_pos),nbu)) #np.random.choice(range(num_pos), size=nbu, replace=False, p=None)
    #types are different
    if A[B1]==1:
      num_lose+=1
    
  return(num_lose/num_test)


##3

n=10
b=3

def func_check3(n,b,pos_take):
  x=np.zeros(n,float)
  for k in range(0,n):
    if k+1>n-b+1:
      x[k]=0
    else:
      x[k]=math.factorial(n-b)*b*math.factorial(n-(k+1))/(math.factorial(n-b-(k+1)+1)*math.factorial(n))#easy to calculate
  
  return(sum(x[pos_take]))


##time test
pos_take_list=[pos_take,[9,8,7,6,5],[0,2,4,6,8]]

test_num=3
T=np.empty([3,test_num])

for i in range(len(pos_take_list)):
  for n in range(test_num):
    start_T=time.time(); func_check(100000,pos_take_list[i]);  T[0,n]=time.time()-start_T
    
    start_T=time.time(); func_check2(100000,3,pos_take_list[i]); T[1,n]=time.time()-start_T
    
    start_T=time.time(); func_check3(10,3,pos_take_list[i]); T[2,n]=time.time()-start_T

  test_time=T.sum(axis=1)/test_num
  print(test_time)










