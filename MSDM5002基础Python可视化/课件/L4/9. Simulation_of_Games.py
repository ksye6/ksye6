# -*- coding: utf-8 -*-
"""
Here we use random number to simulate a game
@author: Junwei Liu
"""
import numpy as np
import time

#only one bullet in the gun
num_test=10000
num_pos=6             #maximum number of bullets in the gun
pos_take=[1,3,4]      #the orders taking shoots for the player

start_time=time.time()

num_lose=0
for nt in range(num_test):
    A=np.zeros(num_pos,bool)
    A[np.random.randint(0,num_pos)]=True
    #more precise simulations for the scenario in the video
    #since the first shot is empty
    # A[np.random.randint(1,num_pos)]=True
    
    #more elegent
    if any(A[pos_take]):
        num_lose += 1
    
    # ##easy to extend    
    # for n in range(num_pos):
    #     if A[n]==1:
    #         if n in pos_take:
    #             num_lose += 1
    #         break

print("The lose probability is:", num_lose/num_test)
print("Time:",time.time()-start_time)


#multiple bullets in the gun
num_test=10000
num_pos=10

pos_take=[0,1,2,3,4]

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

print("The lose probability is:", num_lose/num_test)



