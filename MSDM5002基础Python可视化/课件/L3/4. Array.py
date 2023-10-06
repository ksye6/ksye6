# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:15:17 2021

@author: cmt
"""
import numpy as np

n1=2;n2=3;n3=4;
A1 =np.array([x for x in range(n1)])
A2 =np.array([[x+y*10 for x in range(n1)] for y in range(n2)])
A3 =np.array([[[x+y*10+z*100 for x in range(n1)] 
               for y in range(n2)] 
              for z in range(n2)])


for x in A1:
    print(x)

print()
for x in A2:
    print(x)

print()
for x in A2:
    print(x)
    for y in x:
        print(y)

print()
for x in A3:
    print(x)
    
print()  
for x in np.nditer(A3):
    print(x)

for x in np.nditer(A3[:, ::2]):
    print(x)     
    
for idx, x in np.ndenumerate(A3):
    print(idx, x)
    if x==100:
        A3[idx]=100000
    
    
  
# for x in np.nditer(A3, flags=['buffered'], op_dtypes=['S']):
#     print(x)
    
# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[5, 6], [7, 8]])
# arr_j0 = np.concatenate((arr1, arr2), axis=0)
# arr_j1 = np.concatenate((arr1, arr2), axis=1)

# print(arr_j0)
# print(arr_j1)


# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# arr_s0 = np.stack((arr1, arr2), axis=0)
# arr_s1 = np.stack((arr1, arr2), axis=0)
# arr_hs = np.hstack((arr1, arr2))
# arr_vs = np.vstack((arr1, arr2))
# arr_ds = np.dstack((arr1, arr2))
# print(arr_s0)
# print(arr_s1)
# print(arr_hs)
# print(arr_vs)
# print(arr_ds)

# arr = np.array([1, 2, 3, 4, 5, 6])
# newarr = np.array_split(arr, 4)


# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
# newarr2d_a0 = np.array_split(arr2d, 3)
# newarr2d_a1 = np.array_split(arr2d, 3, axis=1)

# print(newarr2d_a0) 
# print(newarr2d_a1) 

# arr = np.array([1, 2, 3, 4, 2, 6])
# x = np.where(arr%2 == 0)

# print(x)

# #The method starts the search from the left and returns the first index
# # where the number 2.5 is no longer larger than the next value.
# x_s = np.searchsorted(arr, 2.5)
# x_ms = np.searchsorted(arr, [2.5,4.5,6.5])


# arr = np.array([[3,2,4], [5,0,1],[7,9,3]])
# print(np.sort(arr,axis=0)) 
# print(np.sort(arr,axis=1))

# ind=np.argsort(arr,axis=0)
# print(np.take_along_axis(arr,ind,axis=0))

# #sort based on the first row
# print(arr[:, arr[0].argsort()])

# arr = np.array([41,42,43,44])
# x = [True,False,True,False]
# newarr = arr[x]
# print(newarr) 

# filter_arr = arr > 42
# print(arr[filter_arr])

# filter_arr = arr%2==0
# print(arr[filter_arr])


# # Create an empty list
# filter_arr = []

# # go through each element in arr
# for element in arr:
#   # if the element is completely divisble by 2, set the value to True, otherwise False
#   if element % 2 == 0:
#     filter_arr.append(True)
#   else:
#     filter_arr.append(False)



# ### id of different element of list and array








