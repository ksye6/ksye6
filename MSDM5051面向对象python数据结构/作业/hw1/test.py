
import csv
# 
# # 打开CSV文件
# with open('C:/Users/mzhangdb/Desktop/data/TDCS_M06A_20190830_080000.csv', newline='') as csvfile:
#   reader = csv.reader(csvfile)
#   VT = []
#   DT = []
#   GID = []
#   for row in reader:
#     c1 = row[:1]
#     VT.append(c1)
#     c2 = row[:2]
#     DT.append(c2)
#     c3 = row[:3]
#     GID.append(c3)
#     
# 
# VT[:10]
# DT[:10]
# GID[:10]

import time
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_sort_result(func, data):
    start = time.time()
    data_copy = data[:] # important -- copy the list, instead of copy the reference.
    result = func(data_copy)
    time_used = time.time() - start
    for i in range(len(result)-1):
        if result[i] > result[i+1]:
            print("Check failed: func(data) is not sorted properly.")
            return
    print("The method {0} returned successfully for data size {1} with {2} seconds.".format(func.__name__.center(15), len(result), time_used))

def check_performance(func, data, size_samples = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]):
    timing_array = []
    for size in size_samples:
        n_repeated = size_samples[-1] // size
        start = time.time()
        for run in range(n_repeated):
            data_truncated = data[:size]
            func(data_truncated)
        timing_array.append((time.time() - start) / n_repeated)
    ref_linear = [i * timing_array[0] / size_samples[0] for i in size_samples]
    ref_quadratic = [i**2 * timing_array[0] / size_samples[0]**2 for i in size_samples]
    plt.loglog(size_samples, ref_linear, size_samples, ref_quadratic, size_samples, timing_array)
    plt.show()


traffic_data = pd.read_csv("C:\\Users\\mzhangdb\\Desktop\\data\\TDCS_M06A_20190830_080000.csv",header=None)
traffic_data = traffic_data.iloc[:10000,]

# traffic_data.iloc[1,1]  #位置选取
# traffic_data.loc[1,1]   #标签索引

VT = traffic_data.iloc[:,0]
DT = traffic_data.iloc[:,1]
GID = traffic_data.iloc[:,2]

distances = [row[5] for row in traffic_data.values.tolist()][:10000]

import sys
module_path = 'C:\\Users\\mzhangdb\\Desktop'
sys.path.append(module_path)

from sorting import * 

# check_sort_result(bubble_sort,VT.tolist())
# check_performance(bubble_sort,VT.tolist())

# check_sort_result(insertion_sort,VT.tolist())
# check_performance(insertion_sort,VT.tolist())

# sys.setrecursionlimit(50000)
# check_sort_result(quicksort,VT.tolist())
# check_performance(quicksort,VT.tolist())

check_sort_result(mergesort,DT.tolist())
check_performance(mergesort,VT.tolist())

# check_sort_result(heapsort,VT.tolist())
# check_performance(heapsort,VT.tolist())

from sortedcontainers import SortedDict

def BST_sort(data):
  bst = SortedDict()

  for val in data:
    if val in bst:
      bst[val] += 1
    else:
      bst[val] = 1

  sorted_data = []
  for key, count in bst.items():
    sorted_data.extend([key] * count)
  
  return sorted_data

check_sort_result(BST_sort,DT.tolist())
check_performance(BST_sort,DT.tolist())


