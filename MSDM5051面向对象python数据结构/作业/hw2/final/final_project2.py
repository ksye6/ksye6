# -*- coding: gbk -*-
import csv
import time
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from tabulate import tabulate
from pandasgui import show


## pip install ttkwidgets
from ttkwidgets.autocomplete import AutocompleteCombobox

traffic_data = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5051面向对象python数据结构\\作业\\hw2\\final\\TDCS_M06A_20190830_080000.csv",
                   header = None,
                   names = ['VehicleType','DirectionTime_O','GantryID_O','DirectionTime_D','GantryID_D','TripLength','TripEnd','TripInformation'])
# traffic_data = pd.read_csv('D:\\Python_code\\hw_4_MSDM5051\\test1.csv',
#                    header = None,
#                    names = ['VehicleType','DirectionTime_O','GantryID_O','DirectionTime_D','GantryID_D','TripLength','TripEnd','TripInformation'])

def find_via_station(data,station_node):
    data['index1'] = data[data['TripInformation'].str.contains(station_node)]['TripInformation'].apply(lambda x:x.find(station_node))-20
    data['index2'] = data[data['TripInformation'].str.contains(station_node)]['TripInformation'].apply(lambda x:x.find(station_node))-1
    data = data.fillna(0)
    data['index1'] = data['index1'].astype(int)
    data['index2'] = data['index2'].astype(int)
    data['Via_time'] = data.apply(lambda x:x['TripInformation'][x['index1']:x['index2']],axis = 1)
    return data[data['Via_time'].apply(len) > 0]

def link_start_ultra():

    # 处理函数，输出一个数据集
    def main_part():
        # 若空则初始化
        if timepoint_entry.get() == "":
            timepoint_entry.insert(tk.END, "08:00")
        if bus_entry.get() == "":
            bus_entry.insert(tk.END, "5,31,32,41,42")
        if path_entry.get() != "":
            join_data = pd.read_csv(path_entry.get(),
                    header = None,
                    names = ['VehicleType','DirectionTime_O','GantryID_O','DirectionTime_D','GantryID_D','TripLength','TripEnd','TripInformation'])
            traffic_data_join = pd.concat([traffic_data,join_data])
            print(join_data.head())
        else:
            traffic_data_join = traffic_data

        # 获取用户输入的起点站和终点站
        buslist = [int(num) for num in bus_entry.get().split(",")]
        via_station = via_station_entry.get()
        timepoint = timepoint_entry.get()

        filtered1=traffic_data_join[traffic_data_join.loc[:,"VehicleType"].isin(buslist)]
        filtered2 = find_via_station(filtered1,via_station)
        filtered2["Via_station"] = via_station
        sorted_object = filtered2.sort_values(by=["Via_time","VehicleType"],ascending=True)
        time_obj = datetime.strptime(timepoint, "%H:%M")
        formatted_time = time_obj.strftime("%H:%M:%S")
        filtered3 = sorted_object[sorted_object.loc[:,"Via_time"] > formatted_time]
        if path_entry.get() != "":
            result = filtered3.loc[:,["VehicleType","TripLength","DirectionTime_O","GantryID_O","DirectionTime_D","GantryID_D","Via_station","Via_time"]].tail(30)
        else:
            result = filtered3.loc[:,["VehicleType","TripLength","DirectionTime_O","GantryID_O","DirectionTime_D","GantryID_D","Via_station","Via_time"]].tail(30)

        return result

    # 输出查询结果
    def search_button_click():
        result = main_part()
        show(result)

    # 输出排序结果
    def sort_button_click():

        result = main_part()
        indi = combo_sort_column.get()
        sorted_result = result.sort_values(by=indi, ascending=True)
        show(sorted_result)

    # 输出排序结果
    def join_button_click():

        result = main_part()
        show(result)

    # 创建主窗口
    root = tk.Tk()
    root.title("Inquiry System of China Taiwan Traffic Data ")

    # 设置窗口大小
    root.geometry('800x1000')

    # 标题
    title = tk.Label(root, text='Check any history traffic data you want', font=('Arial', 25, 'bold'))#, width=20, height=3
    title.pack()

    # 子标题1
    subtitle1 = tk.Label(root, text='Search', font=('Arial', 20, 'bold'), width=10, height=2)
    subtitle1.place(x=2, y=250)

    # 创建公交标签和输入框
    bus_label = tk.Label(root, text="VehicleType:", font=('Arial', 15, 'bold'))
    bus_label.place(x=150, y=165)
    # bus_label.pack()

    bus_entry = tk.Entry(root, font=('Arial', 15, 'bold'))
    bus_entry.place(x=500, y=165)
    # bus_entry.pack()

    # 创建途径站标签和输入框
    via_station_label = tk.Label(root, text="Via station:", font=('Arial', 15, 'bold'))
    via_station_label.place(x = 150, y=250)
    # start_station_label.pack()

    all_station_list = list(np.sort(list(set(traffic_data.loc[:,"GantryID_O"].unique().tolist() + traffic_data.loc[:,"GantryID_D"].unique().tolist()))))
    via_station_entry = AutocompleteCombobox(completevalues= all_station_list, font=('Arial', 15, 'bold'), width=15)
    default_station_column = all_station_list[0]
    via_station_entry.set(default_station_column)
    via_station_entry.place(x=500, y=250)
    # via_station_entry.pack()


    # 创建时间点标签和输入框
    timepoint_label = tk.Label(root, text="Time passing through via_station:", font=('Arial', 15, 'bold'))
    timepoint_label.place(x = 150, y=325)
    # timepoint_label.pack()

    start_time = "08:00"
    end_time = "09:00"
    interval = 1  # 间隔时间，单位为分钟
    start_hour, start_minute = map(int, start_time.split(":"))
    end_hour, end_minute = map(int, end_time.split(":"))
    start_total_minutes = start_hour * 60 + start_minute
    end_total_minutes = end_hour * 60 + end_minute
    time_list = []
    current_minutes = start_total_minutes
    while current_minutes <= end_total_minutes:
        hour = current_minutes // 60
        minute = current_minutes % 60
        time_str = f"{hour:02d}:{minute:02d}"
        time_list.append(time_str)
        current_minutes += interval

    timepoint_entry = AutocompleteCombobox(completevalues=time_list, font=('Arial', 15, 'bold'))
    timepoint_entry.place(x=500, y=325)
    # timepoint_entry.pack()

    # 创建搜索按钮
    search_button = tk.Button(root, text="Click to search", command=search_button_click, font=('Arial', 15, 'bold'))
    search_button.place(x=55, y=415)
    # search_button.pack()

    # 排序部分的布局
    subtitle2 = tk.Label(root, text='Sort', font=('Arial', 20, 'bold'), width=10, height=2)
    subtitle2.place(x=2, y=600)

    # 排序列标签和下拉框
    label_sort_column = tk.Label(root,text='sort columns:', font=('Arial', 15, 'bold'))
    label_sort_column.place(x=150, y=600)

    COLUMN_NAMES = ["VehicleType","TripLength","Via_time"]
    combo_sort_column = AutocompleteCombobox(completevalues=COLUMN_NAMES, font=('Arial', 15, 'bold'), width=15)
    default_sort_column = COLUMN_NAMES[0]
    combo_sort_column.set(default_sort_column)
    combo_sort_column.place(x=400, y=600)

    # 排序按钮
    sort_button = tk.Button(root, text="Click to sort", command=sort_button_click, font=('Arial', 15, 'bold'))
    sort_button.place(x=55, y=700)

    # 创建公交标签和输入框
    path_label = tk.Label(root, text="Join:", font=('Arial', 20, 'bold'), width=10, height=2)
    path_label.place(x=2, y = 800)

    # 排序部分的布局
    subtitle3 = tk.Label(root, text='input file path', font=('Arial', 15, 'bold'))
    subtitle3.place(x=150, y = 800)

    # 创建合并标签与输入框
    path_entry = tk.Entry(root, font=('Arial', 15, 'bold'))
    path_entry.place(x=400, y= 800)

    # 创建合并按钮
    merge_botton = tk.Button(root, text="Click to join", command = join_button_click, font=('Arial', 15, 'bold'))
    merge_botton.place(x=55, y=900)

    # 启动主循环
    root.mainloop()

link_start_ultra()
