# -*- coding: gbk -*-
import csv
import time
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import datetime

## 忽略此块
#################################################### 测试+思路 #########################################

traffic_data = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5051面向对象python数据结构\\作业\\hw2\\TDCS_M06A_20190830_080000.csv")

# 输出所有车，若确定车，搜索车，否则全搜→输出所有会经过的车站名，继续搜索起点站和终点站→按需要到达的时间排序→输出/过滤某个时间点的车次信息

# 所有车
set(traffic_data.loc[:,"VehicleType"].unique())
# 筛选某些车名
busnameset=set()
busnameset.update([5,31,41])
filtered1=traffic_data[traffic_data.loc[:,"VehicleType"].isin(busnameset)]

# 所有车站
set(traffic_data.loc[:,"GantryID_O"].unique())
# set(traffic_data.loc[:,"GantryID_D"].unique()) 一样的
# 筛选当前车站和目的地车站名
start_stationname="01F0005S"
filtered2=filtered1[filtered1.loc[:,"TripInformation"].str.contains(start_stationname)]
end_stationname="01H0447S"
filtered2=filtered2[filtered2.loc[:,"TripInformation"].str.contains(end_stationname)]

# 筛选时间点
# 提取"information"里当前车站名之前的字符串
pattern = f"(.{{8}})\+"+start_stationname
extracted_strings = filtered2.loc[:,"TripInformation"].str.extract(pattern, expand=False)
# 将提取的字符串存储为新列,排序
sorted_object = filtered2.copy()
sorted_object["RealTime"] = extracted_strings
sorted_object["Start"] = start_stationname
sorted_object["End"] = end_stationname
sorted_object = sorted_object.sort_values(by=["RealTime","VehicleType"],ascending=True)

# 某个时间点之后的车次
Timepoint = "08:30"
time_obj = datetime.strptime(Timepoint, "%H:%M")
formatted_time = time_obj.strftime("%H:%M:%S")
filtered3 = sorted_object[sorted_object.loc[:,"RealTime"] > formatted_time]

result = filtered3.loc[:,["Start","End","RealTime","VehicleType","TripLength"]].head(30)
print(result)

###########################################################################################################
import tkinter as tk
from tkinter import ttk

## pip install ttkwidgets
from ttkwidgets.autocomplete import AutocompleteCombobox

traffic_data = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5051面向对象python数据结构\\作业\\hw2\\TDCS_M06A_20190830_080000.csv")

# 忽略此函数(试验品)
def test():

    def search_button_click():
        # 若空则初始化
        if timepoint_entry.get() == "":
            timepoint_entry.insert(tk.END, "08:00")
        if bus_entry.get() == "":
            bus_entry.insert(tk.END, "5,31,32,41,42")

        # 获取用户输入的起点站和终点站
        buslist = [int(num) for num in bus_entry.get().split(",")]
        start_station = start_station_entry.get()
        end_station = end_station_entry.get()
        timepoint = timepoint_entry.get()

        filtered1=traffic_data[traffic_data.loc[:,"VehicleType"].isin(buslist)]
        filtered2=filtered1[filtered1.loc[:,"TripInformation"].str.contains(start_station)]
        filtered2=filtered2[filtered2.loc[:,"TripInformation"].str.contains(end_station)]
        pattern = f"(.{{8}})\+"+start_station
        extracted_strings = filtered2.loc[:,"TripInformation"].str.extract(pattern, expand=False)
        filtered2["RealTime"] = extracted_strings
        filtered2["Start"] = start_station
        filtered2["End"] = end_station
        sorted_object = filtered2.sort_values(by=["RealTime","VehicleType"],ascending=True)
        time_obj = datetime.strptime(timepoint, "%H:%M")
        formatted_time = time_obj.strftime("%H:%M:%S")
        filtered3 = sorted_object[sorted_object.loc[:,"RealTime"] > formatted_time]
        result = filtered3.loc[:,["Start","End","RealTime","VehicleType","TripLength"]].head(30)

        # 更新结果显示
        # 清空结果显示框
        result_text.delete('1.0', tk.END)
        # 将处理好的数据集输出到结果显示框
        result_text.insert(tk.END, result.to_string(index=False))
        # 设置文本样式和对齐方式
        result_text.tag_configure("center", justify="center")
        result_text.tag_add("center", "1.0", "end")

    # 创建主窗口
    root = tk.Tk()
    root.title("交通数据查询")

    # 设置窗口大小
    root.geometry("1080x960")

    # # 创建label
    # additional_text = ttk.Label(root, text="All buses: "+str(traffic_data.loc[:,"VehicleType"].unique())[1:-1], anchor="n")
    # additional_text.pack()

    # 创建多行文本框
    text_widget = tk.Text(root, height=30, width=100)
    text_widget.pack()

    # 插入多行文本
    text_widget.insert(tk.END, "全部公交:\n"+str(np.sort(traffic_data.loc[:,"VehicleType"].unique()))[1:-1]+"\n")
    text_widget.insert(tk.END, "全部车站:\n "+str(np.sort(traffic_data.loc[:,"GantryID_O"].unique()))[1:-1].replace("'", ""))

    # 设置文本样式和对齐方式
    text_widget.tag_configure("center", justify="center")
    text_widget.tag_add("center", "1.0", "end")

    # 创建公交标签和输入框
    bus_label = tk.Label(root, text="所需公交(格式如:5,31,41):")
    bus_label.pack()
    bus_entry = tk.Entry(root)
    bus_entry.pack()

    # 创建起点站标签和输入框
    start_station_label = ttk.Label(root, text="起点站(格式如:01F0005S):")
    start_station_label.pack()
    start_station_entry = ttk.Entry(root)
    start_station_entry.pack()

    # 创建终点站标签和输入框
    end_station_label = ttk.Label(root, text="终点站(格式如:01H0447S):")
    end_station_label.pack()
    end_station_entry = ttk.Entry(root)
    end_station_entry.pack()

    # 创建时间点标签和输入框
    timepoint_label = ttk.Label(root, text="到达起点站时间(仅限8-9点,格式如:08:30):")
    timepoint_label.pack()
    timepoint_entry = ttk.Entry(root)
    timepoint_entry.pack()

    # 创建搜索按钮
    search_button = ttk.Button(root, text="单击搜索", command=search_button_click)
    search_button.pack()

    # 创建结果显示框
    result_text = tk.Text(root, height=300, width=200)
    result_text.pack()

    # 启动主循环
    root.mainloop()

test()

##########################################################################################################
################################################  GUI  ###################################################
##########################################################################################################
# 只需要 import 和 导入数据集 即可，最好电脑上有 思源黑体CN bold 的字体

# 全新版本
def link_start_ultra():
    
    # 处理函数，输出一个数据集
    def main_part():
        # 若空则初始化
        if timepoint_entry.get() == "":
            timepoint_entry.insert(tk.END, "08:00")
        if bus_entry.get() == "":
            bus_entry.insert(tk.END, "5,31,32,41,42")
        
        # 获取用户输入的起点站和终点站
        buslist = [int(num) for num in bus_entry.get().split(",")]
        start_station = start_station_entry.get()
        end_station = end_station_entry.get()
        timepoint = timepoint_entry.get()
        
        filtered1=traffic_data[traffic_data.loc[:,"VehicleType"].isin(buslist)]
        filtered2=filtered1[filtered1.loc[:,"TripInformation"].str.contains(start_station)]
        filtered2=filtered2[filtered2.loc[:,"TripInformation"].str.contains(end_station)]
        pattern = f"(.{{8}})\+"+start_station
        extracted_strings = filtered2.loc[:,"TripInformation"].str.extract(pattern, expand=False)
        filtered2["RealTime"] = extracted_strings
        filtered2["Start"] = start_station
        filtered2["End"] = end_station
        sorted_object = filtered2.sort_values(by=["RealTime","VehicleType"],ascending=True)
        time_obj = datetime.strptime(timepoint, "%H:%M")
        formatted_time = time_obj.strftime("%H:%M:%S")
        filtered3 = sorted_object[sorted_object.loc[:,"RealTime"] > formatted_time]
        result = filtered3.loc[:,["Start","End","RealTime","VehicleType","TripLength"]].head(30)
        return result
    
    # 输出查询结果
    def search_button_click():
        result = main_part()
        
        # 更新结果显示
        # 清空结果显示框
        result_text.delete('1.0', tk.END)
        # 将处理好的数据集输出到结果显示框
        result_text.insert(tk.END, result.to_string(index=False))
        # 设置文本样式和对齐方式
        result_text.tag_configure("center", justify="center")
        result_text.tag_add("center", "1.0", "end")
    
    # 输出排序结果
    def sort_button_click():
      
        result = main_part()
        indi = combo_sort_column.get()
        sorted_result = result.sort_values(by=indi, ascending=True)
        
        # 更新结果显示
        # 清空结果显示框
        result_text.delete('1.0', tk.END)
        # 将处理好的数据集输出到结果显示框
        result_text.insert(tk.END, sorted_result.to_string(index=False))
        # 设置文本样式和对齐方式
        result_text.tag_configure("center", justify="center")
        result_text.tag_add("center", "1.0", "end")
    
    # 创建主窗口
    root = tk.Tk()
    root.title("交通数据查询")
    
    # 设置窗口大小
    root.geometry('1280x720')
    
    # 标题
    title = tk.Label(root, text='欢迎查询台湾交通数据', font=('思源黑体 CN', 25, 'bold'), width=20, height=3)
    title.pack()
    
    # 子标题1
    subtitle1 = tk.Label(root, text='查询', font=('思源黑体 CN', 20, 'bold'), width=10, height=2)
    subtitle1.place(x=2, y=70)
    
    # # 创建多行文本框
    # text_widget = tk.Text(root, height=2, width=100)
    # text_widget.pack()
    # 
    # # 插入多行文本
    # text_widget.insert(tk.END, "全部公交:\n"+str(np.sort(traffic_data.loc[:,"VehicleType"].unique()))[1:-1]+"\n")
    # 
    # # 设置文本样式和对齐方式
    # text_widget.tag_configure("center", justify="center")
    # text_widget.tag_add("center", "1.0", "end")
    
    # 创建公交标签和输入框
    bus_label = tk.Label(root, text="全部车辆: 5,31,32,41,42", font=('思源黑体 CN', 12, 'bold'))
    bus_label.place(x=52, y=135)  
    
    bus_label2 = tk.Label(root, text="车辆类型(格式如:5,31,41):", font=('思源黑体 CN', 15, 'bold'))
    bus_label2.place(x=50, y=165)
    # bus_label.pack()
    
    bus_entry = tk.Entry(root, font=('思源黑体 CN', 15, 'bold'))
    bus_entry.place(x=55, y=205)
    # bus_entry.pack()
    
    # 创建起点站标签和输入框
    start_station_label = tk.Label(root, text="上车站:", font=('思源黑体 CN', 15, 'bold'))
    start_station_label.place(x=50, y=245)
    # start_station_label.pack()
    
    stationlist=np.sort(traffic_data.loc[:,"GantryID_O"].unique()).tolist()
    start_station_entry = AutocompleteCombobox(completevalues=stationlist, font=('思源黑体 CN', 15, 'bold'), width=15)
    # start_station_entry.insert(0, "上车站:")
    start_station_entry.place(x=140, y=248)
    # start_station_entry.pack()
    
    # 创建终点站标签和输入框
    end_station_label = tk.Label(root, text="目的地:", font=('思源黑体 CN', 15, 'bold'))
    end_station_label.place(x=50, y=285)
    # end_station_label.pack()
    
    stationlist=np.sort(traffic_data.loc[:,"GantryID_O"].unique()).tolist()
    end_station_entry = AutocompleteCombobox(completevalues=stationlist, font=('思源黑体 CN', 15, 'bold'), width=15)
    # end_station_entry.insert(0, "目的地:")
    end_station_entry.place(x=140, y=288)
    # end_station_entry.pack()
    
    # 创建时间点标签和输入框
    timepoint_label = tk.Label(root, text="到达起点站时间:", font=('思源黑体 CN', 15, 'bold'))
    timepoint_label.place(x=50, y=325)
    # timepoint_label.pack()
    
    start_time = "08:00"
    end_time = "09:00"
    interval = 5  # 间隔时间，单位为分钟
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
    
    timepoint_entry = AutocompleteCombobox(completevalues=time_list, font=('思源黑体 CN', 15, 'bold'))
    timepoint_entry.place(x=55, y=365)
    # timepoint_entry.pack()
    
    # 创建搜索按钮
    search_button = tk.Button(root, text="单击查询", command=search_button_click, font=('思源黑体 CN', 15, 'bold'))
    search_button.place(x=55, y=415)
    # search_button.pack()
    
    # 创建结果显示框
    result_text = tk.Text(root, height=25, width=72, font=('思源黑体 CN', 15, 'bold'))
    result_text.place(x=410, y=120)
    # result_text.pack()
    
    # 排序部分的布局
    subtitle2 = tk.Label(root, text='排序', font=('思源黑体 CN', 20, 'bold'), width=10, height=2)
    subtitle2.place(x=2, y=470)
    
    # 排序列标签和下拉框
    label_sort_column = tk.Label(root,text='排序列:', font=('思源黑体 CN', 15, 'bold'))
    label_sort_column.place(x=50, y=535)
    
    COLUMN_NAMES = ["RealTime","VehicleType","TripLength"]
    combo_sort_column = AutocompleteCombobox(completevalues=COLUMN_NAMES, font=('思源黑体 CN', 15, 'bold'), width=15)
    default_sort_column = COLUMN_NAMES[0]
    combo_sort_column.set(default_sort_column)
    combo_sort_column.place(x=140, y=538)    
    
    # 排序按钮
    sort_button = tk.Button(root, text="单击排序", command=sort_button_click, font=('思源黑体 CN', 15, 'bold'))
    sort_button.place(x=55, y=590)
    
    # 启动主循环
    root.mainloop()

link_start_ultra()





