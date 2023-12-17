# -*- coding: gbk -*-
import csv
import time
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import datetime

## ���Դ˿�
#################################################### ����+˼· #########################################

traffic_data = pd.read_csv("C:\\Users\\�����\\Desktop\\ѧҵ\\�ۿƴ�\\MSDM5051�������python���ݽṹ\\��ҵ\\hw2\\TDCS_M06A_20190830_080000.csv")

# ������г�����ȷ������������������ȫ�ѡ�������лᾭ���ĳ�վ���������������վ���յ�վ������Ҫ�����ʱ����������/����ĳ��ʱ���ĳ�����Ϣ

# ���г�
set(traffic_data.loc[:,"VehicleType"].unique())
# ɸѡĳЩ����
busnameset=set()
busnameset.update([5,31,41])
filtered1=traffic_data[traffic_data.loc[:,"VehicleType"].isin(busnameset)]

# ���г�վ
set(traffic_data.loc[:,"GantryID_O"].unique())
# set(traffic_data.loc[:,"GantryID_D"].unique()) һ����
# ɸѡ��ǰ��վ��Ŀ�ĵس�վ��
start_stationname="01F0005S"
filtered2=filtered1[filtered1.loc[:,"TripInformation"].str.contains(start_stationname)]
end_stationname="01H0447S"
filtered2=filtered2[filtered2.loc[:,"TripInformation"].str.contains(end_stationname)]

# ɸѡʱ���
# ��ȡ"information"�ﵱǰ��վ��֮ǰ���ַ���
pattern = f"(.{{8}})\+"+start_stationname
extracted_strings = filtered2.loc[:,"TripInformation"].str.extract(pattern, expand=False)
# ����ȡ���ַ����洢Ϊ����,����
sorted_object = filtered2.copy()
sorted_object["RealTime"] = extracted_strings
sorted_object["Start"] = start_stationname
sorted_object["End"] = end_stationname
sorted_object = sorted_object.sort_values(by=["RealTime","VehicleType"],ascending=True)

# ĳ��ʱ���֮��ĳ���
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

traffic_data = pd.read_csv("C:\\Users\\�����\\Desktop\\ѧҵ\\�ۿƴ�\\MSDM5051�������python���ݽṹ\\��ҵ\\hw2\\TDCS_M06A_20190830_080000.csv")

# ���Դ˺���(����Ʒ)
def test():

    def search_button_click():
        # �������ʼ��
        if timepoint_entry.get() == "":
            timepoint_entry.insert(tk.END, "08:00")
        if bus_entry.get() == "":
            bus_entry.insert(tk.END, "5,31,32,41,42")

        # ��ȡ�û���������վ���յ�վ
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

        # ���½����ʾ
        # ��ս����ʾ��
        result_text.delete('1.0', tk.END)
        # ������õ����ݼ�����������ʾ��
        result_text.insert(tk.END, result.to_string(index=False))
        # �����ı���ʽ�Ͷ��뷽ʽ
        result_text.tag_configure("center", justify="center")
        result_text.tag_add("center", "1.0", "end")

    # ����������
    root = tk.Tk()
    root.title("��ͨ���ݲ�ѯ")

    # ���ô��ڴ�С
    root.geometry("1080x960")

    # # ����label
    # additional_text = ttk.Label(root, text="All buses: "+str(traffic_data.loc[:,"VehicleType"].unique())[1:-1], anchor="n")
    # additional_text.pack()

    # ���������ı���
    text_widget = tk.Text(root, height=30, width=100)
    text_widget.pack()

    # ��������ı�
    text_widget.insert(tk.END, "ȫ������:\n"+str(np.sort(traffic_data.loc[:,"VehicleType"].unique()))[1:-1]+"\n")
    text_widget.insert(tk.END, "ȫ����վ:\n "+str(np.sort(traffic_data.loc[:,"GantryID_O"].unique()))[1:-1].replace("'", ""))

    # �����ı���ʽ�Ͷ��뷽ʽ
    text_widget.tag_configure("center", justify="center")
    text_widget.tag_add("center", "1.0", "end")

    # ����������ǩ�������
    bus_label = tk.Label(root, text="���蹫��(��ʽ��:5,31,41):")
    bus_label.pack()
    bus_entry = tk.Entry(root)
    bus_entry.pack()

    # �������վ��ǩ�������
    start_station_label = ttk.Label(root, text="���վ(��ʽ��:01F0005S):")
    start_station_label.pack()
    start_station_entry = ttk.Entry(root)
    start_station_entry.pack()

    # �����յ�վ��ǩ�������
    end_station_label = ttk.Label(root, text="�յ�վ(��ʽ��:01H0447S):")
    end_station_label.pack()
    end_station_entry = ttk.Entry(root)
    end_station_entry.pack()

    # ����ʱ����ǩ�������
    timepoint_label = ttk.Label(root, text="�������վʱ��(����8-9��,��ʽ��:08:30):")
    timepoint_label.pack()
    timepoint_entry = ttk.Entry(root)
    timepoint_entry.pack()

    # ����������ť
    search_button = ttk.Button(root, text="��������", command=search_button_click)
    search_button.pack()

    # ���������ʾ��
    result_text = tk.Text(root, height=300, width=200)
    result_text.pack()

    # ������ѭ��
    root.mainloop()

test()

##########################################################################################################
################################################  GUI  ###################################################
##########################################################################################################
# ֻ��Ҫ import �� �������ݼ� ���ɣ���õ������� ˼Դ����CN bold ������

# ȫ�°汾
def link_start_ultra():
    
    # �����������һ�����ݼ�
    def main_part():
        # �������ʼ��
        if timepoint_entry.get() == "":
            timepoint_entry.insert(tk.END, "08:00")
        if bus_entry.get() == "":
            bus_entry.insert(tk.END, "5,31,32,41,42")
        
        # ��ȡ�û���������վ���յ�վ
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
    
    # �����ѯ���
    def search_button_click():
        result = main_part()
        
        # ���½����ʾ
        # ��ս����ʾ��
        result_text.delete('1.0', tk.END)
        # ������õ����ݼ�����������ʾ��
        result_text.insert(tk.END, result.to_string(index=False))
        # �����ı���ʽ�Ͷ��뷽ʽ
        result_text.tag_configure("center", justify="center")
        result_text.tag_add("center", "1.0", "end")
    
    # ���������
    def sort_button_click():
      
        result = main_part()
        indi = combo_sort_column.get()
        sorted_result = result.sort_values(by=indi, ascending=True)
        
        # ���½����ʾ
        # ��ս����ʾ��
        result_text.delete('1.0', tk.END)
        # ������õ����ݼ�����������ʾ��
        result_text.insert(tk.END, sorted_result.to_string(index=False))
        # �����ı���ʽ�Ͷ��뷽ʽ
        result_text.tag_configure("center", justify="center")
        result_text.tag_add("center", "1.0", "end")
    
    # ����������
    root = tk.Tk()
    root.title("��ͨ���ݲ�ѯ")
    
    # ���ô��ڴ�С
    root.geometry('1280x720')
    
    # ����
    title = tk.Label(root, text='��ӭ��ѯ̨�彻ͨ����', font=('˼Դ���� CN', 25, 'bold'), width=20, height=3)
    title.pack()
    
    # �ӱ���1
    subtitle1 = tk.Label(root, text='��ѯ', font=('˼Դ���� CN', 20, 'bold'), width=10, height=2)
    subtitle1.place(x=2, y=70)
    
    # # ���������ı���
    # text_widget = tk.Text(root, height=2, width=100)
    # text_widget.pack()
    # 
    # # ��������ı�
    # text_widget.insert(tk.END, "ȫ������:\n"+str(np.sort(traffic_data.loc[:,"VehicleType"].unique()))[1:-1]+"\n")
    # 
    # # �����ı���ʽ�Ͷ��뷽ʽ
    # text_widget.tag_configure("center", justify="center")
    # text_widget.tag_add("center", "1.0", "end")
    
    # ����������ǩ�������
    bus_label = tk.Label(root, text="ȫ������: 5,31,32,41,42", font=('˼Դ���� CN', 12, 'bold'))
    bus_label.place(x=52, y=135)  
    
    bus_label2 = tk.Label(root, text="��������(��ʽ��:5,31,41):", font=('˼Դ���� CN', 15, 'bold'))
    bus_label2.place(x=50, y=165)
    # bus_label.pack()
    
    bus_entry = tk.Entry(root, font=('˼Դ���� CN', 15, 'bold'))
    bus_entry.place(x=55, y=205)
    # bus_entry.pack()
    
    # �������վ��ǩ�������
    start_station_label = tk.Label(root, text="�ϳ�վ:", font=('˼Դ���� CN', 15, 'bold'))
    start_station_label.place(x=50, y=245)
    # start_station_label.pack()
    
    stationlist=np.sort(traffic_data.loc[:,"GantryID_O"].unique()).tolist()
    start_station_entry = AutocompleteCombobox(completevalues=stationlist, font=('˼Դ���� CN', 15, 'bold'), width=15)
    # start_station_entry.insert(0, "�ϳ�վ:")
    start_station_entry.place(x=140, y=248)
    # start_station_entry.pack()
    
    # �����յ�վ��ǩ�������
    end_station_label = tk.Label(root, text="Ŀ�ĵ�:", font=('˼Դ���� CN', 15, 'bold'))
    end_station_label.place(x=50, y=285)
    # end_station_label.pack()
    
    stationlist=np.sort(traffic_data.loc[:,"GantryID_O"].unique()).tolist()
    end_station_entry = AutocompleteCombobox(completevalues=stationlist, font=('˼Դ���� CN', 15, 'bold'), width=15)
    # end_station_entry.insert(0, "Ŀ�ĵ�:")
    end_station_entry.place(x=140, y=288)
    # end_station_entry.pack()
    
    # ����ʱ����ǩ�������
    timepoint_label = tk.Label(root, text="�������վʱ��:", font=('˼Դ���� CN', 15, 'bold'))
    timepoint_label.place(x=50, y=325)
    # timepoint_label.pack()
    
    start_time = "08:00"
    end_time = "09:00"
    interval = 5  # ���ʱ�䣬��λΪ����
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
    
    timepoint_entry = AutocompleteCombobox(completevalues=time_list, font=('˼Դ���� CN', 15, 'bold'))
    timepoint_entry.place(x=55, y=365)
    # timepoint_entry.pack()
    
    # ����������ť
    search_button = tk.Button(root, text="������ѯ", command=search_button_click, font=('˼Դ���� CN', 15, 'bold'))
    search_button.place(x=55, y=415)
    # search_button.pack()
    
    # ���������ʾ��
    result_text = tk.Text(root, height=25, width=72, font=('˼Դ���� CN', 15, 'bold'))
    result_text.place(x=410, y=120)
    # result_text.pack()
    
    # ���򲿷ֵĲ���
    subtitle2 = tk.Label(root, text='����', font=('˼Դ���� CN', 20, 'bold'), width=10, height=2)
    subtitle2.place(x=2, y=470)
    
    # �����б�ǩ��������
    label_sort_column = tk.Label(root,text='������:', font=('˼Դ���� CN', 15, 'bold'))
    label_sort_column.place(x=50, y=535)
    
    COLUMN_NAMES = ["RealTime","VehicleType","TripLength"]
    combo_sort_column = AutocompleteCombobox(completevalues=COLUMN_NAMES, font=('˼Դ���� CN', 15, 'bold'), width=15)
    default_sort_column = COLUMN_NAMES[0]
    combo_sort_column.set(default_sort_column)
    combo_sort_column.place(x=140, y=538)    
    
    # ����ť
    sort_button = tk.Button(root, text="��������", command=sort_button_click, font=('˼Դ���� CN', 15, 'bold'))
    sort_button.place(x=55, y=590)
    
    # ������ѭ��
    root.mainloop()

link_start_ultra()





