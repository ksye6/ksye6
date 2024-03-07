# -*- coding: gbk -*-

import fastf1

from urllib.request import urlopen
import json

import pandas as pd

# Basic information

# year: 2024
# session_key: 9472
# session_name/session_type: Race
# meeting_key: 1229

# country_name: Bahrain
# country_key: 36
# country_code: BRN

# circuit_short_name/location: Sakhir
# circuit_key: 63

# team_name: Williams
# driver_full_name: Alexander ALBON; Logan SARGEANT
# driver_number: 23; 2

################################################################## OpenF1

# Car_data # 赛车信息，采样率约为 3.7 Hz (很多行，id为date)
responseCar_data1 = urlopen('https://api.openf1.org/v1/car_data?driver_number=23&session_key=9472') #可选参数速度 &speed>=315
ALBON_Car_data = json.loads(responseCar_data1.read().decode('utf-8'))
ALBON_Car_data = pd.DataFrame(ALBON_Car_data)
ALBON_Car_data.columns
ALBON_Car_data.loc[:,['date', 'rpm', 'speed','n_gear', 'throttle', 'drs', 'brake']]


# Laps # 圈数信息 (圈数行，id为lap_number)
responseLaps1 = urlopen('https://api.openf1.org/v1/laps?session_key=9472&driver_number=23') #可选参数圈数 &lap_number=8
ALBON_laps = json.loads(responseLaps1.read().decode('utf-8'))
ALBON_laps = pd.DataFrame(ALBON_laps)
ALBON_laps.columns
ALBON_laps.loc[:,['lap_number','date_start', 'lap_duration', 'is_pit_out_lap', 
                      'i1_speed', 'i2_speed','st_speed',
                      'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
                      'segments_sector_1', 'segments_sector_2', 'segments_sector_3']]

# responseLaps2 = urlopen('https://api.openf1.org/v1/laps?session_key=9472&driver_number=2')
# SARGEANT_laps = json.loads(responseLaps2.read().decode('utf-8'))
# SARGEANT_laps = pd.DataFrame(ALBON_laps)
# SARGEANT_laps.columns
# SARGEANT_laps

# Location 坐标 采样率约为 3.7 Hz (很多行，id为date)
responseLocation1 = urlopen('https://api.openf1.org/v1/location?session_key=9472&driver_number=23') # 可选参数时间 &date>2023-09-16T13:03:35.200&date<2023-09-16T13:03:35.800
ALBON_Location = json.loads(responseLocation1.read().decode('utf-8'))
ALBON_Location = pd.DataFrame(ALBON_Location)
ALBON_Location.columns
ALBON_Location.loc[:,['date', 'x', 'y', 'z']]

# Pit 维修区信息 仅包含进站行
responsePit1 = urlopen('https://api.openf1.org/v1/pit?session_key=9158&pit_duration<31')
ALBON_Pit = json.loads(responsePit1.read().decode('utf-8'))
ALBON_Pit = pd.DataFrame(ALBON_Pit)
ALBON_Pit.columns
ALBON_Pit.loc[:,['date','pit_duration', 'lap_number']]

# Stints 每套轮胎的使用时间

# Position 似乎是名次

# Intervals 获取车手之间的实时间隔数据以及他们与比赛领先者的差距, 每 4 秒更新一次。

# Weather 天气, 每分钟更新一次

# Sessions 特定时期（练习、排位赛、冲刺、正赛）

# Race control 

################################################################## FastF1

session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()
laps_alb = session.laps.pick_driver('ALB')
laps_alb.columns











