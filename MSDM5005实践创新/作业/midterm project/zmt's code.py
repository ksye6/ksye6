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
# responseCar_data1 = urlopen('https://api.openf1.org/v1/car_data?driver_number=23&session_key=9472') #可选参数速度 &speed>=315
# ALBON_Car_data = json.loads(responseCar_data1.read().decode('utf-8'))
# ALBON_Car_data = pd.DataFrame(ALBON_Car_data)
# ALBON_Car_data.columns
# ALBON_Car_data.loc[:,['date', 'rpm', 'speed','n_gear', 'throttle', 'drs', 'brake']]


# Laps # 圈数信息 (圈数行，id为lap_number)
responseLaps1 = urlopen('https://api.openf1.org/v1/laps?session_key=9472&driver_number=23') #可选参数圈数 &lap_number=8
ALBON_laps = json.loads(responseLaps1.read().decode('utf-8'))
ALBON_laps = pd.DataFrame(ALBON_laps)
ALBON_laps.columns
ALBON_laps['Compound'] = ["SOFT"]*15+["HARD"]*41
# ALBON_laps.loc[:,['lap_number','date_start', 'lap_duration', 'is_pit_out_lap', 
#                       'i1_speed', 'i2_speed','st_speed',
#                       'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
#                       'segments_sector_1', 'segments_sector_2', 'segments_sector_3']]
ALBON_laps.loc[:,['lap_duration','lap_number','is_pit_out_lap','Compound']]

responseLaps2 = urlopen('https://api.openf1.org/v1/laps?session_key=9472&driver_number=2')
SARGEANT_laps = json.loads(responseLaps2.read().decode('utf-8'))
SARGEANT_laps = pd.DataFrame(SARGEANT_laps)
SARGEANT_laps.columns
SARGEANT_laps['Compound'] = ["SOFT"]*10+["HARD"]*30+["SOFT"]*15
SARGEANT_laps.loc[:,['lap_duration','lap_number','is_pit_out_lap','Compound']]

# Location 坐标 采样率约为 3.7 Hz (很多行，id为date)
# responseLocation1 = urlopen('https://api.openf1.org/v1/location?session_key=9472&driver_number=23') # 可选参数时间 &date>2023-09-16T13:03:35.200&date<2023-09-16T13:03:35.800
# ALBON_Location = json.loads(responseLocation1.read().decode('utf-8'))
# ALBON_Location = pd.DataFrame(ALBON_Location)
# ALBON_Location.columns
# ALBON_Location.loc[:,['date', 'x', 'y', 'z']]

# Pit 维修区信息 仅包含进站行
# responsePit1 = urlopen('https://api.openf1.org/v1/pit?session_key=9158&driver_number=23')
# ALBON_Pit = json.loads(responsePit1.read().decode('utf-8'))
# ALBON_Pit = pd.DataFrame(ALBON_Pit)
# ALBON_Pit.columns
# ALBON_Pit.loc[:,['date','pit_duration', 'lap_number']]
# 
# responsePit2 = urlopen('https://api.openf1.org/v1/pit?session_key=9158&driver_number=2')
# SARGEANT_Pit2 = json.loads(responsePit2.read().decode('utf-8'))
# SARGEANT_Pit2 = pd.DataFrame(ALBON_Pit)
# SARGEANT_Pit2.columns
# SARGEANT_Pit2.loc[:,['date','pit_duration', 'lap_number']]

# Stints 每套轮胎的使用时间

# Position 似乎是名次

# Intervals 获取车手之间的实时间隔数据以及他们与比赛领先者的差距, 每 4 秒更新一次。

# Weather 天气, 每分钟更新一次

# Sessions 特定时期（练习、排位赛、冲刺、正赛）

# Race control 

################################################################## FastF1

##### Driver Laptimes Scatterplot (All Laps)
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import seaborn as sns

import fastf1
import fastf1.plotting

# race = fastf1.get_session(2024, 'Bahrain', 'R')
# race.load()
# 
# ALB_driver_laps = race.laps.pick_driver("ALB").pick_quicklaps().reset_index()
# SAR_driver_laps = race.laps.pick_driver("SAR").pick_quicklaps().reset_index()
# 
# ALB_driver_laps[ALB_driver_laps['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
# ALB_driver_laps.loc[:,['Stint','LapNumber','PitOutTime', 'PitInTime']] # MISS LapNumber 16,37
# 
# SAR_driver_laps[SAR_driver_laps['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
# SAR_driver_laps.loc[:,['Stint','LapNumber','PitOutTime', 'PitInTime']] # MISS LapNumber 10,11,14,16,29,41

# {'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'}

##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=ALBON_laps,
                x="lap_number",
                y="lap_duration",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=ALBON_laps,
             x="lap_number",
             y="lap_duration",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

# 标注横坐标为16的点
x1 = 16
y1 = float(ALBON_laps.loc[ALBON_laps["lap_number"] == x1, "lap_duration"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 1), ha='center', color='#1471e3',
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# 标注横坐标为37的点
x2 = 37
y2 = float(ALBON_laps.loc[ALBON_laps["lap_number"] == x2, "lap_duration"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 + 1), ha='center', color='#1471e3',
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Bahrain Race (ALB)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
##

##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=SARGEANT_laps,
                x="lap_number",
                y="lap_duration",
                ax=ax,
                hue="Compound",
                marker="s",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto',
                alpha=1)

sns.lineplot(data=SARGEANT_laps.iloc[:10,:],
             x="lap_number",
             y="lap_duration",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=SARGEANT_laps.iloc[10:40,:],
             x="lap_number",
             y="lap_duration",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=SARGEANT_laps.iloc[40:,:],
             x="lap_number",
             y="lap_duration",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

special = SARGEANT_laps.iloc[28:40,:]
special["Compound"] = ['HARD(Old tire)']*12
sns.lineplot(data=special,
             x="lap_number",
             y="lap_duration",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD(Old tire)': '#13226b', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=3,
             legend='auto')
             
# 标注横坐标为11的点
x1 = 11
y1 = float(SARGEANT_laps.loc[SARGEANT_laps["lap_number"] == x1, "lap_duration"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 3), ha='center', color='#1471e3',
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# 标注横坐标为29的点
x2 = 29
y2 = float(SARGEANT_laps.loc[SARGEANT_laps["lap_number"] == x2, "lap_duration"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 + 6), ha='center', color='#1471e3',
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# 标注横坐标为41的点
x3 = 41
y3 = float(SARGEANT_laps.loc[SARGEANT_laps["lap_number"] == x3, "lap_duration"].values[0])
ax.annotate(f'Pit 3 : Lap {x3}', xy=(x3, y3), xytext=(x3, y3 + 2.5), ha='center', color='#da291c',
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Bahrain Race (SAR)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
##

##### Driver Laptimes Distribution Visualization (Fastest Lap)

f1_24_bahrain_r = fastf1.get_session(2024, 'Bahrain', 'R')
f1_24_bahrain_r.load()

laps_alb = f1_24_bahrain_r.laps.pick_driver('ALB')
laps_sar = f1_24_bahrain_r.laps.pick_driver('SAR')
laps_alb_sar = f1_24_bahrain_r.laps.pick_drivers(['ALB', 'SAR'])

driver_laps = laps_alb_sar.pick_quicklaps()
driver_laps = driver_laps.reset_index()

##
# create the figure
fig, ax = plt.subplots(figsize=(10, 10))

driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

sns.violinplot(data=driver_laps,
               x="Driver",
               y="LapTime(s)",
               hue="Driver",
               inner=None,
               density_norm="area",
               palette={'ALB': '#f7f568', 'SAR': '#4a1212'},
               alpha=0.9
               )

sns.swarmplot(data=driver_laps,
              x="Driver",
              y="LapTime(s)",
              hue="Compound",
              palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
              hue_order=["SOFT", "MEDIUM", "HARD"],
              linewidth=0,
              size=6,
              )

ax.set_xlabel("Driver")
ax.set_ylabel("Lap Time (s)")
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.8)
plt.suptitle("2024 Bahrain Race Time Distributions",y = 0.99)
# sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
##


##### Gear shifts on track (Fastest Lap)

from matplotlib import colormaps
from matplotlib.collections import LineCollection

fastf1.plotting.setup_mpl()

# albon
laps_alb_fastest = laps_alb.pick_fastest()
alb_fastest_car_data = laps_alb_fastest.get_car_data()
alb_fastest_telemetry = laps_alb_fastest.get_telemetry()

# sargeant
laps_sar_fastest = laps_sar.pick_fastest()
sar_fastest_car_data = laps_sar_fastest.get_car_data()
sar_fastest_telemetry = laps_sar_fastest.get_telemetry()

laps_alb_sar_fastest_telemetry = laps_alb_sar.pick_fastest().get_telemetry()

# laps_alb_sar (actually SAR the fastest lap)
x = np.array(laps_alb_sar_fastest_telemetry['X'].values)
y = np.array(laps_alb_sar_fastest_telemetry['Y'].values)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
gear = laps_alb_sar_fastest_telemetry['nGear'].to_numpy().astype(float)

cmap = colormaps['Paired']
lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
lc_comp.set_array(gear)
lc_comp.set_linewidth(4)

##
fig = plt.figure(figsize=(10,10), dpi=300)
plt.gca().add_collection(lc_comp)
plt.axis('equal')
plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

title = plt.suptitle(
    f"Fastest Lap Gear Shift Visualization\n"
    f"{laps_alb_sar.pick_fastest()['Driver']} - {f1_24_bahrain_r.event['EventName']} {f1_24_bahrain_r.event.year}"
)
cbar = plt.colorbar(mappable=lc_comp, label="Gear",
                    boundaries=np.arange(1, 10))
cbar.set_ticks(np.arange(1.5, 9.5))
cbar.set_ticklabels(np.arange(1, 9))

plt.show()
##

##### Speed visualization on track map (Fastest Lap)

import matplotlib as mpl

year = 2024
wknd = 3
ses = 'R'
colormap = mpl.cm.plasma

##
# albon
driver1 = 'ALB'

# Get telemetry data
x = alb_fastest_telemetry['X']              # values for x-axis
y = alb_fastest_telemetry['Y']              # values for y-axis
color = alb_fastest_telemetry['Speed']      # value to base color gradient on

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# We create a plot with title and adjust some setting to make it look good.
fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(10, 10))
fig.suptitle(f'{wknd} {year} - {driver1} - Speed', size=24, y=0.97)

# Adjust margins and turn of axis
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
ax.axis('off')

# After this, we plot the data itself.
# Create background track line
ax.plot(x, y, color='black', linestyle='-', linewidth=12, zorder=0)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(color.min(), color.max())
lc = LineCollection(segments, cmap=colormap, norm=norm,linestyle='-', linewidth=5)

# Set the values used for colormapping
lc.set_array(color)

# Merge all line segments together
line = ax.add_collection(lc)

# Finally, we create a color bar as a legend.
cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05],visible=False)
normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
legend = mpl.colorbar.ColorbarBase(fig.add_axes([0.25, 0.05, 0.5, 0.05]), norm=normlegend, cmap=colormap,orientation="horizontal")

# Show the plot
plt.show()
##

##
# sargeant
driver2 = 'SAR'

# Get telemetry data
x = sar_fastest_telemetry['X']              # values for x-axis
y = sar_fastest_telemetry['Y']              # values for y-axis
color = sar_fastest_telemetry['Speed']      # value to base color gradient on

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# We create a plot with title and adjust some setting to make it look good.
fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(10,10))
fig.suptitle(f'{wknd} {year} - {driver2} - Speed', size=24, y=0.97)

# Adjust margins and turn of axis
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
ax.axis('off')

# After this, we plot the data itself.
# Create background track line
ax.plot(x, y, color='black', linestyle='-', linewidth=12, zorder=0)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(color.min(), color.max())
lc = LineCollection(segments, cmap=colormap, norm=norm,linestyle='-', linewidth=5)

# Set the values used for colormapping
lc.set_array(color)

# Merge all line segments together
line = ax.add_collection(lc)

# Finally, we create a color bar as a legend.
cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05],visible=False)
normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
legend = mpl.colorbar.ColorbarBase(fig.add_axes([0.25, 0.05, 0.5, 0.05]), norm=normlegend, cmap=colormap,orientation="horizontal")

# Show the plot
plt.show()
##


##### Position changes during a race

fastf1.plotting.setup_mpl(misc_mpl_mods=False)

##
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

for drv in f1_24_bahrain_r.drivers:
    drv_laps = f1_24_bahrain_r.laps.pick_driver(drv)

    abb = drv_laps['Driver'].iloc[0]
    color = fastf1.plotting.driver_color(abb)
    if not drv in ['23','2']:
        ax.plot(drv_laps['LapNumber'], drv_laps['Position'], label=abb, color=color,alpha = 0.25)
    else:
        ax.plot(drv_laps['LapNumber'], drv_laps['Position'], color='white', linewidth=4.5)
        ax.plot(drv_laps['LapNumber'], drv_laps['Position'], label=abb, color=color, linewidth=3)

ax.set_ylim([20.5, 0.5])
ax.set_yticks([1, 5, 10, 15, 20])
ax.set_xlabel('Lap')
ax.set_ylabel('Position')

ax.legend(bbox_to_anchor=(1.0, 1.02))
plt.tight_layout()
plt.suptitle("Position changes during the race", y=0.999)

plt.show()
##


##### Distance to driver ahead during the race

AVG_DistanceToDriverAhead_Bylap = []
for i in range(len(laps_alb)):
    AVG_DistanceToDriverAhead_Bylap.append(np.average(laps_alb.pick_lap(i+1).get_telemetry()['DistanceToDriverAhead']))
laps_alb['AVG_DistanceToDriverAhead_Bylap'] = AVG_DistanceToDriverAhead_Bylap


AVG_DistanceToDriverAhead_Bylap2 = []
for i in range(len(laps_sar)):
    AVG_DistanceToDriverAhead_Bylap2.append(np.average(laps_sar.pick_lap(i+1).get_telemetry()['DistanceToDriverAhead']))
laps_sar['AVG_DistanceToDriverAhead_Bylap'] = AVG_DistanceToDriverAhead_Bylap2

plt.rcdefaults()

# alb
##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

ax.plot(laps_alb['LapNumber'],laps_alb['AVG_DistanceToDriverAhead_Bylap'],color='blue',linewidth=1,linestyle='-', marker='o')

ax.set_xlabel("Full Race Time")
ax.set_ylabel("Distance To Driver Ahead")

plt.suptitle("Distance to driver ahead during the race (ALB)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
##

# sar
##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

ax.plot(laps_sar['LapNumber'],laps_sar['AVG_DistanceToDriverAhead_Bylap'],color='blue',linewidth=1,linestyle='-', marker='o')

ax.set_xlabel("Full Race Time")
ax.set_ylabel("Distance To Driver Ahead")

plt.suptitle("Distance to driver ahead during the race (SAR)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
##
















