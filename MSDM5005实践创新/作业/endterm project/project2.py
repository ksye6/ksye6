import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import fastf1
import fastf1.plotting

from matplotlib.ticker import FuncFormatter
# 定义格式化函数
def format_y_axis(value, _):
    return f"{value/1e9:.0f}"

################################################################################################# 2022 Australia
race_A_2022 = fastf1.get_session(2022, 'Australia', 'R')
race_A_2022.load()

race_A_2022.laps.drop_duplicates(subset='Driver') # ALB, MAG 完赛; SAR, HUL 未参与


ALB_driver_laps_A_2022 = race_A_2022.laps.pick_driver("ALB").reset_index()
MAG_driver_laps_A_2022 = race_A_2022.laps.pick_driver("MAG").reset_index()

ALB_driver_laps_A_2022[ALB_driver_laps_A_2022['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
ALB_driver_laps_A_2022.loc[:,['Compound','LapNumber','Time','LapTime']] # 正常

MAG_driver_laps_A_2022[MAG_driver_laps_A_2022['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_A_2022.loc[:,['Compound','LapNumber','Time','LapTime']] # 缺1
for i in reversed(range(1,len(MAG_driver_laps_A_2022))):
    MAG_driver_laps_A_2022.loc[i,'LapTime'] = MAG_driver_laps_A_2022.loc[i,'Time']-MAG_driver_laps_A_2022.loc[i-1,'Time'] # 正常


palette = {'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'}

## ALB 10
ALB_A_2022_r = int(ALB_driver_laps_A_2022.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=ALB_driver_laps_A_2022,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=ALB_driver_laps_A_2022,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(ALB_driver_laps_A_2022[ALB_driver_laps_A_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(ALB_driver_laps_A_2022.loc[ALB_driver_laps_A_2022["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 0.05*1e11), ha='center', color= palette[ALB_driver_laps_A_2022[ALB_driver_laps_A_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2022 Australia Race (ALB) - No."+str(ALB_A_2022_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

plt.subplots_adjust(right=0.18)
plt.tight_layout()
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.show()
##

## MAG 14
MAG_A_2022_r = int(MAG_driver_laps_A_2022.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_A_2022,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=MAG_driver_laps_A_2022,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(MAG_driver_laps_A_2022[MAG_driver_laps_A_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(MAG_driver_laps_A_2022.loc[MAG_driver_laps_A_2022["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 0.05*1e11), ha='center', color= palette[MAG_driver_laps_A_2022[MAG_driver_laps_A_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2022 Australia Race (MAG) - No."+str(MAG_A_2022_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

plt.tight_layout()
plt.show()
##

################################################################################################# 2022 Japan
race_J_2022 = fastf1.get_session(2022, 'Japan', 'R')
race_J_2022.load()

race_J_2022.laps.drop_duplicates(subset='Driver') # MAG 完赛; ALB 未完赛; SAR, HUL 未参与

ALB_driver_laps_J_2022 = race_J_2022.laps.pick_driver("ALB").reset_index() # 1圈
MAG_driver_laps_J_2022 = race_J_2022.laps.pick_driver("MAG").reset_index() # 28圈

MAG_driver_laps_J_2022[MAG_driver_laps_J_2022['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_J_2022.loc[:,['Compound','LapNumber','Time','LapTime']] # 缺4
for i in reversed(range(1,len(MAG_driver_laps_J_2022))):
    MAG_driver_laps_J_2022.loc[i,'LapTime'] = MAG_driver_laps_J_2022.loc[i,'Time']-MAG_driver_laps_J_2022.loc[i-1,'Time']


## 浪费很多时间
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_J_2022,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2022 Japan Race (MAG)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 3 # MAG_driver_laps_J_2022["LapTime"]
y = MAG_driver_laps_J_2022["LapTime"].loc[2].value
ax.annotate('Waiting for weather', (x+0.5, y), xytext=(x + 2, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

## 去除特异点 MAG 14
MAG_J_2022_r = int(MAG_driver_laps_J_2022.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_J_2022.drop(2),
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=MAG_driver_laps_J_2022.drop(2).iloc[:2,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=MAG_driver_laps_J_2022.drop(2).loc[3:6,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=MAG_driver_laps_J_2022.drop(2).loc[6:,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(MAG_driver_laps_J_2022[MAG_driver_laps_J_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(MAG_driver_laps_J_2022.loc[MAG_driver_laps_J_2022["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 - 0.07*1e11), ha='center', color= palette[MAG_driver_laps_J_2022[MAG_driver_laps_J_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x2 = int(MAG_driver_laps_J_2022[MAG_driver_laps_J_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y2 = float(MAG_driver_laps_J_2022.loc[MAG_driver_laps_J_2022["LapNumber"] == x2, "LapTime"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 + 0.08*1e11), ha='center', color= palette[MAG_driver_laps_J_2022[MAG_driver_laps_J_2022['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2022 Japan Race (MAG) - No."+str(MAG_J_2022_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

plt.tight_layout()
plt.show()
##

################################################################################################# 2023 Australia
race_A_2023 = fastf1.get_session(2023, 'Australia', 'R')
race_A_2023.load()

race_A_2023.laps.drop_duplicates(subset='Driver') # HUL 完赛; ALB, SAR, MAG  未完赛

ALB_driver_laps_A_2023 = race_A_2023.laps.pick_driver("ALB").reset_index() # 7圈 不分析
MAG_driver_laps_A_2023 = race_A_2023.laps.pick_driver("MAG").reset_index() # 53圈
SAR_driver_laps_A_2023 = race_A_2023.laps.pick_driver("SAR").reset_index() # 57圈
HUL_driver_laps_A_2023 = race_A_2023.laps.pick_driver("HUL").reset_index() # 58圈

MAG_driver_laps_A_2023[MAG_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_A_2023.loc[:,['Compound','LapNumber','Time','LapTime']] # 缺2
for i in reversed(range(1,len(MAG_driver_laps_A_2023))):
    MAG_driver_laps_A_2023.loc[i,'LapTime'] = MAG_driver_laps_A_2023.loc[i,'Time']-MAG_driver_laps_A_2023.loc[i-1,'Time']

## MAG
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_A_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Australia Race (MAG)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 8
y = MAG_driver_laps_A_2023["LapTime"].loc[7].value
ax.annotate('Cleaning due to crash', (x+0.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")
x2 = 52
y2 = MAG_driver_laps_A_2023["LapTime"].loc[52].value
ax.annotate('Tire accident', (x2, y2), xytext=(x2 - 20, y2+10*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

SAR_driver_laps_A_2023[SAR_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
SAR_driver_laps_A_2023.loc[:,['Compound','LapNumber','Time','LapTime']] # 缺5
for i in reversed(range(1,len(SAR_driver_laps_A_2023))):
    SAR_driver_laps_A_2023.loc[i,'LapTime'] = SAR_driver_laps_A_2023.loc[i,'Time']-SAR_driver_laps_A_2023.loc[i-1,'Time']

##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=SAR_driver_laps_A_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Australia Race (SAR)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 8
y = SAR_driver_laps_A_2023["LapTime"].loc[7].value
ax.annotate('Cleaning due to crash', (x+0.5, y), xytext=(x + 5, y-10*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

HUL_driver_laps_A_2023[HUL_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
HUL_driver_laps_A_2023.loc[:,['Compound','LapNumber','Time','LapTime']] # 缺5
for i in reversed(range(1,len(HUL_driver_laps_A_2023))):
    HUL_driver_laps_A_2023.loc[i,'LapTime'] = HUL_driver_laps_A_2023.loc[i,'Time']-HUL_driver_laps_A_2023.loc[i-1,'Time']

## HUL 7
HUL_A_2023_r = int(HUL_driver_laps_A_2023.Position.iloc[-1])-1

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=HUL_driver_laps_A_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=HUL_driver_laps_A_2023.iloc[:8,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_A_2023.iloc[8:55,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_A_2023.loc[55:,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(HUL_driver_laps_A_2023[HUL_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(HUL_driver_laps_A_2023.loc[HUL_driver_laps_A_2023["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 - 0.2*1e12), ha='center', color= palette[HUL_driver_laps_A_2023[HUL_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x2 = int(HUL_driver_laps_A_2023[HUL_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y2 = float(HUL_driver_laps_A_2023.loc[HUL_driver_laps_A_2023["LapNumber"] == x2, "LapTime"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 - 0.2*1e12), ha='center', color= palette[HUL_driver_laps_A_2023[HUL_driver_laps_A_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Australia Race (HUL) - No."+str(HUL_A_2023_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 56
y = HUL_driver_laps_A_2023["LapTime"].loc[56].value
ax.annotate('Field accident', (x-0.5, y), xytext=(x - 25, y-50*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

################################################################################################# 2023 Japan
race_J_2023 = fastf1.get_session(2023, 'Japan', 'R')
race_J_2023.load()

race_J_2023.laps.drop_duplicates(subset='Driver') # HUL, MAG 完赛; ALB, SAR  未完赛

ALB_driver_laps_J_2023 = race_J_2023.laps.pick_driver("ALB").reset_index() # 26圈
MAG_driver_laps_J_2023 = race_J_2023.laps.pick_driver("MAG").reset_index() # 52圈
SAR_driver_laps_J_2023 = race_J_2023.laps.pick_driver("SAR").reset_index() # 22圈
HUL_driver_laps_J_2023 = race_J_2023.laps.pick_driver("HUL").reset_index() # 52圈

SAR_driver_laps_J_2023[SAR_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
SAR_driver_laps_J_2023.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(SAR_driver_laps_J_2023))):
    SAR_driver_laps_J_2023.loc[i,'LapTime'] = SAR_driver_laps_J_2023.loc[i,'Time']-SAR_driver_laps_J_2023.loc[i-1,'Time']


##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data= SAR_driver_laps_J_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Japan Race (SAR)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 5
y = 155*1e9
ax.annotate('Crash', (x+0.5, y), xytext=(x + 2, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

ALB_driver_laps_J_2023[ALB_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
ALB_driver_laps_J_2023.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(ALB_driver_laps_J_2023))):
    ALB_driver_laps_J_2023.loc[i,'LapTime'] = ALB_driver_laps_J_2023.loc[i,'Time']-ALB_driver_laps_J_2023.loc[i-1,'Time']


##
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data= ALB_driver_laps_J_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Japan Race (ALB)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 3
y = 150*1e9
ax.annotate('Crash', (x+0.5, y), xytext=(x + 2, y-4*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

MAG_driver_laps_J_2023[MAG_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_J_2023.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(MAG_driver_laps_J_2023))):
    MAG_driver_laps_J_2023.loc[i,'LapTime'] = MAG_driver_laps_J_2023.loc[i,'Time']-MAG_driver_laps_J_2023.loc[i-1,'Time']

## MAG 15
MAG_J_2023_r = int(MAG_driver_laps_J_2023.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_J_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=MAG_driver_laps_J_2023,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(MAG_driver_laps_J_2023[MAG_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(MAG_driver_laps_J_2023.loc[MAG_driver_laps_J_2023["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 - 0.07*1e11), ha='center', color= palette[MAG_driver_laps_J_2023[MAG_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Japan Race (MAG) - No."+str(MAG_J_2023_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

x = 3
y = MAG_driver_laps_J_2023["LapTime"].loc[3].value
ax.annotate('Cleaning due to crash', (x+1.5, y), xytext=(x + 6, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

plt.tight_layout()
plt.show()
##

HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
HUL_driver_laps_J_2023.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(HUL_driver_laps_J_2023))):
    HUL_driver_laps_J_2023.loc[i,'LapTime'] = HUL_driver_laps_J_2023.loc[i,'Time']-HUL_driver_laps_J_2023.loc[i-1,'Time']

## HUL 14
HUL_J_2023_r = int(HUL_driver_laps_J_2023.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=HUL_driver_laps_J_2023,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=HUL_driver_laps_J_2023.loc[:8,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_J_2023.loc[8:21,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_J_2023.loc[21:37,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_J_2023.loc[37:,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(HUL_driver_laps_J_2023.loc[HUL_driver_laps_J_2023["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 0.04*1e11), ha='center', color= palette[HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=10), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x2 = int(HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y2 = float(HUL_driver_laps_J_2023.loc[HUL_driver_laps_J_2023["LapNumber"] == x2, "LapTime"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 + 0.04*1e11), ha='center', color= palette[HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=10), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x3 = int(HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[2,0])
y3 = float(HUL_driver_laps_J_2023.loc[HUL_driver_laps_J_2023["LapNumber"] == x3, "LapTime"].values[0])
ax.annotate(f'Pit 3 : Lap {x3}', xy=(x3, y3), xytext=(x3, y3 + 0.04*1e11), ha='center', color= palette[HUL_driver_laps_J_2023[HUL_driver_laps_J_2023['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[2,1]],
fontproperties=FontProperties(weight='bold', size=10), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

x = 3
y = HUL_driver_laps_J_2023["LapTime"].loc[3].value
ax.annotate('Cleaning due to crash', (x+1.5, y), xytext=(x + 6, y-1*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2023 Japan Race (HUL) - No."+str(HUL_J_2023_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
##

################################################################################################# 2024 Australia
race_A_2024 = fastf1.get_session(2024, 'Australia', 'R')
race_A_2024.load()

race_A_2024.laps.drop_duplicates(subset='Driver') #  ALB, HUL, MAG完赛; SAR 未参赛

ALB_driver_laps_A_2024 = race_A_2024.laps.pick_driver("ALB").reset_index() # 57圈
MAG_driver_laps_A_2024 = race_A_2024.laps.pick_driver("MAG").reset_index() # 57圈
HUL_driver_laps_A_2024 = race_A_2024.laps.pick_driver("HUL").reset_index() # 58圈

ALB_driver_laps_A_2024[ALB_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
ALB_driver_laps_A_2024.loc[:,['Compound','LapNumber','Time','LapTime']]

## ALB 11
ALB_A_2024_r = int(ALB_driver_laps_A_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=ALB_driver_laps_A_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=ALB_driver_laps_A_2024,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(ALB_driver_laps_A_2024[ALB_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(ALB_driver_laps_A_2024.loc[ALB_driver_laps_A_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 0.02*1e11), ha='center', color= palette[ALB_driver_laps_A_2024[ALB_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x = 56
y = ALB_driver_laps_A_2024["LapTime"].loc[56].value
ax.annotate('Field accident', (x-0.5, y), xytext=(x - 20, y-2*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Australia Race (ALB) - No."+str(ALB_A_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

plt.tight_layout()
plt.show()
##

MAG_driver_laps_A_2024[MAG_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_A_2024.loc[:,['Compound','LapNumber','Time','LapTime']]

## MAG 10
MAG_A_2024_r = int(MAG_driver_laps_A_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_A_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=MAG_driver_laps_A_2024,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(MAG_driver_laps_A_2024[MAG_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(MAG_driver_laps_A_2024.loc[MAG_driver_laps_A_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 + 0.02*1e11), ha='center', color= palette[MAG_driver_laps_A_2024[MAG_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Australia Race (MAG) - No."+str(MAG_A_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 56
y = MAG_driver_laps_A_2024["LapTime"].loc[56].value
ax.annotate('Field accident', (x-0.5, y), xytext=(x - 20, y-1*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

HUL_driver_laps_A_2024[HUL_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
HUL_driver_laps_A_2024.loc[:,['Compound','LapNumber','Time','LapTime']]

## HUL 9
HUL_A_2024_r = int(HUL_driver_laps_A_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=HUL_driver_laps_A_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=HUL_driver_laps_A_2024.iloc[:17,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_A_2024.loc[17:35,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=HUL_driver_laps_A_2024.loc[35:,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(HUL_driver_laps_A_2024[HUL_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(HUL_driver_laps_A_2024.loc[HUL_driver_laps_A_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 - 0.055*1e11), ha='center', color= palette[HUL_driver_laps_A_2024[HUL_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x2 = int(HUL_driver_laps_A_2024[HUL_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y2 = float(HUL_driver_laps_A_2024.loc[HUL_driver_laps_A_2024["LapNumber"] == x2, "LapTime"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 + 0.02*1e11), ha='center', color= palette[HUL_driver_laps_A_2024[HUL_driver_laps_A_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Australia Race (HUL) - No."+str(HUL_A_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 56
y = HUL_driver_laps_A_2024["LapTime"].loc[56].value
ax.annotate('Field accident', (x-0.5, y), xytext=(x - 20, y-1*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

################################################################################################# 2024 Japan
race_J_2024 = fastf1.get_session(2024, 'Japan', 'R')
race_J_2024.load()

race_J_2024.laps.drop_duplicates(subset='Driver') # HUL, MAG, SAR 完赛; ALB 未完赛

ALB_driver_laps_J_2024 = race_J_2024.laps.pick_driver("ALB").reset_index() # 1圈
MAG_driver_laps_J_2024 = race_J_2024.laps.pick_driver("MAG").reset_index() # 52圈
SAR_driver_laps_J_2024 = race_J_2024.laps.pick_driver("SAR").reset_index() # 52圈
HUL_driver_laps_J_2024 = race_J_2024.laps.pick_driver("HUL").reset_index() # 52圈

SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
SAR_driver_laps_J_2024.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(SAR_driver_laps_J_2024))):
    SAR_driver_laps_J_2024.loc[i,'LapTime'] = SAR_driver_laps_J_2024.loc[i,'Time']-SAR_driver_laps_J_2024.loc[i-1,'Time']

## SAR 异常值
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=SAR_driver_laps_J_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Japan Race (SAR)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 1
y = SAR_driver_laps_J_2024["LapTime"].loc[1].value
ax.annotate('Cleaning due to crash', (x+1.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

## SAR 17 去除异常值
SAR_J_2024_r = int(SAR_driver_laps_J_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=SAR_driver_laps_J_2024.drop(1),
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=SAR_driver_laps_J_2024.drop(1).iloc[1:34,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=SAR_driver_laps_J_2024.drop(1).loc[34:41,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=SAR_driver_laps_J_2024.drop(1).loc[41:,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(SAR_driver_laps_J_2024.loc[SAR_driver_laps_J_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1+3, y1), xytext=(x1+5, y1 - 0.05*1e11), ha='center', color= palette[SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x2 = int(SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[2,0])
y2 = float(SAR_driver_laps_J_2024.loc[SAR_driver_laps_J_2024["LapNumber"] == x2, "LapTime"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 + 0.06*1e11), ha='center', color= palette[SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[2,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x3 = int(SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[3,0])
y3 = float(SAR_driver_laps_J_2024.loc[SAR_driver_laps_J_2024["LapNumber"] == x3, "LapTime"].values[0])
ax.annotate(f'Pit 3 : Lap {x3}', xy=(x3, y3), xytext=(x3, y3 + 0.06*1e11), ha='center', color= palette[SAR_driver_laps_J_2024[SAR_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[3,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Japan Race (SAR) - No."+str(SAR_J_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

plt.tight_layout()
plt.show()
##

MAG_driver_laps_J_2024[MAG_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_J_2024.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(MAG_driver_laps_J_2024))):
    MAG_driver_laps_J_2024.loc[i,'LapTime'] = MAG_driver_laps_J_2024.loc[i,'Time']-MAG_driver_laps_J_2024.loc[i-1,'Time']

## MAG 异常值
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_J_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Japan Race (MAG)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 1
y = MAG_driver_laps_J_2024["LapTime"].loc[1].value
ax.annotate('Cleaning due to crash', (x+1.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

## MAG 13 去除异常值
MAG_J_2024_r = int(MAG_driver_laps_J_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_J_2024.drop(1),
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=MAG_driver_laps_J_2024.drop(1),
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(MAG_driver_laps_J_2024[MAG_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y1 = float(MAG_driver_laps_J_2024.loc[MAG_driver_laps_J_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1+3, y1), xytext=(x1, y1 + 0.1*1e11), ha='center', color= palette[MAG_driver_laps_J_2024[MAG_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Japan Race (MAG) - No."+str(MAG_J_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

plt.tight_layout()
plt.show()
##

HUL_driver_laps_J_2024[HUL_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
HUL_driver_laps_J_2024.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(HUL_driver_laps_J_2024))):
    HUL_driver_laps_J_2024.loc[i,'LapTime'] = HUL_driver_laps_J_2024.loc[i,'Time']-HUL_driver_laps_J_2024.loc[i-1,'Time']

## HUL 异常值
fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=HUL_driver_laps_J_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Japan Race (HUL)")

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 1
y = HUL_driver_laps_J_2024["LapTime"].loc[1].value
ax.annotate('Cleaning due to crash', (x+1.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

## HUL 11 去除异常值
HUL_J_2024_r = int(HUL_driver_laps_J_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=HUL_driver_laps_J_2024.drop(1),
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=HUL_driver_laps_J_2024.drop(1),
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(HUL_driver_laps_J_2024[HUL_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y1 = float(HUL_driver_laps_J_2024.loc[HUL_driver_laps_J_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1+3, y1), xytext=(x1, y1 + 0.06*1e11), ha='center', color= palette[HUL_driver_laps_J_2024[HUL_driver_laps_J_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 Japan Race (HUL) - No."+str(HUL_J_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

plt.tight_layout()
plt.show()
##

################################################################################################# 2024 China
race_C_2024 = fastf1.get_session(2024, 'China', 'R')
race_C_2024.load()

race_C_2024.laps.drop_duplicates(subset='Driver') # 均完赛

ALB_driver_laps_C_2024 = race_C_2024.laps.pick_driver("ALB").reset_index() # 56圈
MAG_driver_laps_C_2024 = race_C_2024.laps.pick_driver("MAG").reset_index() # 56圈
SAR_driver_laps_C_2024 = race_C_2024.laps.pick_driver("SAR").reset_index() # 56圈
HUL_driver_laps_C_2024 = race_C_2024.laps.pick_driver("HUL").reset_index() # 56圈

ALB_driver_laps_C_2024[ALB_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
ALB_driver_laps_C_2024.loc[:,['Compound','LapNumber','Time','LapTime']]

## ALB 12
ALB_C_2024_r = int(ALB_driver_laps_C_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=ALB_driver_laps_C_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=ALB_driver_laps_C_2024,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(ALB_driver_laps_C_2024[ALB_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y1 = float(ALB_driver_laps_C_2024.loc[ALB_driver_laps_C_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1+3, y1), xytext=(x1, y1 - 0.06*1e11), ha='center', color= palette[ALB_driver_laps_C_2024[ALB_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 China Race (ALB) - No."+str(ALB_C_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 30
y = ALB_driver_laps_C_2024["LapTime"].loc[28].value
ax.annotate('Field accident', (x+1.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

SAR_driver_laps_C_2024[SAR_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
SAR_driver_laps_C_2024.loc[:,['Compound','LapNumber','Time','LapTime']]

## SAR 17
SAR_C_2024_r = int(SAR_driver_laps_C_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=SAR_driver_laps_C_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')

sns.lineplot(data=SAR_driver_laps_C_2024.iloc[:12,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=SAR_driver_laps_C_2024.loc[12:24,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)
sns.lineplot(data=SAR_driver_laps_C_2024.loc[24:,:],
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(SAR_driver_laps_C_2024[SAR_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(SAR_driver_laps_C_2024.loc[SAR_driver_laps_C_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1, y1), xytext=(x1, y1 - 0.02*1e11), ha='center', color= palette[SAR_driver_laps_C_2024[SAR_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

x2 = int(SAR_driver_laps_C_2024[SAR_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y2 = float(SAR_driver_laps_C_2024.loc[SAR_driver_laps_C_2024["LapNumber"] == x2, "LapTime"].values[0])
ax.annotate(f'Pit 2 : Lap {x2}', xy=(x2, y2), xytext=(x2, y2 - 0.04*1e11), ha='center', color= palette[SAR_driver_laps_C_2024[SAR_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=11), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 China Race (SAR) - No."+str(SAR_C_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 30
y = SAR_driver_laps_C_2024["LapTime"].loc[28].value
ax.annotate('Field accident', (x+1.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

HUL_driver_laps_C_2024[HUL_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
HUL_driver_laps_C_2024.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(HUL_driver_laps_C_2024))):
    HUL_driver_laps_C_2024.loc[i,'LapTime'] = HUL_driver_laps_C_2024.loc[i,'Time']-HUL_driver_laps_C_2024.loc[i-1,'Time']

## HUL 10
HUL_C_2024_r = int(HUL_driver_laps_C_2024.Position.iloc[-1])

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=HUL_driver_laps_C_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=HUL_driver_laps_C_2024,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(HUL_driver_laps_C_2024[HUL_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,0])
y1 = float(HUL_driver_laps_C_2024.loc[HUL_driver_laps_C_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1+3, y1), xytext=(x1, y1 + 0.05*1e11), ha='center', color= palette[HUL_driver_laps_C_2024[HUL_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[0,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 China Race (HUL) - No."+str(HUL_C_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 30
y = HUL_driver_laps_C_2024["LapTime"].loc[28].value
ax.annotate('Field accident', (x+1.5, y), xytext=(x + 5, y+2*1e9), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

MAG_driver_laps_C_2024[MAG_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','PitOutTime', 'PitInTime']]
MAG_driver_laps_C_2024.loc[:,['Compound','LapNumber','Time','LapTime']]
for i in reversed(range(1,len(MAG_driver_laps_C_2024))):
    MAG_driver_laps_C_2024.loc[i,'LapTime'] = MAG_driver_laps_C_2024.loc[i,'Time']-MAG_driver_laps_C_2024.loc[i-1,'Time']


## MAG 16
MAG_C_2024_r = int(MAG_driver_laps_C_2024.Position.iloc[-1])+1

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

sns.scatterplot(data=MAG_driver_laps_C_2024,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
                s=60,
                linewidth=0,
                legend='auto')
                
sns.lineplot(data=MAG_driver_laps_C_2024,
             x="LapNumber",
             y="LapTime",
             hue="Compound",
             palette={'SOFT': '#da291c', 'MEDIUM': '#ffd12e', 'HARD': '#1471e3', 'INTERMEDIATE': '#43b02a', 'WET': '#0067ad', 'UNKNOWN': '#00ffff', 'TEST-UNKNOWN': '#434649'},
             linewidth=2,
             legend=False)

x1 = int(MAG_driver_laps_C_2024[MAG_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,0])
y1 = float(MAG_driver_laps_C_2024.loc[MAG_driver_laps_C_2024["LapNumber"] == x1, "LapTime"].values[0])
ax.annotate(f'Pit 1 : Lap {x1}', xy=(x1+3, y1), xytext=(x1, y1 - 0.05*1e11), ha='center', color= palette[MAG_driver_laps_C_2024[MAG_driver_laps_C_2024['PitInTime'].notna()].loc[:,['LapNumber','Compound']].iloc[1,1]],
fontproperties=FontProperties(weight='bold', size=12), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time /s")

# ax.invert_yaxis()
plt.suptitle("Laptimes in the 2024 China Race (MAG) - No."+str(MAG_C_2024_r))

# Turn on major grid lines
plt.grid(color='black', which='major', axis='both', linestyle='dashed',alpha=0.7)
sns.despine(left=True, bottom=True)
ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

x = 30
y = MAG_driver_laps_C_2024["LapTime"].loc[27].value
ax.annotate('Field accident', (x+1.5, y), xytext=(x + 5, y), arrowprops=dict(arrowstyle="->"), fontsize=12, color="red",fontweight="bold")

plt.tight_layout()
plt.show()
##

################################################################################################# 评分
from scipy.optimize import curve_fit


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y= np.array([25, 18, 15, 12, 10, 8, 7, 4, 2, 1])

# 多项式拟合
degree = 4  # 设置多项式的阶数
coeffs = np.polyfit(x, y, degree)  # 多项式拟合的系数

# 生成拟合曲线上的数据点
x_fit = np.linspace(1, 10, 21)
y_fit = np.polyval(coeffs, x_fit)  # 计算拟合曲线上的y值


plt.figure(figsize=(10,10), dpi=300)
plt.scatter(x, y, label='Data Points')
plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# SAR
index_SAR=[-1,-1,17,17]
total_SAR=sum(y_fit[index_SAR])
avg_SAR=total_SAR/len(index_SAR)

# ALB
index_ALB=[10,-1,-1,-1,11,-1,12]
total_ALB=sum(y_fit[index_ALB])
avg_ALB=total_ALB/len(index_ALB)

# HUL
index_HUL=[7,14,9,11,10]
total_HUL=sum(y_fit[index_HUL])
avg_HUL=total_HUL/len(index_HUL)

# MAG
index_MAG=[14,14,-1,15,10,13,16]
total_MAG=sum(y_fit[index_MAG])
avg_MAG=total_MAG/len(index_MAG)

# 画图
# 平均值数据
averages = [avg_SAR, avg_ALB, avg_HUL, avg_MAG]
drivers = ['SAR', 'ALB', 'HUL', 'MAG']
# 颜色映射
cmap = plt.get_cmap('Blues')  # 获取蓝色系的颜色映射
normalized_averages = [(value +2) / (max(averages) +2) for value in averages]
colors = [cmap(value) if value >= 0 else cmap(0) for value in normalized_averages]
# 绘制柱状图
plt.figure(figsize=(10,10), dpi=300)
plt.bar(drivers, averages, color=colors)
for i, value in enumerate(averages):
    plt.text(i, value + 0.08, str(round(value,3)), ha='center')
plt.xlabel('Drivers')
plt.ylabel('Average Point')
plt.title('Average Point Comparison')
plt.show()

############# ALL 2022-2024
# SAR
index_SAR=[12,16,16,16,20,18,20,-1,13,11,18,17,-1,13,14,-1,-1,10,16,11,16,16,20,14,17,17]
total_SAR=sum(y_fit[index_SAR])
avg_SAR=total_SAR/len(index_SAR)

# ALB
index_ALB=[13,14,10,11,9,18,-1,12,13,1,12,13,17,10,12,-1,-1,13,12,15,13,10,-1,-1,12,14,14,16,7,11,8,11,14,8,7,11,-1,13,9,9,-1,12,14,15,11,11,-1,12]
total_ALB=sum(y_fit[index_ALB])
avg_ALB=total_ALB/len(index_ALB)

# HUL
index_HUL=[17,12,15,12,7,17,15,17,15,15,-1,13,14,18,12,17,13,14,16,11,13,12,19,15,16,10,9,11,10]
total_HUL=sum(y_fit[index_HUL])
avg_HUL=total_HUL/len(index_HUL)

# MAG
index_MAG=[5,9,14,9,16,17,1,-1,17,10,8,-1,16,16,15,16,12,14,9,17,-1,17,13,10,17,13,10,19,18,17,18,-1,17,15,16,18,10,15,14,14,-1,-1,13,20,12,12,10,13,16]
total_MAG=sum(y_fit[index_MAG])
avg_MAG=total_MAG/len(index_MAG)

# 画图
# 平均值数据
averages = [avg_SAR, avg_ALB, avg_HUL, avg_MAG]
drivers = ['SAR', 'ALB', 'HUL', 'MAG']
# 颜色映射
cmap = plt.get_cmap('Blues')  # 获取蓝色系的颜色映射
normalized_averages = [(value +2) / (max(averages) +2) for value in averages]
colors = [cmap(value) if value >= 0 else cmap(0) for value in normalized_averages]
# 绘制柱状图
plt.figure(figsize=(10,10), dpi=300)
plt.bar(drivers, averages, color=colors)
for i, value in enumerate(averages):
    plt.text(i, value + 0.08, str(round(value,3)), ha='center')
plt.xlabel('Drivers')
plt.ylabel('Average Point')
plt.title('Average Point Comparison From All 2022-2024 races')
plt.show()


############# ALL 2022-2024

data = [y_fit[index_SAR],y_fit[index_ALB],y_fit[index_HUL],y_fit[index_MAG]]

plt.figure(figsize=(10,10), dpi=300)
plt.boxplot(data)
plt.xticks([1,2,3,4], ['SAR','ALB','HUL','MAG'])
plt.xlabel('Drivers')
plt.ylabel('Average Point')
plt.title('Boxplot of All points')
plt.show()































