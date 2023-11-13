import pandas as pd
import requests
import json

#1

#a
# need "Stop Data", "Route-Stop List Data" and "Stop ETA Data".

#b

# json.dumps(data, sort_keys=True, indent=4)

def retrieve_stops(route,bound):
    url = "https://data.etabus.gov.hk/v1/transport/kmb/route-stop"
    response = requests.get(url)
    data = response.json()['data']
    
    route_stops = []
    for item in data:
        if item['route'] == route and item['bound'] == bound:
            route_stops.append(item['stop'])
    
    tentlist=[]
    for i in route_stops:
        response = requests.get("https://data.etabus.gov.hk/v1/transport/kmb/stop/" + i)
        tentlist.append(response.json()['data'])
    
    
    df0 = pd.DataFrame(tentlist)
    df = df0.drop(df0.columns[-2:], axis=1)

    return df

stops_I_91M = retrieve_stops("91M","I")
stops_I_91M = stops_I_91M.drop_duplicates(subset='stop')
stops_O_91M = retrieve_stops("91M","O")
stops_O_91M = stops_O_91M.drop_duplicates(subset='stop')

print(stops_I_91M)
print(stops_O_91M)

#c

# B3E60EE895DBBF06 香港科技大学(北)
# B002CEF0DBC568F5 香港科技大学(南)

def get_timetable(direction):
    
    if direction == "N":
        ID = "B3E60EE895DBBF06"
    elif direction == "S":
        ID = "B002CEF0DBC568F5"
    else :
        return "No such direction"
    
    response = requests.get("https://data.etabus.gov.hk/v1/transport/kmb/stop-eta/" + ID)
    data0 = response.json()['data']
    filtered_data = [{k: v for k, v in d.items() if k in ["route", "eta"]} for d in data0]
    df = pd.DataFrame(filtered_data).sort_values('eta')
    
    return df

get_timetable("N")
get_timetable("S")
get_timetable("E")


#2
#a
import matplotlib.pyplot as plt
import numpy as np

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ser = Service(r'D:\app\chromedriver\chromedriver.exe')
options = Options()
options.add_argument("headless")              # headless is the option to whether to have a browser popping out
driver = webdriver.Chrome(service=ser, options=options)

url_template = "https://www.hko.gov.hk/en/cis/dailyExtract.htm?y={}&m={}"

def collect():
    df0 = pd.DataFrame(columns=["Date", "Mean Pressure", "Absolute Daily Max Air Temp.",
                                 "Mean Air Temp.", "Absolute Daily Min Air Temp.", "Mean Dew Point",
                                 "Mean Relative Humidity","Mean Amount of Cloud", "Total Rainfall"])

    start_date = datetime(2003, 11, 30)
    end_date = datetime(2023, 11, 1)

    while start_date <= end_date:
        driver.implicitly_wait(5)
        url = url_template.format(start_date.year, start_date.month)
        driver.get(url)

        elements = driver.find_elements(By.XPATH, '//table[@id="t1"]/tr[position() >= 4 and position() <= last()]/td[position() < 10]')
        data_list = [element.text for element in elements]
        df = pd.DataFrame([data_list[i:i+9] for i in range(0, len(data_list), 9)],
                              columns=["Date", "Mean Pressure", "Absolute Daily Max Air Temp.",
                              "Mean Air Temp.", "Absolute Daily Min Air Temp.", "Mean Dew Point",
                              "Mean Relative Humidity", "Mean Amount of Cloud", "Total Rainfall"])

        df = df[df.loc[:,'Date'].str.isnumeric()]
        df.loc[:,'Date'] = pd.to_datetime(str(start_date.year) +"-"+ str(start_date.month)
                      +"-"+ df['Date'], format='%Y-%m-%d').dt.date
        df.loc[:,'Total Rainfall'] = df['Total Rainfall'].replace('Trace', '0.02')
        df0 = pd.concat([df0, df], axis=0)
        
        start_date += relativedelta(months=1)
    
    return df0

df0 = collect()

df0.to_csv('C:\\Users\\张铭韬\\Desktop\\weather_data_all.csv')

dfw = pd.read_csv('C:\\Users\\张铭韬\\Desktop\\weather_data_all.csv')
dfw = dfw.iloc[:,1:]
print(dfw)


#b
driver.get("https://www.hko.gov.hk/en/wxinfo/climat/warndb/warndb13.shtml")

line_no = driver.find_element(By.ID, 'startdate')
line_no.clear()
line_no.send_keys('200311')

line_no_ = driver.find_element(By.ID, 'enddate')
line_no_.clear()
line_no_.send_keys('202310')

search_button = driver.find_element(By.ID, 'warningsearch')
search_button.click()

elements_h = driver.find_elements(By.XPATH, '//table[@id="result"]/tbody/tr[position() >= 3 and position() <= last()]/td[position() < 5]')

hot = [element.text for element in elements_h]

month_mapping = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

for i in range(1, len(hot), 2):
    month_g = hot[i].split('/')[1]
    month_number = month_mapping.get(month_g)
    hot[i] = hot[i].replace(month_g, month_number)

df_h = pd.DataFrame([hot[i:i+4] for i in range(0, len(hot), 4)],
                      columns=["starttime", "startdate", "endtime","enddate"])

start = []
end = []
for i in range(len(df_h['starttime'])):
    m1 = df_h['startdate'][i] + ' ' + df_h['starttime'][i]
    start.append(datetime.strptime(m1, '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S'))
    m2 = df_h['enddate'][i] + ' ' + df_h['endtime'][i]
    end.append(datetime.strptime(m2, '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S'))

dfh = pd.DataFrame({"Start": start, "End": end})

dfh.to_csv('C:\\Users\\张铭韬\\Desktop\\hot_weather.csv')

dfh = pd.read_csv('C:\\Users\\张铭韬\\Desktop\\hot_weather.csv')
dfh = dfh.iloc[:,1:]
print(dfh)


####
driver.get("https://www.hko.gov.hk/en/wxinfo/climat/warndb/warndb12.shtml")

line_no = driver.find_element(By.ID, 'startdate')
line_no.clear()
line_no.send_keys('200311')

line_no_ = driver.find_element(By.ID, 'enddate')
line_no_.clear()
line_no_.send_keys('202310')

search_button = driver.find_element(By.ID, 'warningsearch')
search_button.click()

elements_c = driver.find_elements(By.XPATH, '//table[@id="result"]/tbody/tr[position() >= 3 and position() <= last()]/td[position() < 5]')

cold = [element.text for element in elements_c]

for i in range(1, len(cold), 2):
    month_g = cold[i].split('/')[1]
    month_number = month_mapping.get(month_g)
    cold[i] = cold[i].replace(month_g, month_number)

df_c = pd.DataFrame([cold[i:i+4] for i in range(0, len(cold), 4)],
                      columns=["starttime", "startdate", "endtime","enddate"])

start = []
end = []
for i in range(len(df_c['starttime'])):
    m1 = df_c['startdate'][i] + ' ' + df_c['starttime'][i]
    start.append(datetime.strptime(m1, '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S'))
    m2 = df_c['enddate'][i] + ' ' + df_c['endtime'][i]
    end.append(datetime.strptime(m2, '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S'))

dfc = pd.DataFrame({"Start": start, "End": end})

dfc.to_csv('C:\\Users\\张铭韬\\Desktop\\cold_weather.csv')

dfc = pd.read_csv('C:\\Users\\张铭韬\\Desktop\\cold_weather.csv')
dfc = dfc.iloc[:,1:]
print(dfc)

#c

dfw['Very Hot Weather Warning'] = dfw['Date'].apply(lambda x: any((x >= (datetime.strptime(start, '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')) and (x <= end) for start, end in zip(dfh['Start'], dfh['End'])))

# dfw[dfw['Very Hot Weather Warning']==True]

dfw['Cold Weather Warning'] = dfw['Date'].apply(lambda x: any((x >= (datetime.strptime(start, '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')) and (x <= end) for start, end in zip(dfc['Start'], dfc['End'])))

# dfw[dfw['Cold Weather Warning']==True]

print(dfw)


#d
HOT = dfw[dfw['Very Hot Weather Warning']==True].iloc[:,1:9]
NOTHOT = dfw[dfw['Very Hot Weather Warning']==False].iloc[:,1:9]

COLD = dfw[dfw['Cold Weather Warning']==True].iloc[:,1:9]
NOTCOLD = dfw[dfw['Cold Weather Warning']==False].iloc[:,1:9]

# hot
fig, axs = plt.subplots(2, 4, figsize=(10, 5))

parameters = ["Mean Pressure", "Absolute Daily Max Air Temp.",
              "Mean Air Temp.", "Absolute Daily Min Air Temp.", "Mean Dew Point",
              "Mean Relative Humidity","Mean Amount of Cloud", "Total Rainfall"]

for i, param in enumerate(parameters):
    row = i // 4
    col = i % 4
    axs[row, col].hist(NOTHOT.iloc[:,i], bins=40, alpha=0.8,label='NOT HOT')
    axs[row, col].hist(HOT.iloc[:,i], bins=40, alpha=0.8,label='HOT')
    axs[row, col].set_title(param,fontsize=7.5)
    axs[row, col].legend(loc='upper left',fontsize=6)
    axs[row, col].tick_params(axis='y', pad=0)
    
fig.suptitle("Histograms of Climate Parameters between Hot/Not Hot days", fontsize=12)
plt.tight_layout()
plt.show()

# cold
fig, axs = plt.subplots(2, 4, figsize=(10, 5))

for i, param in enumerate(parameters):
    row = i // 4
    col = i % 4
    axs[row, col].hist(NOTCOLD.iloc[:,i], bins=40, alpha=0.8,label='NOT COLD')
    axs[row, col].hist(COLD.iloc[:,i], bins=40, alpha=0.8,label='COLD')
    axs[row, col].set_title(param,fontsize=7.5)
    axs[row, col].legend(loc='upper left',fontsize=6)
    axs[row, col].tick_params(axis='y', pad=0)

fig.suptitle("Histograms of Climate Parameters between Cold/Not Cold days", fontsize=12)
plt.tight_layout()
plt.show()





