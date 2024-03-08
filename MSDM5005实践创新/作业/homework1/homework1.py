# 1. GET request

import requests

################################################################################################
def get_bus_route_info(company_id, route):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/route/{company_id}/{route}"
    response = requests.get(url)
    data = response.json()
    return data

# ���ú�����ȡA28������·��Ϣ
company_id = "CTB"
route = "A28"
route_info = get_bus_route_info(company_id, route)

# ��ӡ���
print(route_info)

################################################################################################

def get_bus_route_stop_info(company_id, route, direction):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/route-stop/{company_id}/{route}/{direction}"
    response = requests.get(url)
    data = response.json()
    return data

# ���ú�����ȡA28������·��վ����վ����Ϣ
company_id = "CTB"
route = "A28"
direction = "inbound"
route_stop_info = get_bus_route_stop_info(company_id, route, direction)

# ��ӡ���
print(route_stop_info)

# ��ȡվ����Ϣ���������б���
stops = [entry['stop'] for entry in route_stop_info['data']]

# ��ӡվ����Ϣ�б�
print(stops)

################################################################################################
import pandas as pd

def get_stop_name(stop_id):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/stop/{stop_id}"
    response = requests.get(url)
    data = response.json()
    return data

# վ��ID�б�
stop_ids = ['001837', '002672', '003304', '003482', '003540', '001854', '001766', '001523', '001688', '001696', '001697', '001653', '001652', '001780', '001788', '003498', '003499', '003500', '002701', '003068', '002705', '001677', '001678', '003234', '001823', '001763', '001764', '001824', '001825', '001826', '003160', '003225', '002919', '002928', '002929', '003329']

# ����һ���յ��б�
data = []

# ����ÿ��վ��ID����ȡ��Ӧ��name_tc������ӵ��б���
for stop_id in stop_ids:
    stop_info = get_stop_name(stop_id)
    name_tc = stop_info['data']['name_tc']
    data.append({'stop_id': stop_id, 'name_tc': name_tc})

# ����DataFrame
df = pd.DataFrame(data)

# ��ӡ���ݼ�
print(df)

################################################################################################

stop_name = "��܊��վ, ����·"
direction = "inbound"

# ��������ɸѡDataFrame
filtered_df = df[(df['name_tc'] == stop_name)]

# ��ȡƥ���stop_id
stop_id = filtered_df['stop_id'].values[0]

# ��ӡstop_id
print(stop_id)


################################################################################################

def get_eta(company_id, stop_id, route):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/eta/{company_id}/{stop_id}/{route}"  # �滻ΪETA API��URL
    response = requests.get(url)
    data = response.json()
    return data

company_id = "CTB"
route = "A28"
stop_id = "001825"

eta_data = get_eta(company_id, stop_id, route)

print(eta_data)

# ��ȡ�����ֶ�
company_id = eta_data['data'][0]['co']
route = eta_data['data'][0]['route']
direction = eta_data['data'][0]['dir']
stop_id = eta_data['data'][0]['stop']
destination = eta_data['data'][0]['dest_tc']
eta = eta_data['data'][0]['eta']

# �������ݼ���DataFrame��
data = {'��˾ID': [company_id], '·��': [route], '����': [direction],
        'ͣ��վID': [stop_id], 'Ŀ�ĵ�': [destination], 'Ԥ�Ƶ���ʱ��': [eta]}
df = pd.DataFrame(data)

# ��ӡ���ݼ���DataFrame��
print(df)

################################################################################################
# 2. POST request

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="63d53d8ef5074a17ac71f85bd7de1fd0",
    api_version="2023-05-15",
    azure_endpoint="https://hkust.azure-api.net"
)

response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role":"system","content": "You are a helpful assistant."},
        {"role":"user","content": "what is openAI?"},
    ],
)

print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role":"system","content": "You are a helpful assistant."},
        {"role":"user","content": "what is Virtual Youtuber?"},
    ],
)

print(response.choices[0].message.content)

################################################################################################
# 3. Web Scraping
import numpy as np
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup

url = f"https://en.wikipedia.org/wiki/List_of_Formula_One_World_Drivers%27_Champions"

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ser = Service(r'D:\app\chromedriver\chromedriver.exe')
options = Options()
options.add_argument("headless")              # headless is the option to whether to have a browser popping out
driver = webdriver.Chrome(service=ser, options=options)
driver.get(url)

elements = driver.find_elements(By.XPATH, '//table[@class="wikitable sortable jquery-tablesorter"][@style="font-size:85%; text-align:center;"]/tbody/tr[position() >= 1 and position() <= last()]/td[position() = 3]')
data_list1 = [element.text for element in elements]
data_list1[5]='43'
print(data_list1)

data_list1 = [int(age) if age.isdigit() else 0 for age in data_list1]
mean_age = np.mean(data_list1)

import matplotlib.pyplot as plt
# Create a bar chart to display the age distribution
fig = plt.figure(figsize=(10,10), dpi=300)
plt.hist(data_list1, bins=range(20, 50, 5), edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()


#####################################################################
elements = driver.find_elements(By.XPATH, '//table[@class="wikitable sortable jquery-tablesorter"][@style="font-size:85%; text-align:center;"]/tbody/tr[position() >= 1 and position() <= last()]/td[position() = 12]')
data_list2 = [element.text for element in elements]
data_list2.insert(6, '66.667 (45.833)')

for i in range(len(data_list2)):
    data_list2[i] = data_list2[i][:6]

print(data_list2)

data_list2 = [float(points) for points in data_list2]

points_counts = {points: data_list2.count(points) for points in set(data_list2)}
sorted_points = sorted(points_counts.keys())
freq = [points_counts[points] for points in sorted_points]

# Construct a line chart to depict the distribution
fig = plt.figure(figsize=(10,10), dpi=300)
plt.plot(sorted_points, freq, marker='o')
plt.xlabel('points')
plt.ylabel('Frequency')
plt.title('points Distribution')
plt.show()

#####################################################################
fig = plt.figure(figsize=(10,10), dpi=300)
plt.scatter(data_list1, data_list2)
plt.xlabel("Ages of World Drivers' Champions")
plt.ylabel("% Points Data")
plt.title("Correlation between % Points Data and Ages of World Drivers' Champions")
plt.show()




