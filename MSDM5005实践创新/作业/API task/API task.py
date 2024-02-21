import requests

################################################################################################
def get_bus_route_info(company_id, route):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/route/{company_id}/{route}"
    response = requests.get(url)
    data = response.json()
    return data

# 调用函数获取A28公交线路信息
company_id = "CTB"
route = "A28"
route_info = get_bus_route_info(company_id, route)

# 打印结果
print(route_info)

################################################################################################

def get_bus_route_stop_info(company_id, route, direction):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/route-stop/{company_id}/{route}/{direction}"
    response = requests.get(url)
    data = response.json()
    return data

# 调用函数获取A28公交线路进站方向站点信息
company_id = "CTB"
route = "A28"
direction = "inbound"
route_stop_info = get_bus_route_stop_info(company_id, route, direction)

# 打印结果
print(route_stop_info)

# 提取站点信息并保存在列表中
stops = [entry['stop'] for entry in route_stop_info['data']]

# 打印站点信息列表
print(stops)

################################################################################################
import pandas as pd

def get_stop_name(stop_id):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/stop/{stop_id}"
    response = requests.get(url)
    data = response.json()
    return data

# 站点ID列表
stop_ids = ['001837', '002672', '003304', '003482', '003540', '001854', '001766', '001523', '001688', '001696', '001697', '001653', '001652', '001780', '001788', '003498', '003499', '003500', '002701', '003068', '002705', '001677', '001678', '003234', '001823', '001763', '001764', '001824', '001825', '001826', '003160', '003225', '002919', '002928', '002929', '003329']

# 创建一个空的列表
data = []

# 遍历每个站点ID，获取对应的name_tc，并添加到列表中
for stop_id in stop_ids:
    stop_info = get_stop_name(stop_id)
    name_tc = stop_info['data']['name_tc']
    data.append({'stop_id': stop_id, 'name_tc': name_tc})

# 创建DataFrame
df = pd.DataFrame(data)

# 打印数据集
print(df)

################################################################################################

stop_name = "將軍澳站, 寶邑路"
direction = "inbound"

# 根据条件筛选DataFrame
filtered_df = df[(df['name_tc'] == stop_name)]

# 获取匹配的stop_id
stop_id = filtered_df['stop_id'].values[0]

# 打印stop_id
print(stop_id)


################################################################################################

def get_eta(company_id, stop_id, route):
    url = f"https://rt.data.gov.hk/v2/transport/citybus/eta/{company_id}/{stop_id}/{route}"  # 替换为ETA API的URL
    response = requests.get(url)
    data = response.json()
    return data

company_id = "CTB"
route = "A28"
stop_id = "001825"

eta_data = get_eta(company_id, stop_id, route)

print(eta_data)

# 提取数据字段
company_id = eta_data['data'][0]['co']
route = eta_data['data'][0]['route']
direction = eta_data['data'][0]['dir']
stop_id = eta_data['data'][0]['stop']
destination = eta_data['data'][0]['dest_tc']
eta = eta_data['data'][0]['eta']

# 创建数据集（DataFrame）
data = {'公司ID': [company_id], '路线': [route], '方向': [direction],
        '停靠站ID': [stop_id], '目的地': [destination], '预计到达时间': [eta]}
df = pd.DataFrame(data)

# 打印数据集（DataFrame）
print(df)

################################################################################################
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










