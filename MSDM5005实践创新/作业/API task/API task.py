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










