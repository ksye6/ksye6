import requests
import schedule
import time
import pandas as pd

#################################### schedule part
data = None

def get_weather_data():
    url = 'https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=sc'
    response = requests.get(url)
    data = response.json()

def job():
    print("Running the script...")
    get_weather_data()

schedule.every(12).hours.do(job)
# while True:
#     schedule.run_pending()
#     time.sleep(1)

schedule.get_jobs()

schedule.clear()

#################################### schedule part
response = requests.get('https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=sc')
data = response.json()

df = pd.DataFrame(data['weatherForecast'])
print(df)

df.to_csv('weather_data.csv', index=False, encoding='gbk')



