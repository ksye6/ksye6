import schedule
import time

############################ ����ʹ��  https://zhuanlan.zhihu.com/p/501021369

def job():
    print("I'm working...")

# ÿʮ����ִ������
schedule.every(10).minutes.do(job)
# ÿ��Сʱִ������
schedule.every().hour.do(job)
# ÿ���10:30ִ������
schedule.every().day.at("10:30").do(job)
# ÿ����ִ������
schedule.every().monday.do(job)
# ÿ����������13:15��ִ������
schedule.every().wednesday.at("13:15").do(job)
# ÿ���ӵĵ�17��ִ������
schedule.every().minute.at(":17").do(job)


schedule.every(2).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)

############################ ����ʹ��

def job_that_executes_once():
    # �˴���д������ֻ��ִ��һ��...
    return schedule.CancelJob

schedule.every().day.at('22:30').do(job_that_executes_once)

while True:
    schedule.run_pending()
    time.sleep(1)


############################ ��������


def greet(name):
    print('Hello', name)

# do() ������Ĳ������ݸ�job����
schedule.every(2).seconds.do(greet, name='Alice')
schedule.every(4).seconds.do(greet, name='Bob')


############################ ��ȡĿǰ���е���ҵ

schedule.get_jobs()

############################ ȡ��������ҵ


schedule.clear()


############################ ͨ����ǩ���˻�ȡ��ҵ��ȡ����ҵ

def greet(name):
    print('Hello {}'.format(name))

# .tag ���ǩ
schedule.every().day.do(greet, 'Andrea').tag('daily-tasks', 'friend')
schedule.every().hour.do(greet, 'John').tag('hourly-tasks', 'friend')
schedule.every().hour.do(greet, 'Monica').tag('hourly-tasks', 'customer')
schedule.every().day.do(greet, 'Derek').tag('daily-tasks', 'guest')

# get_jobs(��ǩ)�����Ի�ȡ���иñ�ǩ������
friends = schedule.get_jobs('friend')

# ȡ������ daily-tasks ��ǩ������
schedule.clear('daily-tasks')


############################ �趨��ҵ��ֹʱ��

def job():
    print('Boo')

# ÿ��Сʱ������ҵ��18:30��ֹͣ
schedule.every(1).hours.until("18:30").do(job)

# ÿ��Сʱ������ҵ��2030-01-01 18:33 today
schedule.every(1).hours.until("2030-01-01 18:33").do(job)

# ÿ��Сʱ������ҵ��8��Сʱ��ֹͣ
schedule.every(1).hours.until(timedelta(hours=8)).do(job)

# ÿ��Сʱ������ҵ��11:32:42��ֹͣ
schedule.every(1).hours.until(time(11, 33, 42)).do(job)

# ÿ��Сʱ������ҵ��2020-5-17 11:36:20��ֹͣ
schedule.every(1).hours.until(datetime(2020, 5, 17, 11, 36, 20)).do(job)



############################ ��������������ҵ���������䰲�����

def job_1():
    print('Foo')

def job_2():
    print('Bar')

schedule.every().monday.at("12:40").do(job_1)
schedule.every().tuesday.at("16:40").do(job_2)

schedule.run_all()

# ��������������ҵ��ÿ����ҵ���10��
schedule.run_all(delay_seconds=10)












