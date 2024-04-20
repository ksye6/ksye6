import schedule
import time

############################ 基本使用  https://zhuanlan.zhihu.com/p/501021369

def job():
    print("I'm working...")

# 每十分钟执行任务
schedule.every(10).minutes.do(job)
# 每个小时执行任务
schedule.every().hour.do(job)
# 每天的10:30执行任务
schedule.every().day.at("10:30").do(job)
# 每个月执行任务
schedule.every().monday.do(job)
# 每个星期三的13:15分执行任务
schedule.every().wednesday.at("13:15").do(job)
# 每分钟的第17秒执行任务
schedule.every().minute.at(":17").do(job)


schedule.every(2).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)

############################ 基本使用

def job_that_executes_once():
    # 此处编写的任务只会执行一次...
    return schedule.CancelJob

schedule.every().day.at('22:30').do(job_that_executes_once)

while True:
    schedule.run_pending()
    time.sleep(1)


############################ 参数传递


def greet(name):
    print('Hello', name)

# do() 将额外的参数传递给job函数
schedule.every(2).seconds.do(greet, name='Alice')
schedule.every(4).seconds.do(greet, name='Bob')


############################ 获取目前所有的作业

schedule.get_jobs()

############################ 取消所有作业


schedule.clear()


############################ 通过标签过滤获取作业或取消作业

def greet(name):
    print('Hello {}'.format(name))

# .tag 打标签
schedule.every().day.do(greet, 'Andrea').tag('daily-tasks', 'friend')
schedule.every().hour.do(greet, 'John').tag('hourly-tasks', 'friend')
schedule.every().hour.do(greet, 'Monica').tag('hourly-tasks', 'customer')
schedule.every().day.do(greet, 'Derek').tag('daily-tasks', 'guest')

# get_jobs(标签)：可以获取所有该标签的任务
friends = schedule.get_jobs('friend')

# 取消所有 daily-tasks 标签的任务
schedule.clear('daily-tasks')


############################ 设定作业截止时间

def job():
    print('Boo')

# 每个小时运行作业，18:30后停止
schedule.every(1).hours.until("18:30").do(job)

# 每个小时运行作业，2030-01-01 18:33 today
schedule.every(1).hours.until("2030-01-01 18:33").do(job)

# 每个小时运行作业，8个小时后停止
schedule.every(1).hours.until(timedelta(hours=8)).do(job)

# 每个小时运行作业，11:32:42后停止
schedule.every(1).hours.until(time(11, 33, 42)).do(job)

# 每个小时运行作业，2020-5-17 11:36:20后停止
schedule.every(1).hours.until(datetime(2020, 5, 17, 11, 36, 20)).do(job)



############################ 立即运行所有作业，而不管其安排如何

def job_1():
    print('Foo')

def job_2():
    print('Bar')

schedule.every().monday.at("12:40").do(job_1)
schedule.every().tuesday.at("16:40").do(job_2)

schedule.run_all()

# 立即运行所有作业，每次作业间隔10秒
schedule.run_all(delay_seconds=10)












