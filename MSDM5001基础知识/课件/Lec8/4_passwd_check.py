from hashlib import sha256

def func_change_pwd(acnt,pwd,pwd_table={}):
    sha_acnt = sha256(acnt.encode('utf-8'))
    sha_pwd = sha256(pwd.encode('utf-8'))
    if sha_acnt.hexdigest() in pwd_table:
        pwd_table[sha_acnt.hexdigest()]=sha_pwd.hexdigest()
    else:
        print(acnt,'does not exist')
    return pwd_table

def func_register(acnt,pwd,pwd_table={}):
    sha_acnt = sha256(acnt.encode('utf-8'))
    sha_pwd = sha256(pwd.encode('utf-8'))
    if sha_acnt.hexdigest() in pwd_table:
        print(acnt,'has been used, please try another one')
    else:
        pwd_table[sha_acnt.hexdigest()]=sha_pwd.hexdigest()
    return pwd_table

pwd_table={}
pwd_table=func_register('hkust','good',pwd_table)
pwd_table=func_register('DDM','very good',pwd_table)
pwd_table=func_register('student','Excellent',pwd_table)

acnt=input('Enter your account:\n')
pwd=input('Enter your password:\n')

sha_acnt = sha256(acnt.encode('utf-8'))
sha_pwd = sha256(pwd.encode('utf-8'))

if sha_acnt.hexdigest() in pwd_table:
    if pwd_table[sha_acnt.hexdigest()]==sha_pwd.hexdigest():
        print('your password is correct')
    else:
        print('your password is incorrect')
else:
    print(acnt,'does not exist!')