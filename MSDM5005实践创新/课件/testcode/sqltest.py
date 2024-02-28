import sqlite3

# create a database
conn = sqlite3.connect('test.db')
print("Opened database successfully")

# 创建一个游标对象，用于执行 SQL 语句
c = conn.cursor()
c.execute('''CREATE TABLE `products` (
  `product_id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `quantity_in_stock` int(11) NOT NULL,
  `unit_price` decimal(4,2) NOT NULL,
  PRIMARY KEY (`product_id`)
);''')

# 插入记录
cursor.execute("INSERT INTO products VALUES (1,'Foam Dinner Plate',70,1.21)")
cursor.execute("INSERT INTO products VALUES (2,'Pork - Bacon,back Peameal',49,4.65)")
cursor.execute("INSERT INTO products VALUES (3,'Lettuce - Romaine, Heart',38,3.35)")
cursor.execute("INSERT INTO products VALUES (4,'Brocolinni - Gaylan, Chinese',90,4.53)")
cursor.execute("INSERT INTO products VALUES (5,'Sauce - Ranch Dressing',94,1.63)")
cursor.execute("INSERT INTO products VALUES (6,'Petit Baguette',14,2.39)")
cursor.execute("INSERT INTO products VALUES (7,'Sweet Pea Sprouts',98,3.29)")
cursor.execute("INSERT INTO products VALUES (8,'Island Oasis - Raspberry',26,0.74)")


# 提交更改
conn.commit()

# 获取列名
column_names = [description[0] for description in c.description]
print(column_names)

#取得所有资料
cursor = c.execute("select * from products ORDER BY quantity_in_stock ASC;")
for row in cursor:
    print(row)

conn.close()













