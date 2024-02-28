import sqlite3

# create a database
conn = sqlite3.connect('test.db')
print("Opened database successfully")

# ����һ���α��������ִ�� SQL ���
c = conn.cursor()
c.execute('''CREATE TABLE `products` (
  `product_id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `quantity_in_stock` int(11) NOT NULL,
  `unit_price` decimal(4,2) NOT NULL,
  PRIMARY KEY (`product_id`)
);''')

# �����¼
cursor.execute("INSERT INTO products VALUES (1,'Foam Dinner Plate',70,1.21)")
cursor.execute("INSERT INTO products VALUES (2,'Pork - Bacon,back Peameal',49,4.65)")
cursor.execute("INSERT INTO products VALUES (3,'Lettuce - Romaine, Heart',38,3.35)")
cursor.execute("INSERT INTO products VALUES (4,'Brocolinni - Gaylan, Chinese',90,4.53)")
cursor.execute("INSERT INTO products VALUES (5,'Sauce - Ranch Dressing',94,1.63)")
cursor.execute("INSERT INTO products VALUES (6,'Petit Baguette',14,2.39)")
cursor.execute("INSERT INTO products VALUES (7,'Sweet Pea Sprouts',98,3.29)")
cursor.execute("INSERT INTO products VALUES (8,'Island Oasis - Raspberry',26,0.74)")


# �ύ����
conn.commit()

# ��ȡ����
column_names = [description[0] for description in c.description]
print(column_names)

#ȡ����������
cursor = c.execute("select * from products ORDER BY quantity_in_stock ASC;")
for row in cursor:
    print(row)

conn.close()













