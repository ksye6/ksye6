use qq;

SELECT order_id, order_date, total_amount, first_name, last_name, email
FROM orders o
JOIN customers c
	ON o.customer_id = c.customer_id;

SELECT order_id, order_date, total_amount, first_name, last_name, email
FROM orders o
RIGHT JOIN customers c
	ON o.customer_id = c.customer_id;

