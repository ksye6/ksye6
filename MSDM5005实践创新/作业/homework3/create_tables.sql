use qq;

drop table if exists `orders`;
drop table if exists `customers`;

CREATE TABLE `orders` (
  `order_id` int(11) NOT NULL,
  `customer_id` varchar(255) DEFAULT NULL,
  `order_date` varchar(255) DEFAULT NULL,
  `total_amount` varchar(255) DEFAULT NULL
) ;

INSERT INTO `orders` (`order_id`, `customer_id`, `order_date`, `total_amount`) VALUES
(1, '101', '2023-01-15', '150.00'),
(2, '102', '2023-02-20', '210.50'),
(3, '103', '2023-03-10', '95.25');

CREATE TABLE `customers` (
  `customer_id` varchar(255) NOT NULL,
  `first_name` varchar(255) DEFAULT NULL,
  `last_name` varchar(255) DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL
) ;

INSERT INTO `customers` (`customer_id`, `first_name`, `last_name`, `email`) VALUES
('101', 'John', 'Doe', 'john.doe@email.com'),
('102', 'Jane', 'Smith', 'jane.smith@email.com'),
('103', 'Mike', 'Johnson', 'mike.johnson@email.com'),
('104', 'Jennifer', 'Williams', 'jennifer.williams@email.com');



