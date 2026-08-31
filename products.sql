-- Customer Credit Project: Product Data
-- Run this file in MySQL Workbench

CREATE DATABASE IF NOT EXISTS customer_credit_db;
USE customer_credit_db;

DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id VARCHAR(10) PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(100),
    interest_rate DECIMAL(5,2),
    stock INT
);

INSERT INTO products (product_id, product_name, category, interest_rate, stock) VALUES
('P001', 'Personal Loan Basic', 'Personal Loan', 10.5, 1000),
('P002', 'Personal Loan Premium', 'Personal Loan', 12.0, 800),
('P003', 'Home Loan', 'Home Loan', 8.5, 500),
('P004', 'Education Loan', 'Education Loan', 9.25, 600),
('P005', 'Business Loan', 'Business Loan', 11.75, 400),
('P006', 'Credit Builder Loan', 'Credit Loan', 14.5, 700),
('P007', 'Gold Loan', 'Secured Loan', 9.75, 300),
('P008', 'Vehicle Loan', 'Vehicle Loan', 10.0, 450);

SELECT * FROM products;
