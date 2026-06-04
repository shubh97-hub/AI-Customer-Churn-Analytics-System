
CREATE DATABASE CustomerChurnDB;

USE CustomerChurnDB;

CREATE TABLE customer_churn (
    CustomerID VARCHAR(20),
    Gender VARCHAR(10),
    Age INT,
    TenureMonths INT,
    MonthlyCharges DECIMAL(10,2),
    TotalCharges DECIMAL(10,2),
    InternetService VARCHAR(50),
    ContractType VARCHAR(50),
    PaymentMethod VARCHAR(50),
    SupportTickets INT,
    LatePayments INT,
    Churn VARCHAR(10)
);

-- Total Customers
SELECT COUNT(*) AS Total_Customers
FROM customer_churn;


-- Churn Rate
SELECT 
    Churn,
    COUNT(*) AS Customer_Count
FROM customer_churn
GROUP BY Churn;


-- Highest Churn by Contract Type
SELECT 
    ContractType,
    COUNT(*) AS Churned_Customers
FROM customer_churn
WHERE Churn = 'Yes'
GROUP BY ContractType
ORDER BY Churned_Customers DESC;


-- Monthly Revenue
SELECT 
    SUM(MonthlyCharges) AS Monthly_Revenue
FROM customer_churn;


-- Top Risk Customers
SELECT *
FROM customer_churn
WHERE SupportTickets > 5
AND LatePayments > 3
AND Churn = 'Yes';
