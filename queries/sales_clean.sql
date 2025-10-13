DROP VIEW IF EXISTS sales_clean;

CREATE VIEW sales_clean AS
SELECT
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    InvoiceDate,              
    UnitPrice,
    CustomerID,
    Country,
    (Quantity * UnitPrice) AS Revenue
FROM transactions
WHERE
    InvoiceNo NOT LIKE 'C%'
    AND Quantity  > 0
    AND UnitPrice > 0
    AND InvoiceDate IS NOT NULL;
