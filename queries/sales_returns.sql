DROP VIEW IF EXISTS returns;

CREATE VIEW returns AS
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
    (InvoiceNo LIKE 'C%' OR Quantity < 0)
    AND InvoiceDate IS NOT NULL;
