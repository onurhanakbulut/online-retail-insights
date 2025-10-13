DROP VIEW IF EXISTS sales_rfm;

CREATE VIEW sales_rfm AS
SELECT
    InvoiceNo,
    CustomerID,
    InvoiceDate,
    Quantity,
    UnitPrice,
    (Quantity * UnitPrice) AS Revenue,
    Country
FROM sales_clean
WHERE
    CustomerID IS NOT NULL
    AND InvoiceDate IS NOT NULL;
