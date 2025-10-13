DROP TABLE IF EXISTS customer_rfm_base;

CREATE TABLE customer_rfm_base AS
WITH ref AS (
  SELECT datetime(MAX(InvoiceDateIso), 'start of day') AS ref_date
  FROM sales_rfm_iso
),
agg AS (
  SELECT
    CustomerID,
    MAX(InvoiceDateIso)   AS LastPurchase,
    COUNT(DISTINCT InvoiceNo)   AS Frequency,
    SUM(Quantity * UnitPrice)   AS Monetary
  FROM sales_rfm_iso
  GROUP BY CustomerID
)
SELECT
  a.CustomerID,
  a.LastPurchase,
  CAST( (strftime('%s', (SELECT ref_date FROM ref)) - strftime('%s', a.LastPurchase)) / 86400 AS INTEGER ) AS RecencyDays,  --strftime unix zamanina çevirir / gün
  a.Frequency,
  ROUND(a.Monetary, 2) AS Monetary
FROM agg a;
