DROP TABLE IF EXISTS customer_rfm_scores;

CREATE TABLE customer_rfm_scores AS
WITH s AS (
  SELECT
    CustomerID,
    RecencyDays,
    Frequency,
    Monetary,
    NTILE(5) OVER (ORDER BY RecencyDays DESC) AS R_Score,
    NTILE(5) OVER (ORDER BY Frequency ASC)         AS F_Score,
    NTILE(5) OVER (ORDER BY Monetary  ASC)         AS M_Score
  FROM customer_rfm_base
)
SELECT
  CustomerID, RecencyDays, Frequency, Monetary,
  R_Score, F_Score, M_Score,
  CAST(R_Score AS TEXT) || CAST(F_Score AS TEXT) || CAST(M_Score AS TEXT) AS RFM_Code
FROM s;
