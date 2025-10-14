DROP VIEW IF EXISTS churn_features;

CREATE VIEW churn_features AS
SELECT
	b.CustomerID,
	b.RecencyDays,
	b.Frequency,
	b.Monetary,
	ROUND(b.Monetary / NULLIF(b.Frequency, 0), 2) AS AvgOrderValue,
	CASE
		WHEN b.RecencyDays > 90 THEN 1
		ELSE 0
	END AS Churn
FROM customer_rfm_base b;