DROP VIEW IF EXISTS churn_features;

CREATE VIEW churn_features AS
WITH
ref AS (
	SELECT datetime(MAX(InvoiceDateIso), 'start of day') AS ref_date
	FROM sales_rfm_iso
),

first_purchase AS (
	SELECT
		CustomerID,
		MIN(InvoiceDateIso) AS FirstPurchase
	FROM sales_rfm_iso
	GROUP BY CustomerID
),


base AS (
	SELECT
		b.CustomerID,
		b.RecencyDays,
		b.Frequency,
		b.Monetary,
		ROUND(b.Monetary / NULLIF(b.Frequency, 0), 2) AS AvgOrderValue,
		fp.FirstPurchase,
		CAST(julianday((SELECT ref_date FROM ref)) - julianday(fp.FirstPurchase) AS INTEGER) AS CustomerLifetimeDays
	FROM customer_rfm_base b
	LEFT JOIN first_purchase fp
		on b.CustomerID = fp.CustomerID
),
		
scores AS (
	SELECT
		CustomerID,
		R_Score, F_Score, M_Score,
		CAST(R_Score AS TEXT) || CAST(F_Score AS TEXT) || CAST(M_Score AS TEXT) AS RFM_Code,
		ROUND(0.5*R_Score + 0.3*F_Score + 0.2*M_Score, 2) AS RFM_TotalScore
	FROM customer_rfm_scores
)


SELECT
	ba.CustomerID,
	ba.RecencyDays,
	ba.Frequency,
	ba.Monetary,
	ba.AvgOrderValue,
	ba.CustomerLifetimeDays,
	sc.R_Score, sc.F_Score, sc.M_Score,
	sc.RFM_TotalScore,
	sc.RFM_Code,
	CASE WHEN ba.RecencyDays > 90 THEN 1 ELSE 0 END AS Churn
FROM base ba
LEFT JOIN scores sc
	ON ba.CustomerID = sc.CustomerID
	
	
	
	
	
	
		
		