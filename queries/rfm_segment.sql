DROP TABLE IF EXISTS customer_rfm_segments;

CREATE TABLE customer_rfm_segments AS
SELECT
	s.*,
	CASE
		WHEN R_Score >= 4 AND F_Score >=4 AND M_Score >= 4 THEN 'Champions'
		WHEN R_Score >= 3 AND F_Score >= 4	THEN 'Loyal'
		WHEN R_Score >= 4 AND F_Score = 3	THEN 'Potential Loyalist'
		WHEN R_score <= 2 AND F_Score >= 3	THEN 'At Risk'
		WHEN R_Score = 1 AND F_Score <= 2 AND M_Score <= 2 THEN 'Hibernating'
		ELSE 'Regular'
	END AS Segment
FROM customer_rfm_scores s;
		