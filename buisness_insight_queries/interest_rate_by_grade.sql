USE micro_lending;

SELECT
    grade,
    COUNT(loan_id) AS total_loans,
    ROUND(AVG(interest_rate), 2) AS avg_interest_rate,
    ROUND(MIN(interest_rate), 2) AS min_interest_rate,
    ROUND(MAX(interest_rate), 2) AS max_interest_rate
FROM loans
GROUP BY grade
ORDER BY grade;