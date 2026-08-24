USE micro_lending;

SELECT
    loan_status,
    COUNT(*) AS total_loans,
    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM loans),
        2
    ) AS percentage
FROM loans
GROUP BY loan_status
ORDER BY total_loans DESC;