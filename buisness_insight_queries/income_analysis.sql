USE micro_lending;

SELECT
    CASE
        WHEN c.annual_income < 30000
            THEN '< 30K'

        WHEN c.annual_income < 60000
            THEN '30K - 60K'

        WHEN c.annual_income < 100000
            THEN '60K - 100K'

        ELSE '100K+'
    END AS income_group,

    COUNT(l.loan_id) AS total_loans,

    ROUND(
        AVG(l.loan_amount),
        2
    ) AS avg_loan_amount,

    ROUND(
        AVG(l.interest_rate),
        2
    ) AS avg_interest_rate

FROM customers c

JOIN loans l
    ON c.customer_id = l.customer_id

GROUP BY income_group

ORDER BY
    MIN(c.annual_income);