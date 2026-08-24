USE micro_lending;

SELECT
    grade,

    COUNT(*) AS total_loans,

    SUM(
        CASE
            WHEN loan_status IN (
                'Charged Off',
                'Default',
                'Late (16-30 days)',
                'Late (31-120 days)',
                'In Grace Period',
                'Does not meet the credit policy. Status:Charged Off'
            )
            THEN 1
            ELSE 0
        END
    ) AS risky_loans,

    ROUND(
        SUM(
            CASE
                WHEN loan_status IN (
                    'Charged Off',
                    'Default',
                    'Late (16-30 days)',
                    'Late (31-120 days)',
                    'In Grace Period',
                    'Does not meet the credit policy. Status:Charged Off'
                )
                THEN 1
                ELSE 0
            END
        ) * 100.0 / COUNT(*),
        2
    ) AS risk_rate

FROM loans

GROUP BY grade

ORDER BY risk_rate DESC;