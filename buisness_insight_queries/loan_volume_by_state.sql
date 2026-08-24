USE micro_lending;

SELECT
    c.addr_state,
    COUNT(l.loan_id) AS total_loans,
    SUM(l.loan_amount) AS total_loan_amount,
    AVG(l.loan_amount) AS average_loan_amount
FROM customers c
JOIN loans l
    ON c.customer_id = l.customer_id
GROUP BY c.addr_state
ORDER BY total_loans DESC
LIMIT 10;