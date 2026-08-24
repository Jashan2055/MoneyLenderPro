# SQL / Business Analysis Notes

The MySQL layer was not only used as storage. It was also used to answer business questions before machine learning.

Typical business-insight categories for the project included:

1. Loan status distribution
2. Loan performance by grade
3. Loan performance by purpose
4. Risk by verification status
5. Financial / borrower characteristics associated with bad loans

One of the most useful exploratory results was the relationship between loan grade and bad-loan rate.

Observed approximate bad-loan rates:

```text
A ≈ 6%
B ≈ 13%
C ≈ 22%
D ≈ 30%
E ≈ 40%
F ≈ 45%
G ≈ 49%
```

This analysis later motivated the grade-removal experiment.

The broader lesson is:

```text
SQL → understand the business/data

Python → transform and model the data
```

The SQL layer should remain understandable independently from the ML pipeline.
