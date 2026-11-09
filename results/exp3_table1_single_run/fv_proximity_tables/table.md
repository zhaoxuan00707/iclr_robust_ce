# Future validity and proximity

Each cell is **FV (%) / proximity**. FV is the saved mean classifier future validity over all factuals and45 test scenarios; proximity is mean factual-to-CF L2 in the standardized feature representation (lower is closer). One seed; no standard errors. These are reconstructed diagnostics, not a verified reproduction of historical Table1. HELOC and FICO represent the same underlying corpus.

GRACE rows use relaxed simplex supports for German/FICO; these are not legal discrete recourse. Separate hard rows report argmax realization. “Infeasible” means the hard result violates geometry; its raw diagnostic is retained in the CSV. “Failed” means all native attempts failed; fallback factuals are not reported as successful CFs. “Unsupported” indicates no categorical implementation.

* T-Rex has9/128 native failures on HELOC/MLP and5/128 on FICO/MLP. Displayed FV and proximity retain all cases, including fallback factuals. Failure-aware FV is92.97% and96.09%, respectively (raw FICO FV96.34%).

† German/logistic ElliCE has100% future validity but0/43 native ellipsoid-robust outputs under the original selected epsilon.05. The task101 epsilon.01 follow-up is separate and does not replace this table.


## MLP classifier

| Method               | HELOC          | Diabetes       | German          | FICO           |
|:---------------------|:---------------|:---------------|:----------------|:---------------|
| Wachter Base         | 77.86 / 0.991  | 73.29 / 1.102  | 58.68 / 1.998   | 79.31 / 1.086  |
| GRACE–Wachter        | 99.90 / 1.242  | 99.96 / 1.835  | 100.00 / 2.588  | 100.00 / 1.510 |
| GRACE–Wachter (hard) | —              | —              | 96.58 / 2.431   | 99.93 / 1.502  |
| T-Rex                | 92.97 / 3.646* | 100.00 / 3.161 | 100.00 / 9.126  | 96.34 / 4.713* |
| ROAR                 | 84.18 / 2.890  | 78.58 / 2.130  | Unsupported     | Unsupported    |
| PROPLACE             | Failed         | Failed         | Unsupported     | Unsupported    |
| ElliCE               | 100.00 / 2.458 | 100.00 / 2.476 | 100.00 / 10.825 | 100.00 / 6.214 |
| DiCE Base            | 92.26 / 4.306  | 94.98 / 2.785  | 88.29 / 3.851   | 94.93 / 5.665  |
| GRACE–DiCE           | 100.00 / 4.332 | 99.91 / 2.958  | 99.94 / 3.957   | 100.00 / 5.730 |
| GRACE–DiCE (hard)    | —              | —              | 98.02 / 3.929   | 100.00 / 5.728 |


## Logistic regression

| Method               | HELOC          | Diabetes       | German           | FICO           |
|:---------------------|:---------------|:---------------|:-----------------|:---------------|
| Wachter Base         | 90.14 / 1.625  | 85.11 / 0.928  | 59.64 / 1.769    | 84.34 / 1.766  |
| GRACE–Wachter        | 99.53 / 2.004  | 100.00 / 1.405 | 99.33 / 2.099    | 98.91 / 2.091  |
| GRACE–Wachter (hard) | —              | —              | 91.16 / 1.997    | Infeasible‡    |
| T-Rex                | 100.00 / 4.400 | 100.00 / 2.652 | 100.00 / 8.788   | 100.00 / 4.888 |
| ROAR                 | 56.28 / 2.870  | 91.44 / 1.941  | Unsupported      | Unsupported    |
| PROPLACE             | 99.06 / 3.320  | 100.00 / 2.316 | Unsupported      | Unsupported    |
| ElliCE               | 100.00 / 3.797 | 100.00 / 2.031 | 100.00 / 10.420† | 100.00 / 6.110 |


‡ FICO/logistic hard GRACE–Wachter: raw diagnostic98.87% /2.090, but violates the geometry budget.


Source: `../combined_summary.csv`. Original raw outputs are unchanged.
