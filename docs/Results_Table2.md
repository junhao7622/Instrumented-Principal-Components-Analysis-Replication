# Replication of Table II from Kelly, Pruitt, Su (2019, JFE)

This script provides a replication of Table II from the Journal of Financial Economics paper, "Characteristics are covariances: A unified model of risk and return" by Kelly, Pruitt, Su (2019).

***NOTE***: The Methodology is fully reproduced in the code, which replicate the methodology from the paper, yet the dataset is different, producing **different result** but **get the same conclusion**

**Author:** Junhao Gao  
**Date:** 2025-09

Before running, you can adjust the configuration settings at the top of the script:
* `DATA_CSV`, `FF5_CSV`, `MOM_CSV`: Paths to your data files---analyzed dataset file, Fama-French 5-factor file and Momentum factor data file
* `START_DATE` / `END_DATE`: The sample period for the analysis.
* `ALS_MAX_ITER` / `ALS_TOL`: Parameters for the PCA ALS algorithm convergence.

## Interpretation align with all Conclusions from Table 2 from Kelly, Pruitt, and Su (2019)
**IPCA vs. Observable Factor Models (e.g., Fama-French 5-Factor)**:
Parsimony and Risk Explanation: IPCA achieves a comparable ability to explain total return variation (Total R 
2) while being vastly more parsimonious.

**IPCA vs. Standard PCA**:
PCA's Strength and Weakness: While PCA excels at explaining realized return variation (achieving the highest Total R 
2), it completely fails to explain average returns. For individual stocks, PCA's Predictive R2
is negative in all specifications, showing it has no explanatory power for risk compensation.

## Output

The script's final output is a formatted table printed to the console that assembles the results from Panels B, C, and D. The table is structured with a multi-level index to clearly distinguish between the different models (Observable Factors, Observable Factors with Instruments, PCA) and test assets (`r_t`, `x_t`).
# IPCA Replication Results

This repository contains replication code for **Instrumented Principal Component Analysis (IPCA)** based on Kelly, Pruitt & Su (2020).

Below are the reproduced results for Table II (Panels B–D).  

---

## Panel B: Observable Factors (no instruments)

| Asset | Statistic |    1   |    3    |    4    |    5    |    6    |
|-------|-----------|--------|---------|---------|---------|---------|
| r_t   | Total R²  |  0.59  |   0.71  |   1.61  |   3.93  |   5.03  |
|       | Pred. R²  | -1.41  | -13.80  | -13.08  |  -7.75  |  -3.19  |
|       | Np        | 3908   | 11724   | 15632   | 19540   | 23448   |
| x_t   | Total R²  |  1.48  |   1.71  |   1.86  |   2.39  |   2.48  |
|       | Pred. R²  | 23.96  |  25.63  |  15.03  |   3.76  |  -2.55  |
|       | Np        |   37   |   111   |   148   |   185   |   222   |

---

## Panel C: Observable Factors (with instruments)

| Asset | Statistic |   1   |   3   |   4   |   5   |   6   |
|-------|-----------|-------|-------|-------|-------|-------|
| r_t   | Total R²  | 0.21  | 0.44  | 0.55  | 0.58  | 0.69  |
|       | Pred. R²  | -1.52 | -2.94 | -1.65 | -2.35 | -1.41 |
|       | Np        |   37  |  111  |  148  |  185  |  222  |
| x_t   | Total R²  | 1.73  | 3.88  | 5.15  | 5.02  | 6.29  |
|       | Pred. R²  | 7.30  | 10.59 | 19.49 | 17.09 | 23.13 |
|       | Np        |   37  |  111  |  148  |  185  |  222  |

---

## Panel D: PCA (returns r_t and x_t)

| Asset | Statistic |    1    |    3    |    4    |    5    |    6    |    2    |
|-------|-----------|---------|---------|---------|---------|---------|---------|
| r_t   | Total R²  |  27.70  |  35.82  |  38.66  |  40.95  |  43.03  |  32.81  |
|       | Pred. R²  |  -0.09  | -22402182.07 | -7626433.79 | -65612349.19 | -16586366.17 | -3498204000.00 |
|       | Np        |  9508   | 28524   | 38032   | 47540   | 57048   | 19016  |
| x_t   | Total R²  |  90.17  |  96.86  |  97.39  |  97.68  |  98.16  |  95.28 |
|       | Pred. R²  |   0.00  |   0.00  |   0.00  |  -0.00  |  -0.00  |   0.00 |
|       | Np        |   648   |  1944   |  2592   |  3240   |  3888   |  1296  |

---

*All values are reproduced from IPCA replication code (Table II).*
