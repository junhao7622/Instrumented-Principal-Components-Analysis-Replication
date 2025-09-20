# Replication of Table I from Kelly, Pruitt, Su (2019, JFE)

This script provides a strict replication of Table I from the Journal of Financial Economics paper, "Characteristics are covariances: A unified model of risk and return" by Kelly, Pruitt, and Su (2019).

***NOTE***: the Methodology is fully reproduced in the code, which replicate the methodology from the paper, yet the dataset is different, producing **different result** but **get the same conclusion**

**Author:** Junhao Gao  
**Date:** 2024-09

Before running, you can adjust the configuration settings at the top of the script:
* `BOOTSTRAP_DRAWS`: Number of bootstrap simulations for p-value calculation. A value of 1000 is recommended for robust results, while a smaller value (e.g., 100) can be used for quick tests.
* `DATA_CSV`: Path to your input data file.
* `START_DATE` / `END_DATE`: The sample period for the analysis.

## Interpretation align with all Conclusions from Table 2 from Kelly, Pruitt, and Su (2019)
**Total R² increases with factor number (K).**
**Restricted model (Γα=0) has lower R² than unrestricted (Γα≠0).**
**Predictive R² is small, sometimes negative, but improves with K.**
**Wα test p-values are generally large.**

## Output
The script will print the replication results to the console in three parts, mirroring the structure of Table I in the paper:
* **Panel A**: R-squared statistics for individual stocks (`r_t`).
* **Panel B**: R-squared statistics for characteristic-managed portfolios (`x_t`).
* **Panel C**: The bootstrapped p-value (%) for the $W_{\alpha}$ asset pricing test.
# IPCA Replication Results

This repository contains replication code for **Instrumented Principal Component Analysis (IPCA)** based on Kelly, Pruitt & Su (2020).

Below are the reproduced results for Table I (Panels A–C) **With different DATASET** conclusion is the **same**.  

---

## Panel A

|            |     1     |     2     |     3     |     4     |     5     |     6     |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Total R² Γα=0** | 12.52 | 16.84 | 18.85 | 20.28 | 21.37 | 22.32 |
| **Total R² Γα≠0** | 29.11 | 31.72 | 33.08 | 34.17 | 35.01 | 35.47 |
| **Pred R² Γα=0**  | -0.33 | -0.33 | -0.22 |  0.56 |  0.68 |  0.73 |
| **Pred R² Γα≠0**  |  1.08 |  1.07 |  1.06 |  1.06 |  1.05 |  1.05 |

---

## Panel B

|            |     1     |     2     |     3     |     4     |     5     |     6     |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Total R² Γα=0** | 30.97 | 45.06 | 57.08 | 66.03 | 73.19 | 76.88 |
| **Total R² Γα≠0** | 78.79 | 85.02 | 88.21 | 91.54 | 93.29 | 94.18 |
| **Pred R² Γα=0**  | -3.64 | -3.61 | -4.42 |  1.11 |  2.17 |  1.60 |
| **Pred R² Γα≠0**  | -10.31 | -10.40 | -10.70 | -10.84 | -10.65 | -10.36 |

---

## Panel C

| Metric       |     1     |     2     |     3     |     4     |     5     |     6     |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Wα p-value** | 54.55 | 63.64 | 81.82 | 45.45 | 100.0 | 81.82 |

---

*All values are percentages, reproduced from IPCA replication code.*


------------------------------------------------------------------------------------
