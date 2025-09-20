# IPCA-Replication-Code

## 1. Overview
***The theoretical highlight of IPCA is that it ingeniously transforms an economic structure—the idea that factor loadings are driven by characteristics—into a statistical regularization constraint, thereby achieving robust dimensionality reduction, variable selection, and overfitting control in the "factor zoo" environment where predictors can far outnumber observations.***

***methodological replication***This repository contains Python scripts to replicate Tables I and II from the paper 
**"Characteristics are covariances: A unified model of risk and return"** (Kelly, Pruitt, Su, 2019, JFE). 

* **`IPCA_Table1_Replication.py`**: Estimates the IPCA model and tests for alpha significance ($W_{\alpha}$).
* **`IPCA_TABLE2_Replication.py`**: Compares IPCA against alternative models (Observable Factors, PCA).

## 2. Prerequisites
* **see requirement.txt** Python3.X required

## 3. How to Run
1.  Adjust settings (e.g., `BOOTSTRAP_DRAWS`, date ranges) at the top of each script.
2.  Execute from the terminal:
    ```bash
    # To replicate Table I
    python IPCA_Table1_Replication.py

    # To replicate Table II
    python IPCA_TABLE2_Replication.py
    ```

## 4. Key Implementation Details
* **Table1 Panel A & B (IPCA Model):** Jointly and iteratively solves for **latent factors** and a high-dimensional **characteristic-to-loading mapping matrix ($\Gamma$)** across the entire asset panel using **Alternating Least Squares (ALS)**.

* **Table1 Panel C (Bootstrap Test):** Implements a custom **Wild Residual Bootstrap** function (`compute_Walpha_pval`) to generate p-values for the $W_{\alpha}$ asset pricing test by simulating returns under the null hypothesis ($\Gamma_{\alpha}=0$), ensuring methodological transparency and control.

* **Table2 Panel B (Static Factor Model):** Performs a large-scale estimation of static betas for the entire cross-section by executing thousands of independent **time-series OLS regressions** for each individual stock against Fama-French factors.

* **Table2 Panel C (Instrumented Factor Model):** Constructs a high-dimensional design matrix from the interaction of `L` firm characteristics and `K` observable factors, solving for the **instrument-to-beta mapping ($\Gamma_{\delta}$)** via a single, large-scale panel OLS regression.

* **Table2 Panel D (PCA for Panel Data):** Implements a custom Principal Component Analysis via **Alternating Least Squares (ALS)** (`pca_als`) to handle missing observations (NaNs) in the high-dimensional stock return panel, a task where standard SVD-based PCA would fail.

## 5. Background and Results support
* For IPCA and related background and Math implementation, could be found at docs/Literature_review and at docs/Results_Table1 or Results_Table2

## 6. License & Acknowledgements
* This replication code is licensed under the MIT License.
* The `ipca.py` library is used under its original MIT License, authored by Matthias Buechner and Leland Bybee.
