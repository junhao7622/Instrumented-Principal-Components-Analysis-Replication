# IPCA-Replication-Code

## 1. Overview
***The theoretical highlight of IPCA is that it ingeniously transforms an economic structure—the idea that factor loadings are driven by characteristics—into a statistical regularization constraint, thereby achieving robust dimensionality reduction, variable selection, and overfitting control in the "factor zoo" environment where predictors can far outnumber observations.***

***Verification of the performance of IPCA on different stock datasets, providing a methodological tool to replicate the IPCA asset pricing tests (Table I) and the observable factor and PCA comparisons (Table II) from "Characteristics are Covariances: A Unified Model of Risk and Return" (Kelly, Pruitt, Su, 2019, JFE)*** 

* **`IPCA_Table1_Replication.py`**: Estimates the IPCA model and tests for alpha significance ($W_{\alpha}$).
* **`IPCA_TABLE2_Replication.py`**: Compares IPCA against alternative models (Observable Factors, PCA).
* **`ipca.py`**: original IPCA method from third party, used for Table1 IPCA output R²（in-sample & predictive）and α for wild residual bootstrap defined in Table1 code

## 2. Prerequisites
* **See requirement.txt**
*  Python3.X required
*  The replication of IPCA relies on a combination of proprietary and public datasets. Monthly stock returns are obtained from CRSP and merged with firm fundamentals from Compustat using the CRSP/Compustat Merged (CCM) link table. These data provide excess returns and approximately 36 firm characteristics that serve as instruments for factor loadings **See exactly columns mentioned in(Kelly, Pruitt, Su, 2019, JFE)**. In addition, standard benchmark factors (MKT, SMB, HML, RMW, CMA, UMD) are taken from Kenneth French’s data library for comparison and robustness checks. Due to licensing restrictions, CRSP and Compustat data are not included in this repository and must be accessed via WRDS. **Open data file for more information**

## 3. How to Run
1.  Adjust settings (e.g., `BOOTSTRAP_DRAWS`, date ranges) at the top of each script.
2.  Data ready and ipca.py file in root directory
3.  Execute from the terminal:
    ```bash
    # To replicate Table I
    python IPCA_Table1_Replication.py

    # To replicate Table II
    python IPCA_TABLE2_Replication.py
    ```

## 4. Key Implementation Details
* **Table1 Panel A & B (IPCA Model):** Jointly and iteratively solves for **latent factors** and a high-dimensional **characteristic-to-loading mapping matrix $(\Gamma)$ ** across the entire asset panel using **Alternating Least Squares (ALS)**.

* **Table1 Panel C (Bootstrap Test):** Implements a custom **Wild Residual Bootstrap** function (`compute_Walpha_pval`) to generate p-values for the $W_{\alpha}$ asset pricing test by simulating returns under the null hypothesis ($\Gamma_{\alpha}=0$), ensuring methodological transparency and control.

* **Table2 Panel B (Static Factor Model):** Performs a large-scale estimation of static betas for the entire cross-section by executing thousands of independent **time-series OLS regressions** for each individual stock against Fama-French factors.

* **Table2 Panel C (Instrumented Factor Model):** Constructs a high-dimensional design matrix from the interaction of `L` firm characteristics and `K` observable factors, solving for the **instrument-to-beta mapping ($\Gamma_{\delta}$)** via a single, large-scale panel OLS regression.

* **Table2 Panel D (PCA for Panel Data):** Implements a custom Principal Component Analysis via **Alternating Least Squares (ALS)** (`pca_als`) to handle missing observations (NaNs) in the high-dimensional stock return panel, a task where standard SVD-based PCA would fail.

## 5. Background and Results support
* For IPCA and related background and Math implementation, could be found at docs/Literature_review and at docs/Results_Table1 or Results_Table2

## 6. License & Acknowledgements
* This replication code is licensed under the MIT License.
* The `ipca.py` library is used under its original MIT License, authored by Matthias Buechner and Leland Bybee.
