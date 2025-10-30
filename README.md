# IPCA-Replication-Code

## 1. Overview
***The theoretical highlight of IPCA is that it ingeniously transforms an economic structure—the idea that factor loadings are driven by characteristics—into a statistical regularization constraint, thereby achieving robust dimensionality reduction, variable selection, and overfitting control in the "factor zoo" environment where predictors can far outnumber observations.***

***Verification of the performance of IPCA on different stock datasets, providing a methodological tool to replicate the IPCA asset pricing tests (Table I) and the observable factor and PCA comparisons (Table II) from "Characteristics are Covariances: A Unified Model of Risk and Return" (Kelly, Pruitt, Su, 2019, JFE)*** 

* **`IPCA_Table1_Replication.py`**: Estimates the IPCA model and tests for alpha significance ($W_{\alpha}$).
* **`IPCA_Table2_Replication.py`**: Compares IPCA against alternative models (Observable Factors, PCA).
* **`ipca.py`**: original IPCA method from third party, used for Table1 IPCA output R²（in-sample & predictive）and α for wild residual bootstrap defined in Table1 code
* **`IPCA OutofSample`**: contains two files, one is refactored core wrapper run/class of IPCA algorithm **`refactored_Table1.py`** based on base code'ipca.py', which is directly used in Replicating table 567 in file **`IPCA_Table567_OutofSample.py`**.
* **`IPCA_Table567_OutofSample.py`**: This script replicates the out-of-sample predictive performance analysis (Tables V, VI, VII) of the IPCA model from Kelly, Pruitt & Su (2019), evaluating factor models across different characteristics based on "stock size portfolios".

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

## 4. Key Implementation Details (please refers to docs_Literature review for more information)
* **Table I Panel A & B (IPCA Model):** Jointly and iteratively solves for **latent factors** and a high-dimensional **characteristic-to-loading mapping matrix($\Gamma$)** across the entire asset panel using **Alternating Least Squares (ALS)**.

* **Table I Panel C (Bootstrap Test):** Implements a custom **Wild Residual Bootstrap** function (`compute_Walpha_pval`) to generate p-values for the $W_{\alpha}$ asset pricing test by simulating returns under the null hypothesis ($\Gamma_{\alpha}=0$), ensuring methodological transparency and control.

* **Table II Panel B (Static Factor Model):** Performs a large-scale estimation of static betas for the entire cross-section by executing thousands of independent **time-series OLS regressions** for each individual stock against Fama-French factors.

* **Table II Panel C (Instrumented Factor Model):** Constructs a high-dimensional design matrix from the interaction of `L` firm characteristics and `K` observable factors, solving for the **instrument-to-beta mapping ($\Gamma_{\delta}$)** via a single, large-scale panel OLS regression.

* **Table II Panel D (PCA for Panel Data):** Implements a custom Principal Component Analysis via **Alternating Least Squares (ALS)** (`pca_als`) to handle missing observations (NaNs) in the high-dimensional stock return panel, a task where standard SVD-based PCA(principles component analysis) would fail.

**Table V: Out-of-Sample Predictive Performance**

1. **Expanding Window + Expanding Mean Instruments**: Constructs instruments using expanding mean (only historical data up to t-1) instead of full-sample mean, ensuring zero look-ahead bias in true out-of-sample prediction:

$$
\bar{c}_{i,t} = \frac{1}{t-1}\sum_{\tau=1}^{t-1}c_{i,\tau}
$$


2. **Rolling ALS Re-estimation**: Re-optimizes both Γ (characteristic-to-loading mapping) and F (latent factors) at every monthly OOS period via full Alternating Least Squares, capturing structural breaks and parameter drift.

3. **R² Decomposition Framework**: Separates Total R² (overall fit) from Predictive R² (time-varying loadings $\Gamma z_{i,t}$ only), isolating pure forward-looking predictive power by removing fixed effects $\alpha_i$.

**Table VI: Large vs Small Stock Performance**

1. **Dual Instrument Paradigm**: Contrasts In-Sample instruments (fixed full-sample mean) vs Out-of-Sample instruments (expanding mean) on identical data splits, quantifying the performance penalty from eliminating forward-looking information.

2. **Median-Split Heterogeneity Test**: Runs separate IPCA estimations on large-cap and small-cap subsamples to test whether the same K=4 latent factors generalize across different liquidity and information efficiency regimes.

3. **IS-to-OOS R² Decay Metric**: Measures $\Delta R^2 = R^2_{IS} - R^2_{OOS}$ separately for each size cohort, diagnosing model overfitting and stability differences between liquid mega-caps and illiquid micro-caps.

**Table VII: Cross-Sample Validation Matrix**

1. **2×2 Transfer Learning Design**: Trains independent models on Large/Small subsamples then cross-evaluates (L-on-S, S-on-L), pioneering out-of-distribution validation in asset pricing to test parameter portability across market segments.

2. **Feature Space Alignment Algorithm**: Constructs bijective mappings between instrument sets from different subsamples (e.g., `{char}_mean_rank` in Large → Small), enabling Γ_large to be applied to Z_small data despite different feature distributions.

3. **Off-Diagonal Generalization Test**: High R² in L-on-S and S-on-L cells validates that factor loadings learned from one market-cap regime contain transferable pricing information for the other, supporting universal factor structure hypothesis.

## 5. Background and Results support
* For IPCA and related background and Math implementation, could be found at docs/Literature_review and at docs/Results_Table1 or Results_Table2

## 6. License & Acknowledgements
* This replication code is licensed under the MIT License.
* The `ipca.py` library is used under its original MIT License, authored by Matthias Buechner and Leland Bybee.
