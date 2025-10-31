# Instrumented Principal Component Analysis (IPCA): A Comprehensive Statistical and Implementation Thought

Based on: Kelly, Pruitt, and Su (2019) "Characteristics are covariances: A unified model of risk and return"

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Statistical Framework](#2-statistical-framework)
3. [Data Preprocessing Module](#3-data-preprocessing-module)
4. [Instrument Construction Module](#4-instrument-construction-module)
5. [IPCA Estimation Algorithm](#5-ipca-estimation-algorithm)
6. [Model Evaluation Module](#6-model-evaluation-module)
7. [Statistical Testing Module](#7-statistical-testing-module)
8. [Out-of-Sample Prediction Module](#8-out-of-sample-prediction-module)
9. [Advanced Topics](#9-advanced-topics)

## 1. Theoretical Foundation

### 1.1 The Asset Pricing Problem

#### The Factor Zoo Challenge

Modern asset pricing faces a fundamental dilemma: hundreds of firm characteristics have been shown to predict returns, creating a "factor zoo" with unclear economic interpretation and severe multiple testing concerns.

**Traditional Approach Problems:**

- **Specification uncertainty:** Which characteristics matter?
- **Multicollinearity:** Characteristics are highly correlated
- **Data mining:** Many "factors" may be spurious
- **Overfitting:** In-sample performance doesn't translate out-of-sample

#### IPCA's Solution

IPCA provides a unified statistical framework that:

- Extracts latent risk factors from the cross-section of returns
- Models factor loadings as functions of firm characteristics
- Provides a parsimonious representation of return predictability
- Enables robust out-of-sample prediction

### 1.2 Mathematical Setup

#### Return Generating Process

The core model specifies excess returns as:

$$r_{i,t+1} = \alpha_i + \beta'_{i,t} f_{t+1} + \varepsilon_{i,t+1}$$

Where:

- $r_{i,t+1}$: Excess return of asset $i$ at time $t+1$
- $\alpha_i$: Asset-specific intercept (pricing error/abnormal return)
- $\beta_{i,t}$: $K \times 1$ vector of factor loadings for asset $i$ at time $t$
- $f_{t+1}$: $K \times 1$ vector of latent factors at time $t+1$
- $\varepsilon_{i,t+1}$: Idiosyncratic error term

**Key Innovation:** Factor loadings are time-varying and modeled as:

$$\beta_{i,t} = \Gamma' z_{i,t}$$

Where:

- $z_{i,t}$: $L \times 1$ vector of instruments (functions of characteristics)
- $\Gamma$: $K \times L$ matrix mapping characteristics to loadings
- $K$: Number of factors (typically 1-6)
- $L$: Number of instruments (typically 73 for 36 characteristics)

#### Compact Representation

Substituting the loading specification:

$$r_{i,t+1} = \alpha_i + (\Gamma' z_{i,t})' f_{t+1} + \varepsilon_{i,t+1} = \alpha_i + z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

This reveals the fundamental insight: **characteristics determine covariances with factors**.

### 1.3 Economic Interpretation

#### What IPCA Estimates

**Latent Factors** ($f_t$): Systematic risk sources

- Not directly observable (unlike Fama-French factors)
- Extracted from cross-sectional return variation
- Orthogonal by construction (PCA property)

**Characteristics-to-Loading Map** ($\Gamma$):

- Shows which characteristics capture which risk exposures
- $K \times L$ matrix where each row corresponds to a factor
- Economic interpretation: how firm attributes predict factor exposures

**Abnormal Returns** ($\alpha_i$):

- Asset-specific pricing errors
- Should be zero under no-arbitrage
- Statistical significance tested via Wald test

#### Connection to Classical Models

**CAPM** ($K=1, L=1$): 

- $\beta_{i,t} = \text{constant}$ (market beta)
- $f_t = \text{market excess return}$

**Fama-French** ($K=3$, fixed portfolios):

- Factors: MKT, SMB, HML
- Loadings: Time-invariant, estimated via time-series regression

**IPCA Generalization:**

- $K$: Data-driven (cross-validation)
- $L$: High-dimensional characteristics
- Loadings: Time-varying, characteristic-predicted

## 2. Statistical Framework

### 2.1 Dimensionality and Identification

#### The Challenge: High-Dimensional Problem

**Panel Structure:**

- $N$ assets (stocks): typically 5,000-20,000
- $T$ time periods: typically 600+ months
- $L$ characteristics: 36 in the paper
- Total observations: $N \times T \approx$ millions

**Curse of Dimensionality:**

- Unrestricted model: $N \times K + T \times K$ parameters
- With $N=10,000, K=5$: 50,000+ parameters
- Infeasible without structure

#### IPCA's Dimension Reduction

**Key Constraint:** Loadings are linear in characteristics:

$$\beta_{i,t} = \Gamma' z_{i,t}$$

**Parameter Count Reduction:**

- Original: $N \times K$ asset-specific loadings
- IPCA: $K \times L$ entries in $\Gamma$
- Reduction ratio: $N/L \approx 300:1$

For $K=5, L=73$:

- Traditional: 50,000 parameters
- IPCA: 365 parameters
- 137× reduction

### 2.2 Identification Strategy

#### Normalization Constraints

**Problem:** Factor models have rotational invariance

- For any invertible $K \times K$ matrix $R$: $(\Gamma R, R^{-1}f)$ fits equally well
- Need constraints for unique solution

**IPCA Normalizations:**

1. **Factor Orthogonality:** $E[f_t f'_t] = I_K$ (identity matrix)
2. **Factor Ordering by Variance:** $\text{Var}(f_1) \geq \text{Var}(f_2) \geq \cdots \geq \text{Var}(f_K)$
3. **Characteristic Scaling:** Instruments $z_{i,t}$ are rank-transformed to $[-0.5, 0.5]$

#### Two Model Variants

**Restricted Model** ($\Gamma_\alpha = 0$):

$$r_{i,t+1} = z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

- No intercept term
- Assumes all expected returns explained by factors
- $K \times L$ parameters

**Unrestricted Model** ($\Gamma_\alpha \neq 0$):

$$r_{i,t+1} = \alpha_i + z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

- Includes asset-specific intercepts
- Tests for pricing errors
- $K \times L + N$ parameters (but $\alpha_i$ not directly estimated)

### 2.3 Instrumental Variable Interpretation

#### Why "Instrumented" PCA?

**Standard PCA Issue:** $r_t = \Lambda f_t + \varepsilon_t$

- Loadings $\Lambda$ are asset-specific
- Cannot predict out-of-sample for new assets
- No economic interpretation

**IPCA Solution:** $r_{i,t} = (\Gamma' z_{i,t})' f_t + \varepsilon_{i,t}$

- Loadings are predicted by characteristics
- Characteristics serve as instruments for exposures
- Out-of-sample prediction possible

#### Formal IV Structure

Think of the two-stage process:

1. **Stage 1 (Implicit):** $\beta_{i,t} = \Gamma' z_{i,t} + \nu_{i,t}$ where $z_{i,t}$ instruments for $\beta_{i,t}$
2. **Stage 2:** $r_{i,t+1} = \beta_{i,t}'f_{t+1} + \varepsilon_{i,t+1}$
3. **Combined:** $r_{i,t+1} = (\Gamma' z_{i,t})' f_{t+1} + \text{residual}$

### 2.4 Comparison with Related Methods

#### vs. Standard PCA

| Aspect | Standard PCA | IPCA |
|--------|-------------|------|
| Loadings | Asset-specific ($N \times K$) | Characteristic-predicted ($K \times L$) |
| Factors | Orthogonal | Orthogonal |
| Parameters | $N \times K + T \times K$ | $K \times L + T \times K$ |
| Out-of-sample | Not possible | Possible via $z_{i,t}$ |
| Interpretation | Statistical only | Economic (via characteristics) |

#### vs. Fama-MacBeth

| Aspect | Fama-MacBeth | IPCA |
|--------|--------------|------|
| Factors | Pre-specified | Estimated |
| Loadings | Time-invariant | Time-varying |
| Standard errors | Two-stage | Bootstrap |
| Efficiency | Less efficient | More efficient |

#### vs. Instrumented Factor Models

**Lettau & Pelger (2020) - PCA with Covariates:**

- Similar idea: characteristics predict loadings
- Difference: IPCA uses full instrument set, LP uses principal components of characteristics

**Kozak, Nagel & Santosh (2020) - SDF approach:**

- Prices kernel directly
- IPCA focuses on return prediction
- Complementary methodologies

## 3. Data Preprocessing Module

### 3.1 Sample Construction

#### Temporal and Cross-Sectional Filters

Implementation:

```python
# Date filtering
START_DATE = "1963-07-01"
END_DATE = "2014-05-31"
df = df[(df['eom'] >= pd.to_datetime(START_DATE)) & 
        (df['eom'] <= pd.to_datetime(END_DATE))]

# Geographic filter
if 'excntry' in df.columns:
    df = df[df['excntry'] == 'USA'].copy()

# Industry filter (exclude financials)
if 'sic' in df.columns:
    df['sic'] = pd.to_numeric(df['sic'], errors='coerce')
    df = df[(df['sic'] < 6000) | (df['sic'] > 6999)].copy()
```

**Statistical Rationale:**

**Start Date (1963-07):**

- CRSP data quality improves post-1963
- Sufficient history for characteristic calculation
- Standard in asset pricing literature

**Geographic Restriction (USA only):**

- Homogeneous market structure
- Currency consistency
- Regulatory regime uniformity

**Exclude Financials (SIC 6000-6999):**

- Different accounting standards
- Leverage has different meaning
- Regulatory capital requirements distort characteristics

#### Return Specification

**Excess Returns:**

```python
y = df['ret_exc_lead1m']  # r_{i,t+1} - r^f_{t+1}
```

**Why Excess Returns?**

- **Theoretical:** Factor models price excess returns
- **Stationarity:** Risk-free rate trends, excess returns don't
- **International:** Handles cross-country differences in rates

**Lead Timing:**

- Characteristics at time $t$
- Returns from $t$ to $t+1$
- Ensures no look-ahead bias

### 3.2 Winsorization

#### Purpose and Statistical Theory

**Problem:** Extreme outliers distort estimates

- **Computational:** Numerical instability in matrix operations
- **Statistical:** Heavy-tailed return distributions
- **Economic:** Data errors, corporate actions

#### Winsorization vs. Trimming

```python
def winsorize_series(s):
    q = s.quantile([0.01, 0.99])  # 1st and 99th percentiles
    return np.clip(s, q.iloc[0], q.iloc[1])

# Applied cross-sectionally
y_w = y.groupby(level='eom').transform(winsorize_series)
X_w = X_raw.groupby(level='eom').transform(winsorize_series)
```

**Statistical Properties:**

**Preserves Sample Size:**

- **Trimming:** Deletes observations
- **Winsorization:** Replaces extremes
- Important for balanced panel

**Bias-Variance Tradeoff:**

- **Bias:** Introduces slight downward bias in variance
- **Variance:** Dramatically reduces estimator variance
- **Net effect:** Lower MSE

**Robustness:**

- Limits influence of single observations
- M-estimator property
- Asymptotically equivalent to trimming

#### Cross-Sectional vs. Time-Series Winsorization

**Paper Implementation: Cross-Sectional**

```python
# Per date, within cross-section
y_w = y.groupby(level='eom').transform(winsorize_series)
```

**Why Cross-Sectional?**

- Accounting periods differ across firms
- Event studies would be affected by time-series
- Cross-sectional comparability is key

### 3.3 Rank Transformation

#### The Central Data Transformation

Implementation:

```python
def rank_transform(panel):
    return panel.groupby(level='eom').rank(
        pct=True, 
        na_option='keep'
    ) - 0.5

X_rank = rank_transform(X_w)
```

**Step-by-Step Process:**

1. **Within Each Date:**
   - Raw values: $[15.2, 8.1, 22.7, 11.3, \text{NaN}, 18.9]$

2. **Compute Ranks:**
   - Ranks: $[3, 1, 5, 2, \text{NaN}, 4]$

3. **Convert to Percentiles:**
   - Percentiles = $[3/5, 1/5, 5/5, 2/5, \text{NaN}, 4/5] = [0.60, 0.20, 1.00, 0.40, \text{NaN}, 0.80]$

4. **Center at Zero:**
   - Final = $[0.10, -0.30, 0.50, -0.10, \text{NaN}, 0.30]$
   - Subtracting 0.5 centers the distribution at zero

#### Statistical Rationale

**1. Non-Parametric Transformation:**

- Invariant to monotonic transformations
- Eliminates distributional assumptions
- Comparable across characteristics with different units

**2. Outlier Robustness:**

| Type | Original | Z-score | Rank |
|------|----------|---------|------|
| Values | $[-100, 5, 10, 15, 1000]$ | $[-4.5, -0.1, 0, 0.1, 4.5]$ | $[-0.4, -0.1, 0, 0.1, 0.4]$ |
| Note | Extremes raw | Outliers still extreme | Bounded in $[-0.5, 0.5]$ |

**3. Cross-Sectional Uniformity:**

- Each date: Uniform(-0.5, 0.5) distribution
- Time-invariant range
- Numerical stability in optimization

**4. Economic Interpretation:**

- Positive value = above median
- Magnitude = percentile distance from median
- Scale: $-0.5$ (worst) to $+0.5$ (best)

#### Why Not Z-Scores?

**Z-score Alternative:**

$$z_{i,t} = \frac{c_{i,t} - \bar{c}_t}{\sigma_{c,t}}$$

**Drawbacks:**

- **Outlier sensitivity:** $\sigma$ affected by extremes
- **Time-varying scale:** $\sigma_t$ changes over time
- **Normality assumption:** Implicitly assumes Gaussian tails
- **Missing data:** Mean/std sensitive to missing pattern

**Rank advantages:**

- Distribution-free
- Bounded domain $[-0.5, 0.5]$
- Outlier-resistant
- Missing-data robust

### 3.4 Data Validation and Missing Values

#### Valid Observation Criteria

```python
valid = y_w.notna() & X_rank.notna().all(axis=1)
y_final = y_w[valid]
X_final = X_rank[valid]
```

**Listwise Deletion Rationale:**

- **IPCA Requirements:**
  - Needs complete instrument vector $z_{i,t}$
  - Factor extraction requires observed returns

- **Alternative: Imputation**
  - Not used in paper's main specification
  - Could introduce bias in characteristic signals

#### Panel Balance Considerations

**Unbalanced Panel Properties:**

```python
N = y_final.index.get_level_values('id').nunique()  # Cross-section
T = y_final.index.get_level_values('eom').nunique()  # Time-series
print(f"N={N}, T={T}, Obs={len(y_final)}")
```

**Econometric Implications:**

**Survivorship Bias:**

- Long-lived firms may differ systematically
- Mitigated by: Including all CRSP/Compustat firms

**Sample Selection:**

- Firms enter/exit based on data availability
- Not random: IPO, delisting, merger

**Time-Varying Cross-Section:**

- $N_t$ varies over time
- Handled naturally by cross-sectional operations

## 4. Instrument Construction Module

### 4.1 Characteristic Selection

#### The 36 Characteristics

**Categories in KPS (2019):**

- **Value (6):** beme, e2p, c, d2a, a2me, noa
- **Profitability (8):** roa, roe, rna, prof, pm, pcm, ato, cto
- **Investment (4):** investment, noa, dpi2a, ol
- **Momentum (4):** mom_12_2, mom_12_7, mom_2_1, mom_36_13
- **Issuance (2):** s2p, fc2y
- **Trading Frictions (5):** beta, lturnover, spread, suv, idio_vol
- **Other (7):** q, size, free_cf, oa, rel_high, sga2s, lev

#### Implementation Mapping

```python
characteristic_mapping = {
    'beta': 'beta_60m',           # 60-month rolling market beta
    'a2me': 'at_me',              # Assets-to-market equity
    'log_at': 'log_assets',       # Log total assets
    'ato': 'at_turnover',         # Asset turnover
    'beme': 'be_me',              # Book-to-market equity
    'c': 'cash_at',               # Cash-to-assets
    # ... (see code for complete mapping)
}
```

### 4.2 Mean-Deviation Decomposition

#### Theoretical Foundation

**Time-Series Decomposition:**

For each characteristic $c_{i,t}$:

$$c_{i,t} = \bar{c}_i + (c_{i,t} - \bar{c}_i) = \text{mean}_i + \text{dev}_{i,t}$$

Where:
* Firm-specific average (persistent component):
  <div>$$\bar{c}_i = \frac{1}{T_i} \sum_t c_{i,t}$$</div>
* Time-varying deviation (transitory component):
  <div>$$\text{dev}_{i,t} = c_{i,t} - \bar{c}_i$$</div>

#### Economic Interpretation

**Mean Component** ($\bar{c}_i$):

- Persistent firm characteristics
- Captures long-run attributes
- Examples:
  - Size: Large firms stay large
  - Book-to-market: Value firms stay value

**Deviation Component:**

- Transitory shocks
- Time-varying within-firm variation
- Examples:
  - Momentum: Recent price changes
  - Profitability changes

#### Implementation Details

**In-Sample Specification:**

```python
# Compute FULL-SAMPLE mean for each firm
means = X_final.groupby(level='id').transform('mean')

# Compute deviations
devs = X_final - means

# Rename for clarity
means.columns = ['mean_' + c for c in means.columns]
devs.columns = ['dev_' + c for c in devs.columns]
```

**Statistical Properties:**

1. **Orthogonality:** By construction, for each firm $i$:
   <div>$$E[\text{dev}_{i,t}] = 0$$</div>

2. **Variance Decomposition:**
   <div>$$\text{Var}(c_{i,t}) = \text{Var}(\bar{c}_i) + \text{Var}(\text{dev}_{i,t}) + 2 \cdot \text{Cov}(\bar{c}_i, \text{dev}_{i,t})$$</div>
   
   where:
   <div>$$\text{Cov}(\bar{c}_i, \text{dev}_{i,t}) = 0$$</div>

3. **Information Content:**
   - Mean: Cross-sectional variation (firm fixed effects)
   - Deviation: Time-series variation (within-firm dynamics)

#### Out-of-Sample Modification

**Expanding Window Means:**
```python
# EXPANDING mean (changes each period)
char_mean = (
    df_work.groupby('permno')[f'{char}_ranked']
    .expanding().mean()
    .reset_index(0, drop=True)
)
```

**Critical Difference:**

* **In-Sample:** (Uses ALL periods including future)
  <div>$$\bar{c}_i = \frac{1}{T} \sum_{s=1}^{T} c_{i,s}$$</div>

* **Out-of-Sample:** (Uses ONLY past periods)
  <div>$$\bar{c}_{i,t} = \frac{1}{t} \sum_{s=1}^{t} c_{i,s}$$</div>


**Why Different?**

- **Look-Ahead Bias Prevention:**
  - In-sample: Future not used for prediction
  - Out-of-sample: Must use only available information

- **Real-Time Implementability:**
  - Expanding mean can be computed in real-time
  - Full-sample mean requires future data

### 4.3 Instrument Vector Construction

#### Complete Instrument Set

For $L$ characteristics, construct $2L+1$ instruments:

```python
const = pd.Series(1.0, index=X_final.index, name='const')
X_instr_unres = pd.concat([const, means, devs], axis=1)
X_instr_res = pd.concat([means, devs], axis=1)
```

**Structure:**

$$z_{i,t} = \begin{bmatrix} 1 \\ \bar{c}_1^i \\ \vdots \\ \bar{c}_L^i \\ (c_1 - \bar{c}_1)_{i,t} \\ \vdots \\ (c_L - \bar{c}_L)_{i,t} \end{bmatrix} = \begin{bmatrix} 1 \\ \text{mean}_1 \\ \vdots \\ \text{mean}_L \\ \text{dev}_1 \\ \vdots \\ \text{dev}_L \end{bmatrix}$$

**Dimensions:**

- Unrestricted model: $2L+1 = 73$ instruments (36 chars × 2 + 1)
- Restricted model: $2L = 72$ instruments (no constant)

#### Why This Particular Structure?

**Comparison with Alternatives:**

**Alternative 1: Raw Characteristics Only**

$$z_{i,t} = [1, c_{1,i,t}, c_{2,i,t}, \ldots, c_{L,i,t}]'$$

Problems:

- Misses persistent vs. transitory distinction
- Less flexible factor loading dynamics

**Alternative 2: Characteristics + Lags**

$$z_{i,t} = [1, c_{1,i,t}, c_{2,i,t}, \ldots, c_{1,i,t-1}, c_{2,i,t-1}, \ldots]'$$

Problems:

- Data loss from lagging
- Collinearity between contemporaneous and lagged values

**KPS Choice: Mean + Deviation**

$$z_{i,t} = [1, \bar{c}_1^i, \ldots, \bar{c}_L^i, (c_1 - \bar{c}_1)_{i,t}, \ldots, (c_L - \bar{c}_L)_{i,t}]'$$

Advantages:

- ✓ Captures both persistent and transitory components
- ✓ No data loss (mean computable from first observation)
- ✓ Economic interpretation clear
- ✓ Orthogonal decomposition (dev has zero mean)

### 4.4 Secondary Rank Transformation

#### The Double Ranking Procedure

After mean-deviation decomposition, rank AGAIN:

```python
# Rank means cross-sectionally
means_ranked = means.groupby(level='eom').rank(pct=True) - 0.5

# Rank deviations cross-sectionally  
devs_ranked = devs.groupby(level='eom').rank(pct=True) - 0.5
```

#### Why Rank Twice?

**First Ranking** (on raw $c_{i,t}$):

- Makes characteristics comparable
- Removes units and scale
- Outlier robustness

**Mean-Deviation Decomposition:**

- Separates persistent from transitory
- Now have different variables (means vs. deviations)

**Second Ranking** (on means and devs):

- Makes means and deviations comparable
- Puts both on $[-0.5, 0.5]$ scale
- Equal weighting in estimation

#### Statistical Properties of Double-Ranked Instruments

**Distribution:** $z_{i,t} \sim \text{Uniform}(-0.5, 0.5)^{2L+1}$

**Advantages:**

1. **Numerical Stability:**
   - All instruments same scale
   - Condition number of $Z'Z$ improved
   - No dominant variables

2. **Interpretation:**
   - $\Gamma$ coefficients directly comparable
   - Effect sizes comparable across characteristics

3. **Regularization:**
   - Implicit ridge regression effect
   - Prevents overweighting any single characteristic

## 5. IPCA Estimation Algorithm

### 5.1 The Estimation Problem

#### Objective Function

**Panel Data Setup:**

$$r_{i,t+1} = \alpha_i + (\Gamma' z_{i,t})' f_{t+1} + \varepsilon_{i,t+1}$$

**Stacked Panel Form:**

$$r = Z\Gamma F' + \varepsilon$$

Where:

- $r$: $(NT \times 1)$ vector of all returns
- $Z$: $(NT \times L)$ matrix of instruments (varies by $i,t$)
- $\Gamma$: $(L \times K)$ characteristics-to-loadings map
- $F$: $(T \times K)$ matrix of factor realizations
- $\varepsilon$: $(NT \times 1)$ idiosyncratic errors

#### Non-Convex Optimization

Minimize sum of squared errors:

$$\min_{\Gamma, F} \text{SSE} = \|r - Z\Gamma F'\|^2_F = \sum_{i,t} \left(r_{i,t+1} - z'_{i,t} \Gamma f_{t+1}\right)^2$$

**Identification Constraints:**

- Factors orthonormal: $F'F/T = I_K$
- Factors ordered by variance explained
- Scaling: $\Gamma'\Gamma$ diagonal

### 5.2 Alternating Least Squares (ALS) Algorithm

#### Core Iteration Procedure

**Algorithm Structure:**

```
Initialize: F^(0) randomly or via PCA

For iter = 1, 2, ..., max_iter:
    Step 1: Update Γ given F
        Solve: min_Γ ||r - ZΓF'||^2
    
    Step 2: Update F given Γ
        Solve: min_F ||r - ZΓF'||^2
        Subject to: F'F = T·I_K
    
    Step 3: Check convergence
        If ||Γ^(iter) - Γ^(iter-1)||_F < tol:
            STOP
```

#### Step 1: Update $\Gamma$ (Characteristics Loadings)

Conditional on $F$, problem is linear in $\Gamma$:

$$\min_\Gamma \|r - Z\Gamma F'\|^2$$

**Vectorization Trick:**

Using $\text{vec}(\cdot)$ operator and Kronecker product:

$$\text{vec}(r) = \text{vec}(Z\Gamma F') = (F \otimes Z)\text{vec}(\Gamma)$$

Where $\otimes$ is the Kronecker product.

**Normal Equations:**

$$(F \otimes Z)'(F \otimes Z)\text{vec}(\Gamma) = (F \otimes Z)'\text{vec}(r)$$

Using property: $(F \otimes Z)'(F \otimes Z) = (F'F) \otimes (Z'Z)$

$$[(F'F) \otimes (Z'Z)]\text{vec}(\Gamma) = \text{vec}(Z'rF)$$

$$\text{vec}(\Gamma) = [(F'F) \otimes (Z'Z)]^{-1}\text{vec}(Z'rF)$$

**Computational Simplification:**

If $F'F \approx T \cdot I_K$ (nearly orthonormal factors):

$$\text{vec}(\Gamma) \approx [T \cdot I_K \otimes (Z'Z)]^{-1}\text{vec}(Z'rF) = [T \cdot (Z'Z)]^{-1}\text{vec}(Z'rF)$$

$$\Gamma = (Z'Z)^{-1}Z'rF/T$$

**Implementation:**

```python
# Panel setup
Z_mat = np.array(X_instruments)  # NT × L
r_vec = np.array(y_returns)      # NT × 1  
F_mat = factors                   # T × K

# Efficient computation (avoids Kronecker product)
Gamma_new = np.linalg.solve(
    Z_mat.T @ Z_mat,                                  # L × L
    Z_mat.T @ r_vec.reshape(-1, 1) @ F_mat.T        # L × K
)
```

#### Step 2: Update $F$ (Factors)

Conditional on $\Gamma$, problem is linear in $F$:

$$\min_F \|r - Z\Gamma F'\|^2 \quad \text{s.t.} \quad F'F = T \cdot I_K$$

**Define Predicted Loadings:**

$$\beta_{i,t} = \Gamma' z_{i,t}$$

$$B = Z\Gamma \quad \text{(NT × K matrix of predicted loadings)}$$

**Rewrite objective:**

$$\min_F \|r - BF'\|^2$$

**Without Constraint:** OLS gives:

$$F_{\text{ols}} = r'B(B'B)^{-1}$$

**With Orthonormality Constraint:**

This is a Procrustes problem. Solution via SVD:

1. Compute: $M = B'r$ (K × T matrix)
2. SVD: $M = U\Sigma V'$
3. Solution: $F = VU'$ (T × K)

**Why This Works:**

The Procrustes problem seeks:

$$\max_{F: F'F = T \cdot I} \text{trace}(F'B'r) = \text{trace}(B'rF)$$

SVD of $B'r = U\Sigma V'$ gives: $F = VU'$ maximizes $\text{trace}(B'rF)$

**Implementation:**

```python
# Compute predicted loadings
B = Z_mat @ Gamma  # NT × K

# Compute crossproduct
M = B.T @ r_vec  # K × T (approximately)

# SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)
F_new = Vt.T @ U.T  # T × K

# Ensure orthonormality
F_new = F_new * np.sqrt(T)  # Scale to F'F = T·I
```

#### Convergence and Stability

**Convergence Criterion:**

```python
delta = np.linalg.norm(Gamma_new - Gamma_old, 'fro')
if delta < tolerance:
    break
```

**Convergence Rate:**

- Typically converges in 10-50 iterations
- Linear convergence rate
- Not guaranteed globally (non-convex problem)

**Initialization Sensitivity:**

**Good Initialization: PCA**

```python
# Start with standard PCA factors
U, s, Vt = np.linalg.svd(returns_matrix, full_matrices=False)
F_init = U[:, :K] * s[:K]
```

**Random Initialization:**

```python
F_init = np.random.randn(T, K)
# Usually works but slower convergence
```

### 5.3 Restricted vs. Unrestricted Models

#### Model Comparison

**Restricted Model** ($\Gamma_\alpha = 0$):

$$r_{i,t+1} = z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

**Estimation:**

```python
ipca_res = InstrumentedPCA(n_factors=K, intercept=False)
ipca_res.fit(X=X_instr_res, y=y_final)
```

Parameters: $K \times (2L)$ where $L=36$ → $K \times 72$

**Unrestricted Model** ($\Gamma_\alpha \neq 0$):

$$r_{i,t+1} = \alpha_i + z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

Equivalent form with intercept instrument:

$$r_{i,t+1} = [1, z'_{i,t}] \begin{bmatrix} \Gamma_\alpha \\ \Gamma \end{bmatrix} f_{t+1} + \varepsilon_{i,t+1}$$

**Estimation:**

```python
ipca_unres = InstrumentedPCA(n_factors=K, intercept=True)
ipca_unres.fit(X=X_instr_unres, y=y_final)
```

Parameters: $K \times (2L+1)$ where $L=36$ → $K \times 73$

#### Interpretation of $\Gamma_\alpha$

**Abnormal Return Component:**

In the unrestricted model:

$$\beta_{i,t} = [\Gamma_\alpha, \Gamma_1, \ldots, \Gamma_L]' [1, z_{1,i,t}, \ldots, z_{L,i,t}]' = \Gamma_\alpha + \sum_l \Gamma_l z_{l,i,t}$$

So $\Gamma_\alpha$ is the constant term in the loading:

- Average exposure across all firms
- "Market loading" or baseline factor exposure

#### Economic Hypotheses

**$H_0: \Gamma_\alpha = 0$ (Restricted)**

- Interpretation: No systematic pricing errors
- All expected returns explained by characteristics × factors
- Characteristics fully span the SDF

**$H_1: \Gamma_\alpha \neq 0$ (Unrestricted)**

- Interpretation: Systematic unexplained returns remain
- Missing risk factors or mispricing
- Characteristics don't fully span the SDF

**Statistical Test:**

The Wald test tests: $H_0: \Gamma_\alpha = 0_K$ (K-dimensional zero vector)

### 5.4 Handling Missing Data in Panel

#### The Unbalanced Panel Challenge

Reality:

- Firm 1: Present months 1-600
- Firm 2: Present months 50-400
- Firm 3: Present months 200-600
- ...

Standard PCA Assumption: Complete matrix

IPCA Advantage: Missing data handling built-in

#### ALS Implementation for Missing Data

**Modified Algorithm:**

```python
def ipca_with_missing(Z, r, K, max_iter=100):
    """
    Z: Instruments (may have missing rows)
    r: Returns (may have missing values)
    K: Number of factors
    """
    # Mask of observed data
    obs_mask = ~np.isnan(r)
    
    # Initialize
    F = initialize_factors(r[obs_mask], K)
    
    for iteration in range(max_iter):
        # Update Γ using ONLY observed data
        Gamma_new = update_gamma_missing(Z, r, F, obs_mask)
        
        # Update F using ONLY observed data
        F_new = update_factors_missing(Z, r, Gamma_new, obs_mask)
        
        # Check convergence
        if converged(Gamma_new, Gamma):
            break
            
        Gamma = Gamma_new
        F = F_new
    
    return Gamma, F
```

**Key Modifications:**

**$\Gamma$ Update with Missing Data:**

```python
def update_gamma_missing(Z, r, F, mask):
    # For each instrument l and factor k:
    for l in range(L):
        for k in range(K):
            # Use only (i,t) where r_{i,t+1} is observed
            obs_idx = np.where(mask)[0]
            
            X_lk = Z[obs_idx, l] * F[time_idx[obs_idx], k]
            y_lk = r[obs_idx]
            
            # Solve normal equations
            Gamma[l, k] = (X_lk.T @ y_lk) / (X_lk.T @ X_lk)
    return Gamma
```

**$F$ Update with Missing Data:**

```python
def update_factors_missing(Z, r, Gamma, mask):
    # For each time period t:
    for t in range(T):
        # Use only stocks with observed returns at t
        obs_firms_t = np.where(mask[:, t])[0]
        
        if len(obs_firms_t) == 0:
            F[t, :] = 0  # No data this period
            continue
        
        # Predicted loadings for observed stocks
        B_t = Z[obs_firms_t, :] @ Gamma
        r_t = r[obs_firms_t, t]
        
        # OLS (unconstrained)
        F[t, :] = np.linalg.lstsq(B_t, r_t, rcond=None)[0]
    
    return F
```

#### Statistical Properties with Missing Data

**Consistency:**

- ALS with missing data: Still consistent as $N, T \to \infty$
- Requires: Missing at random (MAR) assumption
- Missingness pattern: Firms enter/exit, not selective on unobservables

**Efficiency Loss:**

- Fewer observations → larger standard errors
- Particularly affects: Factor estimates in periods with few stocks

**Practical Considerations:**

```python
# Minimum observations per date
valid_dates = returns_panel.notna().sum(axis=1) >= min_firms
returns_panel = returns_panel.loc[valid_dates]

# Minimum observations per firm  
valid_firms = returns_panel.notna().sum(axis=0) >= min_periods
returns_panel = returns_panel.loc[:, valid_firms]
```

## 6. Model Evaluation Module

### 6.1 $R^2$ Statistics

#### Panel $R^2$ (Total $R^2$)

**Definition:**

Fraction of total return variation explained:

$$R^2_{\text{panel}} = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{i,t} (r_{i,t+1} - \hat{r}_{i,t+1})^2}{\sum_{i,t} r_{i,t+1}^2}$$

Where:

$$\hat{r}_{i,t+1} = z'_{i,t} \hat{\Gamma} \hat{f}_{t+1} \quad \text{(fitted values)}$$

**Implementation:**

```python
def calculate_panel_r2(ipca_model, X_instruments, y_returns):
    # Predict returns
    y_pred = ipca_model.predict(X=X_instruments)
    
    # Align indices (handle missing data)
    y_true = y_returns.loc[X_instruments.index]
    
    # Calculate R²
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    ss_res = np.sum((y_true[mask] - y_pred[mask])**2)
    ss_tot = np.sum(y_true[mask]**2)
    
    return 1 - ss_res / ss_tot
```

**Interpretation:**

- High $R^2$: Factors explain large fraction of returns
- Comparison: Market model vs. Fama-French vs. IPCA

**Why Low $R^2$?**

1. Individual stock returns are noisy: $r_{i,t} = E[r_{i,t}] + \varepsilon_{i,t}$ where $E[r_{i,t}] \approx 5\%$, $\sigma(\varepsilon_{i,t}) \approx 20\%$

2. Idiosyncratic risk dominates:
   - Firm-specific news
   - Measurement error
   - Microstructure noise

3. Monthly frequency:
   - Lower frequency → lower $R^2$
   - Compare: Daily $R^2$ often higher

#### Predictive $R^2$ (Mean Return Prediction)

**Definition:**

How well do factors predict average returns:

$$R^2_{\text{pred}} = 1 - \frac{\sum_i (\bar{r}_i - \hat{\mu}_i)^2}{\sum_i \bar{r}_i^2}$$

Where:

* Average return for stock $i$:
  <div>$$\bar{r}_i = \frac{1}{T_i} \sum_t r_{i,t+1}$$</div>
* Predicted mean return:
  <div>$$\hat{\mu}_i = \frac{1}{T_i} \sum_t z_{i,t}^\top \hat{\Gamma} \hat{\lambda}$$</div>
- And risk premium: $\hat{\lambda} = \frac{1}{T} \sum_t \hat{f}_t$ (time-series average of factors)

**Implementation:**

```python
def calculate_predictive_r2(ipca_model, X_instruments, y_returns):
    # Extract factor premia (time-series means)
    factors = ipca_model.Factors  # T × K
    lambda_hat = factors.mean(axis=0)  # K × 1
    
    # Predicted mean returns
    Gamma = ipca_model.Gamma  # L × K
    Z = X_instruments.values  # NT × L
    
    # Expected return for each observation
    expected_returns = (Z @ Gamma) @ lambda_hat
    
    # Group by firm and compute means
    df = pd.DataFrame({
        'permno': y_returns.index.get_level_values('id'),
        'r_true': y_returns.values,
        'r_pred': expected_returns
    })
    
    actual_means = df.groupby('permno')['r_true'].mean()
    predicted_means = df.groupby('permno')['r_pred'].mean()
    
    # R²
    ss_res = np.sum((actual_means - predicted_means)**2)
    ss_tot = np.sum(actual_means**2)
    
    return 1 - ss_res / ss_tot
```

**Interpretation:**

- High $R^2_{\text{pred}}$: Characteristics predict risk premia well
- Economic Meaning: Cross-sectional variation in expected returns well-explained

**Why Much Higher Than Panel $R^2$?**

- **Panel $R^2$:** Predicts $r_{i,t+1}$ (individual returns) — High noise from idiosyncratic shocks
- **Pred $R^2$:** Predicts $E[r_i]$ (expected returns) — Idiosyncratic noise averages out

### 6.2 Managed Portfolio $R^2$

#### Construction of Managed Portfolios

**Definition:**

Managed portfolio returns:

$$x_t = \sum_i w_{i,t-1} r_{i,t}$$

Where weights are characteristic-based:

$$w_{i,t-1} = \frac{z_{i,t-1}}{\sum_j z_{j,t-1}}$$

Dimension: If $z_{i,t}$ is $L \times 1$, then $x_t$ is $L \times 1$ vector

**Economic Interpretation:**

Each component of $x_t$ is a:

- Characteristics-sorted portfolio
- Weight = characteristic ranking
- Example: $x_t[\text{size}]$ = Size-weighted portfolio

**Implementation:**

```python
def construct_managed_portfolios(y_returns, X_instruments):
    """
    Construct x_t = Z_t' r_t for each date
    """
    managed_portfolios = []
    
    for date in y_returns.index.get_level_values('eom').unique():
        # Get cross-section at date t
        mask = y_returns.index.get_level_values('eom') == date
        
        r_t = y_returns[mask].values  # N_t × 1
        Z_t = X_instruments[mask].values  # N_t × L
        
        # Managed portfolio: x_t = Z_t' r_t
        x_t = Z_t.T @ r_t  # L × 1
        
        managed_portfolios.append(x_t)
    
    # Stack into T × L matrix
    X_panel = np.vstack(managed_portfolios)
    return X_panel
```

#### Managed Portfolio $R^2$

Panel $R^2$ for Managed Portfolios:

$$R^2_{x_t} = 1 - \frac{\sum_t \|x_t - \hat{x}_t\|^2}{\sum_t \|x_t\|^2}$$

Where predicted managed portfolio:

$$\hat{x}_t = \Gamma f_t f'_t \Gamma'$$

**Why Care About $x_t$ $R^2$?**

1. **Noise Reduction:**
   - Individual returns: High idiosyncratic noise
   - Managed portfolios: Diversified, less noise

2. **Power Against Alternatives:**
   - Tests characteristic-based strategies
   - Captures systematic mispricing patterns

3. **Practical Trading:**
   - Implementable strategies
   - Transaction cost considerations

### 6.3 Number of Factors Selection

#### Conceptual Tradeoff

**Bias-Variance Tradeoff:**

$$\text{MSE}(K) = \text{Bias}^2(K) + \text{Variance}(K)$$

- Too few factors ($K$ small): Underfit, miss important factors
- Too many factors ($K$ large): Overfit, capture noise

#### Information Criteria

**Bayesian Information Criterion (BIC):**

$$\text{BIC}(K) = -2 \cdot \log(\mathcal{L}) + (\text{num parameters}) \cdot \log(\text{num observations})$$

$$= T \cdot \log(\text{SSE}/T) + K(L + T) \cdot \log(NT)$$

**Akaike Information Criterion (AIC):**

$$\text{AIC}(K) = -2 \cdot \log(\mathcal{L}) + 2 \cdot (\text{num parameters})$$

$$= T \cdot \log(\text{SSE}/T) + 2 \cdot K(L + T)$$

**Implementation:**

```python
def compute_ic(sse, K, L, N, T):
    NT = N * T
    n_params = K * (L + T)  # Gamma + Factors
    
    bic = NT * np.log(sse / NT) + n_params * np.log(NT)
    aic = NT * np.log(sse / NT) + 2 * n_params
    
    return aic, bic
```

#### Cross-Validation Approach

**Time-Series Cross-Validation:**

```
Training: [====Estimation====][==Test==]
          1963-1980           1981-1985

          [====Estimation====][==Test==]
          1963-1985           1986-1990
          
          ...
```

**Procedure:**

1. Divide sample into training/validation windows
2. For each $K$:
   - Estimate on training window
   - Predict on validation window
   - Compute out-of-sample $R^2$
3. Select $K$ with highest avg OOS $R^2$

**Implementation:**

```python
def cross_validate_factors(y, X, K_range, n_splits=5):
    """
    Time-series cross-validation for K selection
    """
    dates = y.index.get_level_values('eom').unique().sort_values()
    T = len(dates)
    window_size = T // (n_splits + 1)
    
    cv_scores = {K: [] for K in K_range}
    
    for split in range(n_splits):
        # Define windows
        train_end = (split + 1) * window_size
        test_end = train_end + window_size
        
        train_dates = dates[:train_end]
        test_dates = dates[train_end:test_end]
        
        # Split data
        y_train = y[y.index.get_level_values('eom').isin(train_dates)]
        X_train = X[X.index.get_level_values('eom').isin(train_dates)]
        
        y_test = y[y.index.get_level_values('eom').isin(test_dates)]
        X_test = X[X.index.get_level_values('eom').isin(test_dates)]
        
        # Evaluate each K
        for K in K_range:
            model = InstrumentedPCA(n_factors=K)
            model.fit(X=X_train, y=y_train)
            
            # Out-of-sample R²
            r2_oos = model.score(X=X_test, y=y_test)
            cv_scores[K].append(r2_oos)
    
    # Average scores
    avg_scores = {K: np.mean(scores) for K, scores in cv_scores.items()}
    optimal_K = max(avg_scores, key=avg_scores.get)
    
    return optimal_K, avg_scores
```

## 7. Statistical Testing Module

### 7.1 The Wald Test for Pricing Errors

#### Null Hypothesis

**Restricted Model** ($H_0$):

$$r_{i,t+1} = z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

No intercept ⟹ All expected returns explained by factors

**Unrestricted Model** ($H_1$):

$$r_{i,t+1} = \alpha_i + z'_{i,t} \Gamma f_{t+1} + \varepsilon_{i,t+1}$$

Asset-specific intercepts ⟹ Unexplained expected returns

#### Test Statistic

**Wald Statistic:**

Under unrestricted model, intercept vector is:

$$\alpha = [\alpha_1, \alpha_2, \ldots, \alpha_N]'$$

But we don't estimate $\alpha_i$ directly. Instead:

$$\alpha_i \approx \Gamma'_\alpha \cdot 1 \quad \text{(constant loading on factor)}$$

So the Wald test targets:

$$H_0: \Gamma_\alpha = 0_K \quad \text{(K-dimensional zero vector)}$$

**Test Statistic:**

$$W_\alpha = \hat{\Gamma}'_\alpha \hat{\Gamma}_\alpha = \|\hat{\Gamma}_\alpha\|^2 \quad \text{(squared Euclidean norm)}$$

Under $H_0$:

Asymptotically: $W_\alpha \xrightarrow{d} \chi^2_K$ (Chi-squared with K degrees of freedom)

#### Why Not Use Asymptotic Distribution?

**Problems with Standard Asymptotics:**

1. **Panel Structure:**
   - Cross-sectional dependence (factors affect all stocks)
   - Time-series dependence (autocorrelation)
   - Not IID sample

2. **Finite Sample Bias:**
   - $N=10,000, T=600$, but effective sample size smaller
   - Estimation error in $\hat{\Gamma}, \hat{F}$ affects distribution

3. **Non-Standard Null:**
   - Testing boundary of parameter space
   - Standard errors difficult to compute

**Solution: Wild Bootstrap**

### 7.2 Wild Residual Bootstrap

#### Bootstrap Methodology

**Purpose:** Construct empirical distribution of $W_\alpha$ under $H_0$

**Algorithm:**

```
Input: Unrestricted model fit (Γ̂, F̂)
Output: Bootstrap p-value

1. Fit unrestricted model on original data:
   r_{i,t+1} = α̂_i + ẑ'_{i,t} Γ̂ f̂_{t+1} + ε̂_{i,t+1}

2. Compute residuals:
   ε̂_{i,t+1} = r_{i,t+1} - r̂_{i,t+1}

3. Compute observed test statistic:
   W_α^obs = ||Γ̂_α||^2

4. For b = 1, ..., B bootstrap draws:
   a. Draw random signs: η^(b)_{i,t} ~ Uniform({-1, +1})
   b. Generate bootstrap returns:
      r^(b)_{i,t+1} = r̂_{i,t+1} + η^(b)_{i,t} · ε̂_{i,t+1}
   c. Re-estimate unrestricted model on r^(b):
      (Γ̂^(b), F̂^(b)) = IPCA(r^(b), Z)
   d. Compute bootstrap statistic:
      W_α^(b) = ||Γ̂^(b)_α||^2

5. Compute p-value:
   p = (1 + #{W_α^(b) >= W_α^obs}) / (B + 1)
```

#### Why "Wild" Bootstrap?

**Comparison with Standard Bootstrap:**

**Standard (Pairs) Bootstrap:**

- Resample $(i,t)$ pairs with replacement
- Problem: Destroys panel structure

**Block Bootstrap:**

- Resample blocks of consecutive observations
- Problem: Loses cross-sectional dependence

**Wild Bootstrap:**

- Keep all $(i,t)$, randomize only residual signs
- Advantages:
  - ✓ Preserves panel structure
  - ✓ Maintains cross-sectional dependence
  - ✓ Handles heteroskedasticity
  - ✓ Computationally efficient

#### Implementation Details

```python
def compute_Walpha_pval(ipca_unres, X_instr_unres, y_final, 
                        B=1000, seed=12345):
    """
    Wild residual bootstrap for W_α p-value
    """
    rng = np.random.default_rng(seed)
    
    # Step 1: Observed test statistic
    Gamma_alpha = ipca_unres.Gamma[:, -1]  # Last column = intercept
    Walpha_obs = float(Gamma_alpha.T @ Gamma_alpha)
    
    # Step 2: Fitted values and residuals
    y_pred = ipca_unres.predict(X_instr_unres)
    residuals = y_final - y_pred
    
    # Step 3: Bootstrap loop
    boot_stats = []
    
    for b in range(B):
        # Random signs (Rademacher distribution)
        signs = rng.choice([-1, 1], size=len(residuals))
        
        # Bootstrap sample
        y_boot = y_pred + residuals * signs
        
        # Re-estimate model
        boot_model = InstrumentedPCA(
            n_factors=ipca_unres.n_factors,
            intercept=True
        )
        boot_model.fit(X=X_instr_unres, y=y_boot)
        
        # Bootstrap statistic
        Gamma_alpha_b = boot_model.Gamma[:, -1]
        Walpha_b = float(Gamma_alpha_b.T @ Gamma_alpha_b)
        boot_stats.append(Walpha_b)
    
    # Step 4: P-value with finite-sample correction
    boot_stats = np.array(boot_stats)
    count_ge = np.sum(boot_stats >= Walpha_obs)
    pval = (count_ge + 1) / (B + 1)
    
    return Walpha_obs, pval
```

### 7.3 Testing Individual Characteristics

#### Motivation

**Question:** Which characteristics matter for each factor?

**Null Hypothesis:**

For characteristic $l$ and factor $k$:

$$H_0: \Gamma_{l,k} = 0$$

Characteristic $l$ doesn't predict loading on factor $k$

#### $W_\beta$ Test Statistic

**For Single Characteristic** $l$:

$$W_{\beta,l} = \sum_k (\hat{\Gamma}_{l,k})^2 \quad \text{(sum across all K factors)}$$

Tests: Does characteristic $l$ affect ANY factor?

**For Single Factor** $k$ and Characteristic $l$:

$$W_{\beta,l,k} = (\hat{\Gamma}_{l,k})^2$$

Tests: Does characteristic $l$ affect factor $k$ specifically?

#### Bootstrap Implementation

**Algorithm:**

```
1. Fit RESTRICTED model (no intercept):
   r_{i,t+1} = z'_{i,t} Γ̂ f̂_{t+1} + ε̂_{i,t+1}

2. Compute residuals and W_{β,l}^obs

3. For b = 1, ..., B:
   a. Generate bootstrap sample (wild residuals)
   b. Re-estimate model
   c. Compute W_{β,l}^(b)

4. P-value: p_l = (1 + #{W_{β,l}^(b) >= W_{β,l}^obs}) / (B+1)
```

**Implementation:**

```python
def test_characteristic_significance(ipca_model, char_index, 
                                     X_instr, y, ndraws=1000):
    """
    Bootstrap test for individual characteristic
    """
    # Observed statistic
    Gamma = ipca_model.Gamma  # L × K
    W_obs = np.sum(Gamma[char_index, :]**2)  # Sum over factors
    
    # Residuals
    y_pred = ipca_model.predict(X_instr)
    residuals = y - y_pred
    
    # Bootstrap
    boot_stats = []
    rng = np.random.default_rng(12345)
    
    for b in range(ndraws):
        signs = rng.choice([-1, 1], size=len(residuals))
        y_boot = y_pred + residuals * signs
        
        boot_model = InstrumentedPCA(
            n_factors=ipca_model.n_factors,
            intercept=False
        )
        boot_model.fit(X=X_instr, y=y_boot)
        
        W_boot = np.sum(boot_model.Gamma[char_index, :]**2)
        boot_stats.append(W_boot)
    
    # P-value
    pval = (np.sum(boot_stats >= W_obs) + 1) / (ndraws + 1)
    
    return W_obs, pval
```

#### Multiple Testing Correction

**Problem:** Testing $L=36$ characteristics

- Type I error inflation
- Some "significant" results by chance

**Bonferroni Correction:**

Reject $H_0$ if $p_l < \alpha/L$

For $\alpha=0.05, L=36$: Adjusted threshold: $0.05/36 = 0.0014$

**False Discovery Rate (FDR):**

Benjamini-Hochberg procedure:

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(L)}$
2. Find largest $k$ such that: $p_{(k)} \leq (k/L) \cdot \alpha$
3. Reject $H_0$ for characteristics $1, \ldots, k$

**Implementation:**

```python
from statsmodels.stats.multitest import multipletests

def fdr_correction(pvalues, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction
    """
    reject, pvals_corrected, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method='fdr_bh'
    )
    return reject, pvals_corrected
```

### 7.4 Standard Error Estimation

#### Sources of Uncertainty

**Estimation Error in:**

- $\hat{\Gamma}$: Characteristic loadings
- $\hat{F}$: Factor realizations
- Interaction effects

**Panel Data Complications:**

- Cross-sectional correlation (factors)
- Time-series correlation (autocorrelation)
- Heteroskedasticity

#### Bootstrap Standard Errors

**Advantage:** Handles all dependencies naturally

**Procedure:**

For each parameter $\theta$ (e.g., $\Gamma_{l,k}$):

1. Compute point estimate $\hat{\theta}$ from original data
2. For $b=1,\ldots,B$:
   - Generate bootstrap sample
   - Compute $\hat{\theta}^{(b)}$
3. Standard error: $\text{SE}(\hat{\theta}) = \text{std}(\hat{\theta}^{(1)}, \ldots, \hat{\theta}^{(B)})$
4. Confidence interval (percentile method): $\text{CI}_\alpha = [q_{\alpha/2}, q_{1-\alpha/2}]$ where $q_p$ = p-th quantile of $\hat{\theta}^{(b)}$

**Implementation:**

```python
def bootstrap_standard_errors(X_instr, y, n_factors, ndraws=1000):
    """
    Bootstrap standard errors for Gamma
    """
    # Original estimate
    model = InstrumentedPCA(n_factors=n_factors, intercept=False)
    model.fit(X=X_instr, y=y)
    Gamma_hat = model.Gamma
    
    # Residuals
    y_pred = model.predict(X_instr)
    residuals = y - y_pred
    
    # Bootstrap estimates
    Gamma_boot = np.zeros((ndraws, *Gamma_hat.shape))
    rng = np.random.default_rng(12345)
    
    for b in range(ndraws):
        signs = rng.choice([-1, 1], size=len(residuals))
        y_boot = y_pred + residuals * signs
        
        boot_model = InstrumentedPCA(n_factors=n_factors, intercept=False)
        boot_model.fit(X=X_instr, y=y_boot)
        Gamma_boot[b] = boot_model.Gamma
    
    # Standard errors (element-wise)
    SE_Gamma = Gamma_boot.std(axis=0)
    
    # 95% Confidence intervals
    CI_lower = np.percentile(Gamma_boot, 2.5, axis=0)
    CI_upper = np.percentile(Gamma_boot, 97.5, axis=0)
    
    return SE_Gamma, CI_lower, CI_upper
```

#### Fama-MacBeth Standard Errors (Alternative)

**Two-Pass Procedure:**

**Pass 1:** Time-series regressions for loadings

For each stock $i$:

$$r_{i,t} = \alpha_i + \sum_k \beta_{i,k} f_{k,t} + \varepsilon_{i,t}$$

Estimate: $\hat{\beta}_{i,k}$

**Pass 2:** Cross-sectional regression

For each date $t$:

$$r_{i,t} = \gamma_{0,t} + \sum_k \gamma_{k,t} \hat{\beta}_{i,k} + \eta_{i,t}$$

Estimate: $\hat{\gamma}_{k,t}$

**Standard Errors:**

$$\text{SE}(\bar{\gamma}_k) = \sqrt{\frac{\text{Var}(\hat{\gamma}_{k,1}, \ldots, \hat{\gamma}_{k,T})}{T}}$$

With Newey-West correction for autocorrelation

**Not Used in IPCA:**

- Time-invariant loadings assumption (doesn't apply)
- IPCA has time-varying $\beta_{i,t} = \Gamma' z_{i,t}$

## 8. Out-of-Sample Prediction Module

### 8.1 Expanding Window Methodology

#### Real-Time Implementation

**Goal:** Mimic real-world portfolio construction

- Use only information available at time $t$
- No look-ahead bias
- Realistic transaction costs considerations

**Expanding Window Setup:**

```
Estimation Window:    [1963-07, t]
Prediction:           t+1
Next Window:          [1963-07, t+1]
...
```

**Implementation:**

```python
def expanding_window_predictions(df, char_cols, K, 
                                 initial_window=120):
    """
    Expanding window out-of-sample predictions
    
    Parameters:
    - initial_window: Minimum months for first estimation (e.g., 120)
    - K: Number of factors
    """
    dates = df['date'].unique()
    predictions = []
    
    for t_idx in range(initial_window, len(dates)):
        # Training period: From start to t-1
        train_dates = dates[:t_idx]
        test_date = dates[t_idx]
        
        # Training data
        df_train = df[df['date'].isin(train_dates)]
        
        # Build instruments (out-of-sample: expanding mean)
        df_train_instr, instr_cols = build_instruments_oosample(
            df_train, char_cols
        )
        
        # Prepare test data
        df_test = df[df['date'] == test_date]
        
        # CRITICAL: Update test instruments using training history
        df_test_instr = update_test_instruments(
            df_test, df_train, char_cols
        )
        
        # Estimate IPCA on training data
        X_train, y_train = build_panel_matrices(
            df_train_instr, instr_cols
        )
        
        model = InstrumentedPCA(n_factors=K, intercept=False)
        model.fit(X=X_train, y=y_train)
        
        # Predict on test data
        X_test = df_test_instr[instr_cols]
        y_pred = model.predict(X=X_test)
        
        # Store predictions
        predictions.append({
            'date': test_date,
            'predictions': y_pred,
            'actuals': df_test['ret_lead'].values
        })
    
    return predictions
```

#### Critical: Instrument Construction for OOS

**Training Instruments (Period $t$):**

For stock $i$ at time $t$, using data from $\tau=1$ to $t$:

1. Rank characteristic:
   <div>$$c^{\text{rank}}_{i,t}$$</div>

2. Expanding mean:
   <div>$$\bar{c}_{i,t} = \frac{1}{t} \sum_{\tau=1}^{t} c^{\text{rank}}_{i,\tau}$$</div>

3. Deviation:
   <div>$$\text{dev}_{i,t} = c^{\text{rank}}_{i,t} - \bar{c}_{i,t}$$</div>

4. Rank both mean and deviation cross-sectionally

**Test Instruments (Period $t+1$):**

1. Rank characteristic:
   <div>$$c^{\text{rank}}_{i,t+1}$$</div>

2. Update mean:
   <div>$$\bar{c}_{i,t+1} = \frac{t}{t+1}\bar{c}_{i,t} + \frac{1}{t+1}c^{\text{rank}}_{i,t+1}$$</div>

3. Deviation:
   <div>$$\text{dev}_{i,t+1} = c^{\text{rank}}_{i,t+1} - \bar{c}_{i,t+1}$$</div>

4. Rank both cross-sectionally at $t+1$

**Implementation:**

```python
def update_test_instruments(df_test, df_train, char_cols):
    """
    Construct test instruments using only training information
    """
    df_test = df_test.copy()
    
    for char in char_cols:
        # Step 1: Rank characteristic at test date
        df_test[f'{char}_ranked'] = (
            df_test[char].rank(pct=True) - 0.5
        )
        
        # Step 2: Expanding mean from training data
        train_history = df_train.groupby('permno')[f'{char}_ranked'].mean()
        
        # Map to test stocks (handle new stocks)
        df_test[f'{char}_mean'] = df_test['permno'].map(train_history).fillna(0)
        
        # Step 3: Deviation
        df_test[f'{char}_dev'] = (
            df_test[f'{char}_ranked'] - df_test[f'{char}_mean']
        )
        
        # Step 4: Rank mean and deviation
        df_test[f'{char}_mean_rank'] = (
            df_test[f'{char}_mean'].rank(pct=True) - 0.5
        )
        df_test[f'{char}_dev_rank'] = (
            df_test[f'{char}_dev'].rank(pct=True) - 0.5
        )
    
    return df_test
```

### 8.2 Out-of-Sample $R^2$

#### Definition

**OOS $R^2$ for Period $T+1$ to $T+H$:**

$$R^2_{\text{OOS}} = 1 - \frac{\sum_{t=T+1}^{T+H} \sum_i (r_{i,t} - \hat{r}_{i,t|t-1})^2}{\sum_{t=T+1}^{T+H} \sum_i r_{i,t}^2}$$

Where:

- $\hat{r}_{i,t|t-1}$: Prediction made at $t-1$ for return at $t$
- Uses model estimated on data up to $t-1$

**Interpretation:**

- Positive: Model beats naive forecast ($\hat{r}=0$)
- Negative: Model worse than naive (overfitting)
- Comparison: IPCA vs. benchmarks

**Implementation:**

```python
def calculate_oos_r2(predictions_list):
    """
    Compute out-of-sample R² from list of predictions
    """
    all_actuals = []
    all_predictions = []
    
    for pred_dict in predictions_list:
        all_actuals.extend(pred_dict['actuals'])
        all_predictions.extend(pred_dict['predictions'])
    
    actuals = np.array(all_actuals)
    preds = np.array(all_predictions)
    
    # Handle missing values
    mask = np.isfinite(actuals) & np.isfinite(preds)
    actuals = actuals[mask]
    preds = preds[mask]
    
    # R²
    ss_res = np.sum((actuals - preds)**2)
    ss_tot = np.sum(actuals**2)
    
    r2_oos = 1 - ss_res / ss_tot
    
    return r2_oos
```

### 8.3 Benchmark Comparisons

#### Models for Comparison

**1. Historical Mean:**

$$\hat{r}_{i,t|t-1} = \frac{1}{t-1} \sum_{\tau=1}^{t-1} r_{i,\tau}$$

Simplest forecast, often hard to beat!

**2. Fama-French 3-Factor:**

$$r_{i,t} = \alpha_i + \beta_{i,\text{MKT}} \text{MKT}_t + \beta_{i,\text{SMB}} \text{SMB}_t + \beta_{i,\text{HML}} \text{HML}_t + \varepsilon_{i,t}$$

Loadings estimated on training window

**3. OLS on Characteristics:**

$$r_{i,t+1} = \alpha + \sum_l \gamma_l c_{l,i,t} + \varepsilon_{i,t+1}$$

Pooled panel regression

**4. Elastic Net:**

$$\min_\gamma \|r - X\gamma\|^2 + \lambda_1 |\gamma|_1 + \lambda_2 |\gamma|^2$$

Regularized regression with cross-validated $\lambda$

**5. Principal Component Regression (PCR):**

- Extract $K$ PCs from characteristics matrix
- Regress returns on PCs

### 8.4 Campbell-Thompson Bounds

#### Motivation

**Problem:** OOS $R^2$ can be negative

- Doesn't mean model is "bad"
- Just worse than naive forecast ($\hat{r}=0$)
- Small negative $R^2$ from poor market timing

#### Bounding Procedure

**Campbell-Thompson (2008) Restrictions:**

1. **Sign Constraint:** Predicted and actual returns have same sign

$$\text{If } \text{sign}(\hat{r}_{i,t}) \neq \text{sign}(\bar{r}_i), \text{ set } \hat{r}_{i,t} = 0$$

2. **Magnitude Constraint:** Predictions bounded by historical volatility

$$\hat{r}_{i,t} \in [\hat{\mu}_i - 3\hat{\sigma}_i, \hat{\mu}_i + 3\hat{\sigma}_i]$$

**Implementation:**

```python
def apply_ct_bounds(predictions, historical_stats):
    """
    Apply Campbell-Thompson bounds to predictions
    
    historical_stats: Dict with keys 'mean' and 'std' for each stock
    """
    bounded_preds = predictions.copy()
    
    for i, pred in enumerate(predictions):
        stock_id = get_stock_id(i)
        
        mu = historical_stats[stock_id]['mean']
        sigma = historical_stats[stock_id]['std']
        
        # Sign constraint
        if pred * mu < 0:  # Opposite signs
            pred = 0
        
        # Magnitude constraint
        lower = mu - 3 * sigma
        upper = mu + 3 * sigma
        pred = np.clip(pred, lower, upper)
        
        bounded_preds[i] = pred
    
    return bounded_preds
```

## 9. Advanced Topics

### 9.1 Large vs. Small Stock Decomposition

#### Motivation

**Empirical Observation:** Many anomalies stronger in small stocks

- Liquidity differences
- Information frictions
- Institutional trading constraints

**Question:** Do IPCA factors differ across size groups?

#### Sample Splits

**NYSE Size Breakpoint:**

```python
def split_by_size(df):
    """
    Split sample into large (above NYSE median) and small
    """
    # NYSE breakpoint (median market cap of NYSE stocks)
    nyse_stocks = df[df['exchange'] == 'NYSE']
    nyse_breakpoint = nyse_stocks.groupby('date')['market_cap'].median()
    
    # Classify all stocks
    df['size_group'] = 'Small'
    
    for date, breakpoint in nyse_breakpoint.items():
        mask = (df['date'] == date) & (df['market_cap'] > breakpoint)
        df.loc[mask, 'size_group'] = 'Large'
    
    return df
```

#### Cross-Sample Evaluation

**Estimate in One, Evaluate in Other:**

- Model: Estimated on Large, Evaluated on Small
- Model: Estimated on Small, Evaluated on Large

**Implementation:**

```python
# Estimate on large stocks
model_large = InstrumentedPCA(n_factors=4)
model_large.fit(X_large, y_large)

# Evaluate on small stocks
r2_large_to_small = model_large.score(X_small, y_small)

# Vice versa
model_small = InstrumentedPCA(n_factors=4)
model_small.fit(X_small, y_small)
r2_small_to_large = model_small.score(X_large, y_large)
```

### 9.2 Comparison with Observable Factor Models

#### Fama-French 5-Factor Model

**Specification:**

$$r_{i,t} = \alpha_i + \beta_{\text{MKT}} \text{MKT}_t + \beta_{\text{SMB}} \text{SMB}_t + \beta_{\text{HML}} \text{HML}_t + \beta_{\text{RMW}} \text{RMW}_t + \beta_{\text{CMA}} \text{CMA}_t + \varepsilon_{i,t}$$

**Factors:**

- MKT: Market excess return
- SMB: Small minus big
- HML: High minus low book-to-market
- RMW: Robust minus weak profitability
- CMA: Conservative minus aggressive investment

#### Panel Regression with Instruments

**Model C in Table II:**

$$r_{i,t+1} = \sum_k \beta_{k,t} G_{k,t} + \varepsilon_{i,t+1}$$

Where:

$$\beta_{k,t} = z'_{i,t} \Gamma_k \quad \text{(instrumented loadings)}$$

$$G_{k,t} = \text{observable factor (e.g., FF5 factors)}$$

**Estimation:**

Two-stage approach:

1. **Stage 1:** For each factor $k$:

$$r_{i,t+1} = \sum_l [z_{l,i,t} \times G_{k,t}] \delta_{l,k} + \text{residual}$$

Estimate $\delta_{l,k}$ for all $L$ instruments and $K$ factors

2. **Stage 2:** Construct:

$$\hat{\beta}_{k,i,t} = \sum_l z_{l,i,t} \hat{\delta}_{l,k}$$

**Implementation:**

```python
def instrumented_observable_factors(y, X_instruments, factors):
    """
    Estimate observable factor model with instrumented loadings
    
    factors: DataFrame with columns ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    """
    K = factors.shape[1]
    L = X_instruments.shape[1]
    
    # Construct interaction terms: Z_{i,t} ⊗ G_{k,t}
    interactions = []
    for k in range(K):
        factor_k = factors.iloc[:, k]
        
        # Broadcast factor to panel
        factor_panel = y.index.get_level_values('eom').map(
            dict(zip(factors.index, factor_k))
        )
        
        # Multiply each instrument by factor
        for l in range(L):
            interactions.append(
                X_instruments.iloc[:, l] * factor_panel
            )
    
    # Stack into design matrix
    X_design = pd.DataFrame(interactions).T
    
    # OLS
    delta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]
    
    # Reshape to Γ matrix (L × K)
    Gamma = delta_hat.reshape(L, K)
    
    return Gamma
```

### 9.3 SDF Representation and Pricing

#### Stochastic Discount Factor

**Asset Pricing Fundamental Equation:**

$$E_t[m_{t+1} r_{i,t+1}] = 1$$

Where $m_{t+1}$ is the SDF (pricing kernel)

**Factor Model SDF:**

$$m_{t+1} = 1 - \lambda' (f_{t+1} - E[f])$$

Where $\lambda$ is the vector of risk premia

#### IPCA SDF

From the model:

$$r_{i,t+1} = \alpha_i + \beta'_{i,t} f_{t+1} + \varepsilon_{i,t+1}$$

Taking expectations:

$$E[r_i] = \alpha_i + \beta'_{i,t} E[f]$$

No-arbitrage ($\alpha_i=0$) implies:

$$E[r_i] = \beta'_{i,t} \lambda$$

Where $\lambda = E[f]$ is the vector of factor risk premia

#### Estimating $\lambda$

**Two Approaches:**

**1. Time-Series Average:**

```python
lambda_hat = factors.mean(axis=0)  # Mean of f_t over time
```

**2. Cross-Sectional Regression:**

Fama-MacBeth:

For each date $t$:

$$r_{i,t} = \gamma_0 + \beta'_{i,t-1} \gamma + \eta_{i,t}$$

Time-series average: $\hat{\lambda} = \text{mean}(\hat{\gamma}_t \text{ over } t)$

**Implementation:**

```python
def estimate_risk_premia(returns, loadings, factors):
    """
    Estimate factor risk premia via Fama-MacBeth
    """
    dates = returns.index.get_level_values('eom').unique()
    gamma_t = []
    
    for date in dates:
        # Cross-section at date t
        r_t = returns[returns.index.get_level_values('eom') == date]
        beta_t = loadings[loadings.index.get_level_values('eom') == date]
        
        # Cross-sectional regression
        gamma = np.linalg.lstsq(beta_t, r_t, rcond=None)[0]
        gamma_t.append(gamma)
    
    # Time-series average
    lambda_hat = np.mean(gamma_t, axis=0)
    
    # Fama-MacBeth standard errors
    se_lambda = np.std(gamma_t, axis=0) / np.sqrt(len(gamma_t))
    
    return lambda_hat, se_lambda
```

#### Sharpe Ratio of SDF

**Maximum Sharpe Ratio:**

The tangency portfolio has Sharpe ratio:

$$\text{SR}_{\max} = \sqrt{\lambda' \Sigma_f^{-1} \lambda}$$

Where $\Sigma_f = \text{Cov}(f)$ is the factor covariance matrix

**For IPCA:**

```python
def compute_sdf_sharpe(factors):
    """
    Compute maximum Sharpe ratio from IPCA factors
    """
    # Factor risk premia
    lambda_hat = factors.mean(axis=0)
    
    # Factor covariance
    Sigma_f = factors.cov()
    
    # Check invertibility
    try:
        Sigma_f_inv = np.linalg.inv(Sigma_f)
    except np.linalg.LinAlgError:
        return np.nan
    
    # Sharpe ratio (annualized)
    sr_monthly = np.sqrt(lambda_hat.T @ Sigma_f_inv @ lambda_hat)
    sr_annual = sr_monthly * np.sqrt(12)
    
    return sr_annual
```

### 9.4 Relation to Kelly-Jiang (2014) RP-PCA

#### Reduced-Rank Regression

**Kelly-Jiang Framework:**

$$r_{t+1} = \alpha + Bz_t + \varepsilon_{t+1}$$

Where:

- $r_{t+1}$: $N \times 1$ returns
- $z_t$: $L \times 1$ predictors (characteristics)
- $B$: $N \times L$ coefficient matrix

**Rank Constraint:**

$$\text{rank}(B) = K < \min(N, L)$$

Implies: $B = \Lambda \Gamma'$ where $\Lambda$ is $N \times K$, $\Gamma$ is $L \times K$

#### Differences from IPCA

**RP-PCA:**

- Time-invariant loadings ($\Lambda$)
- Predictors $z_t$ enter linearly
- Focus: Return predictability

**IPCA:**

- Time-varying loadings ($\beta_{i,t} = \Gamma' z_{i,t}$)
- Characteristics instrument loadings
- Focus: Factor structure

**Relationship:**

RP-PCA can be seen as restricted IPCA:

$$\text{RP-PCA: } r_{i,t+1} = \sum_k \lambda_{i,k} (\Gamma'_k z_t) + \varepsilon_{i,t+1}$$

$$\text{IPCA: } r_{i,t+1} = (\Gamma' z_{i,t})' f_{t+1} + \varepsilon_{i,t+1}$$

### 9.5 Implementation in Production Systems

#### Computational Considerations

**Bottlenecks:**

- Memory: $NT \times L$ instrument matrix (can be 10GB+)
- ALS Iterations: Each iteration expensive
- Bootstrap: $B=1000$ draws × full estimation

**Optimization Strategies:**

**1. Sparse Matrices:**

```python
from scipy.sparse import csr_matrix

# If many zeros in instruments (unlikely here)
X_sparse = csr_matrix(X_instruments)
```

**2. Batching for Bootstrap:**

```python
def parallel_bootstrap(y, X, B, n_jobs=-1):
    """
    Parallel bootstrap using joblib
    """
    from joblib import Parallel, delayed
    
    def single_bootstrap(seed):
        rng = np.random.default_rng(seed)
        # ... bootstrap iteration
        return W_alpha
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_bootstrap)(seed) 
        for seed in range(B)
    )
    
    return results
```

**3. Warm Starts:**

```python
# Use previous estimate as initialization
F_init = model_previous.Factors
model_current = InstrumentedPCA(n_factors=K)
model_current.fit(X, y, F_init=F_init)
```

#### Real-Time Forecasting

**Monthly Rebalancing:**

```python
class IPCAForecaster:
    def __init__(self, char_cols, K=4, lookback=120):
        self.char_cols = char_cols
        self.K = K
        self.lookback = lookback  # Minimum training months
        self.model = None
        self.history = []
    
    def update(self, new_data):
        """
        Update model with new month of data
        """
        self.history.append(new_data)
        
        # Ensure sufficient history
        if len(self.history) < self.lookback:
            return None
        
        # Combine training data
        df_train = pd.concat(self.history)
        
        # Build instruments
        X_train, y_train = self.prepare_data(df_train)
        
        # Re-estimate model
        self.model = InstrumentedPCA(n_factors=self.K)
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, current_characteristics):
        """
        Generate forecasts for next month
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert characteristics to instruments
        X_current = self.build_instruments(current_characteristics)
        
        # Predict
        predictions = self.model.predict(X_current)
        
        return predictions
```

#### Transaction Costs

**Incorporating Costs:**

```python
def net_return_after_costs(predicted_returns, current_weights, 
                           target_weights, cost_per_trade=0.001):
    """
    Adjust predictions for transaction costs
    """
    # Turnover
    turnover = np.abs(target_weights - current_weights).sum()
    
    # Cost
    total_cost = turnover * cost_per_trade
    
    # Expected return net of costs
    gross_return = predicted_returns @ target_weights
    net_return = gross_return - total_cost
    
    return net_return
```

**Optimal Rebalancing:**

```python
def optimal_weights_with_costs(predicted_returns, current_weights,
                               Sigma, cost, risk_aversion):
    """
    Mean-variance optimization with transaction costs
    """
    from scipy.optimize import minimize
    
    def objective(w):
        # Mean-variance utility
        mu = predicted_returns @ w
        sigma2 = w.T @ Sigma @ w
        utility = mu - (risk_aversion/2) * sigma2
        
        # Transaction cost penalty
        turnover = np.abs(w - current_weights).sum()
        utility -= cost * turnover
        
        return -utility  # Minimize negative utility
    
    # Constraints: weights sum to 1, long-only
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    bounds = [(0, 1) for _ in range(len(predicted_returns))]
    
    result = minimize(
        objective,
        x0=current_weights,
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x
```

## Summary and Conclusion

### Key Takeaways

**IPCA Innovation:**

- Unifies factor models and characteristic-based return prediction
- Time-varying loadings as functions of characteristics
- Dimension reduction through instrumentation

**Statistical Framework:**

- Non-convex optimization via ALS
- Wild bootstrap for inference
- Robust to missing data and unbalanced panels

**Implementation Details:**

- Double rank transformation critical
- Mean-deviation decomposition captures dynamics
- Expanding window for out-of-sample validity

**Practical Deployment:**

- Real-time implementable
- Handles transaction costs
- Scalable with modern computing

### Extensions and Future Research

**Deep Learning IPCA:**

- Neural networks for $\beta_{i,t} = \text{NN}(z_{i,t})$
- Non-linear characteristic effects

**Regime-Switching:**

- Time-varying $\Gamma$: $\Gamma_t = f(\text{macro variables})$
- Crisis vs. normal periods

**High-Frequency:**

- Intraday factors
- Microstructure considerations

**Alternative Assets:**

- Corporate bonds
- Options
- Cryptocurrencies

**Interpretable AI:**

- Attention mechanisms for characteristic importance
- Explainable factor structure

## References

### Primary Paper

Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501-524.

### Methodological

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221.
- Lettau, M., & Pelger, M. (2020). Factors that fit the time series and cross-section of stock returns. *The Review of Financial Studies*, 33(5), 2274-2325.

### Empirical Asset Pricing

- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *The Review of Financial Studies*, 29(1), 5-68.

### Statistical Methods

- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *The Review of Economics and Statistics*, 90(3), 414-427.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

---

This guide provides a complete statistical and computational understanding of IPCA with proper mathematical notation throughout. For specific implementation questions or theoretical extensions, refer to the original paper and code repository.
