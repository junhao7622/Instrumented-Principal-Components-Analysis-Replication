## Background
A central challenge in asset pricing is distinguishing between **risk-based explanations** (systematic factors) and **behavioral anomalies**. Traditional factor models such as CAPM, Fama-French 3-factor, and their extensions impose strong ex-ante assumptions on factor structures. However, the explosion of cross-sectional **characteristics-based factors** (so-called "factor zoo") has led to the need for a more **data-driven, statistically grounded** approach.

## Statistical tools
***1. Factor Model Layer***
- **Concept**: Classic linear factor models (returns = factors × loadings + residuals)  
- **IPCA Role**: Extends traditional factor models by estimating latent factors from high-dimensional characteristics and allowing factor loadings to vary with features.
  
***2. High-Dimensional Statistics Layer***
- **Concept**: Dimensionality reduction, multicollinearity, p ≫ n problems  
- **IPCA Role**: Uses PCA to extract low-dimensional orthogonal factors from high-dimensional characteristics, mitigating collinearity and overfitting.
  
***3. Instrumental Variable / Endogeneity Correction Layer***
- **Concept**: IV corrects for correlation between factors and residuals  
- **IPCA Role**: Applies instruments to adjust for potential endogeneity from characteristics, yielding more robust factor estimates.
  
***4. Prediction & Interpretation Layer***
- **Concept**: Trade-off between interpretability and predictive power  
- **IPCA Role**: Produces factors that explain returns structure while serving as inputs for high-dimensional predictive models, improving both interpretability and robustness.

## IPCA Framework
Kelly, Pruitt, and Su (2020) propose the **Instrumented Principal Component Analysis (IPCA)** as a method to estimate latent factors and their loadings directly from asset returns, using firm characteristics as instruments.

Formally, returns are modeled as:

$$
r_{i,t+1} = \alpha_i + \beta_{i,t}' f_{t+1} + \epsilon_{i,t+1},
$$

where factor loadings are parameterized as:

$$
\beta_{i,t} = \Gamma' z_{i,t},
$$

with $\( z_{i,t} \)$ representing firm-level characteristics and $\( \Gamma \)$ the mapping matrix between characteristics and factor loadings.


### Econometric Contribution
- **Instrumented loadings** mitigate overfitting and allow **out-of-sample prediction** of returns.  
- The framework naturally encompasses **characteristic-sorted portfolios** (like Fama-French factors) but nests them in a unified regression-based system.  
- Statistical tests such as the **Wα test** evaluate whether abnormal returns $(\alpha_i\)$ remain significant after accounting for the characteristic-instrumented factors.
