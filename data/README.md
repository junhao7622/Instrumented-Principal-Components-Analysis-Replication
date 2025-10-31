# Mock Data Dictionary (`characteristic_mapping`)

This document describes each variable in the mock dataset, grouped by category ***used in tabel I II***.  

For raw datasets, refer to following DROPBOX
Table1 2 data https://www.dropbox.com/scl/fi/2crtsguwyxrqmoh6ey4tp/Common_Task_Monthly_Data.csv?rlkey=nzry9obsn1g208mc8q5orxj3j&st=6lh38wod&dl=0
Table567 data https://www.dropbox.com/scl/fi/2crtsguwyxrqmoh6ey4tp/Common_Task_Monthly_Data.csv?rlkey=nzry9obsn1g208mc8q5orxj3j&st=web6job2&dl=0
FFdata ff5factors and Mom https://www.dropbox.com/scl/fi/iex6k5ef3mk8tq7v9rgz0/F-F_Momentum_Factor-dl.csv?rlkey=bh4fyunw6owfhok55cwl9c7ya&st=wlderwer&dl=0
https://www.dropbox.com/scl/fi/3xd6hxa0l07bvp3gxeehm/F-F_Research_Data_5_Factors_2x3.csv?rlkey=3o9dj1h82zl73fi7fxvpyuf11&st=7tdl5ozx&dl=0

---

## Identifiers & Core Variables
- **id**  
  Unique asset identifier (one per stock/firm).  

- **eom**  
  End-of-month date (time index).  

- **eom_ret**  
  Simple return for month *t*. **Note**: may contain **lookahead bias**, not recommended for empirical research.  

- **valid_ret**  
  Indicator for whether return is valid (0/1).  

- **excntry**  
  Country or market identifier (e.g., `USA`).  

- **me**  
  Market equity at month *t*.  

- **sic**  
  Standard Industrial Classification (industry code).  

- **size_grp**  
  Size group classification (Small, Medium, Large).  

- **prc**  
  Stock price.  

- **market_equity**  
  Market capitalization (may duplicate `me`).  

- **ret_exc_lead1m**  
  **Next month’s return (t+1)**. This is the return series you should use for tests, since it avoids lookahead bias.  

---

## Risk, Valuation & Beta
- **beta_60m**  
  CAPM beta estimated over the past 60 months.  

- **at_me**  
  Total assets-to-market equity ratio.  

- **assets**  
  Total assets.  

- **at_turnover**  
  Asset turnover ratio.  

- **be_me**  
  Book-to-market equity ratio.  

---

## Capital Structure, Cash Flow & Profitability
- **cash_at**  
  Cash holdings / total assets.  

- **sale_bev**  
  Sales / book equity value.  

- **tangibility**  
  Tangible assets ratio.  

- **ppeg_gr1a**  
  Growth in property, plant, and equipment.  

- **ni_me**  
  Net income / market equity.  

---

## Free Cash Flow, Volatility & Investment
- **sga_gr1**  
  Growth in SG&A expenses.  

- **dsale_dsga**  
  Change in sales relative to SG&A.  

- **fcf_be**  
  Free cash flow / book equity.  

- **ivol_capm_252d**  
  Idiosyncratic volatility from CAPM residuals (252-day window).  

- **at_gr1**  
  Total asset growth rate.  

- **debt_at**  
  Debt-to-assets ratio.  

---

## Size, Accruals & Operating Leverage
- **market_equity**  
  (Duplicate field) market capitalization.  

- **turnover_126d**  
  Stock turnover over 126 trading days.  

- **noa_at**  
  Net operating assets / total assets.  

- **oaccruals_at**  
  Operating accruals / total assets.  

- **ol_gr1a**  
  Growth in operating leverage.  

---

## Profitability, Valuation & Relative Price
- **gp_sale**  
  Gross profit / sales.  

- **ni_sale**  
  Net income / sales.  

- **gp_at**  
  Gross profit / total assets.  

- **at_mev**  
  Total assets / market equity value.  

- **prc_highprc_252d**  
  Price relative to past 252-day high.  

---

## Returns on Assets, Equity & Accruals
- **net_income**  
  Net income.  

- **noa_at**  
  (Duplicate field) Net operating assets / total assets.  

- **ni_at**  
  Net income / total assets (ROA).  

- **ni_be**  
  Net income / book equity (ROE).  

---

## Momentum & Reversal Factors
- **ret_12_1**  
  Past 12-month momentum, skipping the most recent month.  

- **ret_12_7**  
  Momentum from months t–12 to t–7.  

- **ret_2_0**  
  Short-term return (past 2 months).  

- **ret_36_12**  
  Long-term momentum (months t–36 to t–12).  

---

## Sales, Trading Frictions & Liquidity
- **sale_me**  
  Sales / market equity.  

- **dsale_dsga**  
  (Duplicate field) change in sales relative to SG&A.  

- **bidaskhl_21d**  
  Bid-ask spread high-low measure (21-day window).  

- **turnover_var_126d**  
  Variance of turnover over 126 trading days.  

---

# Notes
1. **id** is the unique firm identifier; each asset has multiple monthly observations.  
2. For return-based tests, use **`ret_exc_lead1m`** rather than `eom_ret` to avoid lookahead bias.  
3. The uploaded `mock_data.csv` files are **randomly generated mock data**, only for pipeline testing.  
   - Dates and distributions mimic real data structure.  
   - They **cannot reproduce actual Table 1 or Table 2 empirical results** (requires CRSP/Compustat).  
4. Intended use: validate data processing and pipeline functionality, not to draw empirical asset pricing conclusions.  
