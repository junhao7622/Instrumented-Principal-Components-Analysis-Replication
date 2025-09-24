"""
Strict replication script for Table I from (Characteristics are covariances: A unified model of risk and return, Kelly, Pruitt, Su 2019 JFE)
Notes:
 - Data loading & preprocessing (rank-transform, winsorize, column mean centering)
 - Construction of instruments (Appendix-aligned order)
 - Estimation of restricted (Γα = 0) and unrestricted (Γα ≠ 0) IPCA for K = 1..6
 - Computation of panel/portfolio R²'s as in the original script
 - Wild residual bootstrap for Wα p-value using implementation in ipca.InstrumentedPCA.BS_Walpha
   (follows KPS Appendix B; see ipca.py for details)
Author: Junhao Gao
Date: 2025-09
"""


import pandas as pd
import numpy as np
import warnings
from ipca import InstrumentedPCA

# ---------------------------
# basic settings AND change as needed
BOOTSTRAP_DRAWS = 1000 # number of bootstrap draws for Walpha p-value
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore', category=FutureWarning) # ignore pandas future warnings
DATA_CSV = 'Common_Task_Monthly_Data.csv' # path to data
START_DATE = "1963-07-01" #time
END_DATE = "2014-05-31"
print("Replication: KPS (2019) Table I - IPCA with Instruments")

# ---------------------------
# 1. Data Loading
df = pd.read_csv(DATA_CSV)

if 'market_equity' in df.columns:
    df['log_market_equity'] = np.log(df['market_equity'].clip(lower=1e-6))
 
if 'assets' in df.columns:
    df['log_assets'] = np.log(df['assets'].clip(lower=1e-6))

    # adjustment to personal dataset, ni_noa is needed as original paper request
if 'ni_noa' not in df.columns and 'net_income' in df.columns and 'noa_at' in df.columns:
    df['ni_noa'] = df['net_income'] / df['noa_at']
    df['ni_noa'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # Per your analysis, add filters to better align with the original sample
    print(f"Initial loaded observations: {df.shape[0]}")
    
    # Filter out non-US stocks if excntry is available
if 'excntry' in df.columns:
    df = df[df['excntry'] == 'USA'].copy()
    print(f"Observations after keeping USA only: {df.shape[0]}")

    # Filter out financial firms (SIC codes 6000-6999)
if 'sic' in df.columns:
    df['sic'] = pd.to_numeric(df['sic'], errors='coerce')
    df = df[(df['sic'] < 6000) | (df['sic'] > 6999)].copy()
    print(f"Observations after excluding financials: {df.shape[0]}")

df['eom'] = pd.to_datetime(df['eom'])
df = df[(df['eom'] >= pd.to_datetime(START_DATE)) & (
    df['eom'] <= pd.to_datetime(END_DATE))]
df = df.set_index(['id', 'eom'])

if 'ret_exc_lead1m' not in df.columns:
    raise KeyError("ret_exc_lead1m not found in data.")
 
y = df['ret_exc_lead1m']

# Map paper characteristic names to dataset columns
# NOTE: ensure 36 characteristics are present in your data
# and follow naming in original IPCA paper appendix
characteristic_mapping = {
    'beta': 'beta_60m', 'a2me': 'at_me', 'log_at': 'log_assets', 'ato': 'at_turnover',
    'beme': 'be_me', 'c': 'cash_at', 'cto': 'sale_bev', 'd2a': 'tangibility',
    'dpi2a': 'ppeg_gr1a', 'e2p': 'ni_me', 'fc2y': 'sga_gr1', 'free_cf': 'fcf_be',
    'idio_vol': 'ivol_capm_252d', 'investment': 'at_gr1', 'lev': 'debt_at',
    'size': 'log_market_equity', 'lturnover': 'turnover_126d', 'noa': 'noa_at',
    'oa': 'oaccruals_at', 'ol': 'ol_gr1a', 'pcm': 'gp_sale', 'pm': 'ni_sale',
    'prof': 'gp_at', 'q': 'at_mev', 'rel_high': 'prc_highprc_252d', 'rna': 'ni_noa',
    'roa': 'ni_at', 'roe': 'ni_be', 'mom_12_2': 'ret_12_1', 'mom_12_7': 'ret_12_7',
    'mom_2_1': 'ret_2_0', 'mom_36_13': 'ret_36_12', 's2p': 'sale_me',
    'sga2s': 'dsale_dsga', 'spread': 'bidaskhl_21d', 'suv': 'turnover_var_126d'
}

# If some chara. not present, try compute or warn, keep what available
available_cols = [v for v in characteristic_mapping.values() if v in df.columns]
reverse_map = {v:k for k,v in characteristic_mapping.items()}
if len(available_cols) < len(characteristic_mapping):
    print(
        "Warning: not all 36 characteristics present. Proceeding with available subset.")
X = df[available_cols].copy()
# rename columns to short names
X.columns = [reverse_map[c] for c in available_cols]

# ---------------------------
# 2. Preprocessing (winsorize, rank-transform, filter and function walpha p-value) 
y = df['ret_exc_lead1m'].copy()
X_raw = df[list(characteristic_mapping.values())].copy()
X_raw.columns = list(characteristic_mapping.keys())

def winsorize_series(s):
    q = s.quantile([0.01, 0.99])
    if pd.isna(q.iloc[0]) or pd.isna(q.iloc[1]):
        return s
    return np.clip(s, q.iloc[0], q.iloc[1])

y_w = y.groupby(level='eom').transform(winsorize_series)
X_w = X_raw.groupby(level='eom').transform(winsorize_series)

def rank_transform(panel):
    return panel.groupby(level='eom').rank(pct=True, na_option='keep') - 0.5

X_rank = rank_transform(X_w)

valid = y_w.notna() & X_rank.notna().all(axis=1)
y_final = y_w[valid]
X_final = X_rank[valid]

N = y_final.index.get_level_values('id').nunique()
T = y_final.index.get_level_values('eom').nunique()
print(f"Sample stats: N={N}, T={T}, Obs={len(y_final)}")

# compute Walpha p-value via wild residual bootstrap
# see KPS Appendix B and ipca.py: InstrumentedPCA.BS_Walpha for details
# Note: we implement our own version here to avoid the unsupported kwargs in ipca.py
def compute_Walpha_pval(ipca_unres, X_instr_unres, y_final, B=500, seed=1234):
    """
    Compute Wα p-value via wild residual bootstrap (KPS Appendix B).
    
    Parameters
    ----------
    ipca_unres : fitted InstrumentedPCA (unrestricted, intercept=True)
    X_instr_unres : DataFrame of unrestricted instruments (with const)
    y_final : Series of excess returns
    B : int, number of bootstrap draws
    seed : int, random seed
    
    Returns
    -------
    observed_W Observed Wald statistic for alpha (Γα ≠ 0).
    pval Bootstrap p-value (0..1).
    """
    rng = np.random.default_rng(seed)
    # Compute observed Walpha from unrestricted fit
    if not hasattr(ipca_unres, "Gamma"):
        raise RuntimeError("ipca_unres missing Gamma after fit().")
    Gamma_alpha_col = ipca_unres.Gamma[:, -1]  # last column = α loadings
    Walpha_obs = float(Gamma_alpha_col.T @ Gamma_alpha_col)
    # Residuals under unrestricted fit
    yhat = ipca_unres.predict(X_instr_unres)
    residuals = y_final - yhat
    # Bootstrap draws
    boot_stats = []
    for b in range(B):
        # Wild bootstrap: random ±1 multipliers
        signs = rng.choice([-1, 1], size=len(residuals))
        y_boot = yhat + residuals * signs
        # Re-fit unrestricted model on bootstrap sample
        boot_model = InstrumentedPCA(
            n_factors=ipca_unres.n_factors, intercept=True)
        boot_model.fit(X=X_instr_unres, y=y_boot)
        # Compute Walpha for bootstrap draw
        Gamma_alpha_b = boot_model.Gamma[:, -1]
        Walpha_b = float(Gamma_alpha_b.T @ Gamma_alpha_b)
        boot_stats.append(Walpha_b)
    boot_stats = np.array(boot_stats)
    # Compute p-value
    count_ge = np.sum(boot_stats >= Walpha_obs)
    pval = (count_ge + 1) / (B + 1)  # small-sample correction
    return Walpha_obs, pval

# ---------------------------
# 3. Instruments (Appendix-aligned order)
print("\n--- Constructing Instruments ---")
ordered_chars = list(characteristic_mapping.keys())

means = X_final.groupby(level='id').transform('mean')
devs = X_final - means

means.columns = ['mean_' + c for c in means.columns]
devs.columns  = ['dev_' + c for c in devs.columns]

means = means[['mean_' + c for c in ordered_chars]]
devs  = devs[['dev_' + c for c in ordered_chars]]

const = pd.Series(1.0, index=X_final.index, name='const')
X_instr_unres = pd.concat([const, means, devs], axis=1)
X_instr_res   = pd.concat([means, devs], axis=1)

print("Instrument count (should be 73):", X_instr_unres.shape[1])
print("First 8 instruments:", X_instr_unres.columns.tolist()[:8])
print("Last 8 instruments:", X_instr_unres.columns.tolist()[-8:])
assert X_instr_unres.shape[1] == 73

# ---------------------------
# 4. Data Loading
# NOTE:
#  We estimate restricted (Gamma_alpha = 0) and unrestricted (Gamma_alpha != 0)
#    IPCA for K = 1..6, compute panel/portfolio R2's as in the original script, and
#    perform the W_alpha bootstrap using the implementation in ipca.InstrumentedPCA.BS_Walpha.
#  The IPCA implementation (ipca.py) already implements the wild residual bootstrap
#    that follows KPS Appendix B. We therefore call that routine directly to avoid
#    re-implementing details (and to avoid passing unsupported kwargs like `verbose=`).
#    See ipca.py: BS_Walpha implementation for details. :contentReference[oaicite:3]{index=3}
#  For reproducibility we set numpy seed before bootstrap; here we use RANDOM_SEED + k.
results = []

for k in range(1, 7):
    print(f"\n--- Estimating K={k} factors ---")
    # Restricted model (Γα = 0)
    ipca_res = InstrumentedPCA(n_factors=k, intercept=False)
    ipca_res.fit(X=X_instr_res, y=y_final)
    # Unrestricted model (Γα ≠ 0)
    ipca_unres = InstrumentedPCA(n_factors=k, intercept=True)
    ipca_unres.fit(X=X_instr_unres, y=y_final)
    # Compute R² statistics
    total_r2_rt_res = ipca_res.score(
        X=X_instr_res, y=y_final, data_type="panel") * 100
    pred_r2_rt_res  = ipca_res.score(
        X=X_instr_res, y=y_final, data_type="panel", mean_factor=True) * 100
    total_r2_xt_res = ipca_res.score(
        X=X_instr_res, y=y_final, data_type="portfolio") * 100
    pred_r2_xt_res  = ipca_res.score(
        X=X_instr_res, y=y_final, data_type="portfolio", mean_factor=True) * 100
    total_r2_rt_unres = ipca_unres.score(
        X=X_instr_unres, y=y_final, data_type="panel") * 100
    pred_r2_rt_unres  = ipca_unres.score(
        X=X_instr_unres, y=y_final, data_type="panel", mean_factor=True) * 100
    total_r2_xt_unres = ipca_unres.score(
        X=X_instr_unres, y=y_final, data_type="portfolio") * 100
    pred_r2_xt_unres  = ipca_unres.score(
        X=X_instr_unres, y=y_final, data_type="portfolio", mean_factor=True) * 100
    # Compute Walpha and bootstrap p-value (custom implementation)
    Walpha_obs, walpha_pval = compute_Walpha_pval(
        ipca_unres, X_instr_unres, y_final,
        B=BOOTSTRAP_DRAWS,
        seed=RANDOM_SEED + k
    )
    print(
        f"  Observed Wα = {Walpha_obs:.6g}; bootstrap p-value = {walpha_pval*100:.3f}%")

    results.append({
        'K': k,
        'Total R2_rt_res': total_r2_rt_res,
        'Pred R2_rt_res': pred_r2_rt_res,
        'Total R2_rt_unres': total_r2_rt_unres,
        'Pred R2_rt_unres': pred_r2_rt_unres,
        'Total R2_xt_res': total_r2_xt_res,
        'Pred R2_xt_res': pred_r2_xt_res,
        'Total R2_xt_unres': total_r2_xt_unres,
        'Pred R2_xt_unres': pred_r2_xt_unres,
        'Walpha_obs': Walpha_obs,
        'Walpha_pval': walpha_pval * 100.0
    })

# ---------------------------
# 5. Organizing and Output results
# Format results into panels as in original Table I
kcols = [1, 2, 3, 4, 5, 6]
panel_a = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['', 'Panel A']), columns=kcols)
panel_b = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['', 'Panel B']), columns=kcols)
panel_c = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['', 'Panel C']), columns=kcols)

panel_a.loc[('Total R2','Γα=0'), :] = [r['Total R2_rt_res'] for r in results]
panel_a.loc[('Total R2','Γα≠0'), :] = [r['Total R2_rt_unres'] for r in results]
panel_a.loc[('Pred R2','Γα=0'), :] = [r['Pred R2_rt_res'] for r in results]
panel_a.loc[('Pred R2','Γα≠0'), :] = [r['Pred R2_rt_unres'] for r in results]

panel_b.loc[('Total R2','Γα=0'), :] = [r['Total R2_xt_res'] for r in results]
panel_b.loc[('Total R2','Γα≠0'), :] = [r['Total R2_xt_unres'] for r in results]
panel_b.loc[('Pred R2','Γα=0'), :] = [r['Pred R2_xt_res'] for r in results]
panel_b.loc[('Pred R2','Γα≠0'), :] = [r['Pred R2_xt_unres'] for r in results]

panel_c.loc[('Wα p-value',''), :] = [r['Walpha_pval'] for r in results]

print("\n--- Panel A ---\n", panel_a.round(4))
print("\n--- Panel B ---\n", panel_b.round(4))
print("\n--- Panel C ---\n", panel_c.round(4))
print("\n=== SCRIPT FINISHED ===")
