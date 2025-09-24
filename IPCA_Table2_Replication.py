"""
Replication script for Table II from (Characteristics are covariances: A unified model of risk and return, Kelly, Pruitt, Su 2019 JFE)
Includes:
 - Data loading & preprocessing 
    (rank-transform, winsorize option, column mean centering)
 - Panel B: Observable static betas on returns r_t and on managed portfolios x_t
 - Panel C: Observable factors with instruments 
    (one-step OLS implementation) on returns r_t and on managed portfolios x_t
 - Panel D: PCA on returns r_t and on managed portfolios x_t
 - Mask-aware R2 and predictive R2 consistent with paper definitions
 - ALS for PCA with missing values 
    NOTE: this is NOT the MATLAB ALS algorithm used in the paper result
 - Optional IPCA ALS routine (strict Kronecker normal-equation version)
Author: Junhao Gao
Date: 2025-09
"""

import os
import sys
import numpy as np
import pandas as pd
from numpy.linalg import lstsq, svd
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# basic settings AND change as needed
SEED = 12345
np.random.seed(SEED)

DATA_CSV = "Common_Task_Monthly_Data.csv"
FF5_CSV = "F-F_Research_Data_5_Factors_2x3.csv"
MOM_CSV = "F-F_Momentum_Factor dl.csv"

START_DATE = "1963-07-01"
END_DATE = "2014-05-31"
# optional winsorize of returns (paper uses rank-transform mainly)
USE_WINSORIZE = False
# None  'mean'  'median'  'zero' - None recommended for Panel D          
IMPUTATION_STRATEGY = None
# max ALS iterations    
ALS_MAX_ITER = 200
# ALS convergence tolerance               
ALS_TOL = 1e-10              
# use bootstrap for R2 SEs (paper uses bootstrap)
USE_BOOTSTRAP = True        
# set 1000 to match the paper; use smaller for quick tests
BOOTSTRAP_SAMPLES = 1000     
# optional variance normalization in centering (robustness)
NORMALIZE_VARIANCE = False      

# ---------------------------
# Helper functions for data processing, PCA, R2 calculations
def print_banner(msg):
    print("\n" + "="*80)
    print(msg)
    print("="*80 + "\n")

def load_data(csv_path=DATA_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found in working directory.")
    df = pd.read_csv(csv_path)
    # Expect columns including 'id', 'eom', 'ret_exc_lead1m', and characteristics
    # check csv columms first
    df['eom'] = pd.to_datetime(df['eom'])
    df = df[(df['eom'] >= START_date_or(START_DATE)) & (
        df['eom'] <= END_date_or(END_DATE))]
    return df

def START_date_or(s):
    # convert to Timestamp month-end
    return pd.to_datetime(s)

def END_date_or(s):
    return pd.to_datetime(s)

def winsorize_series(s, lower=0.01, upper=0.99):
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def rank_transform(panel_series):
    # panel_series: DataFrame indexed by (id,eom) or MultiIndex (id,eom)
    return panel_series.groupby(level='eom').rank(pct=True) - 0.5

def safe_mean(a, axis=None):
    return np.nanmean(a, axis=axis)

def calculate_r2(y_true, y_pred):
    """
    Calculate R^2 (coefficient of determination) between true and 
    predicted values.
    """
    y = np.asarray(y_true)
    yhat = np.asarray(y_pred)
    if y.shape != yhat.shape:
        raise ValueError("calculate_r2: shape mismatch")
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return np.nan
    ss_res = np.nansum((y[mask] - yhat[mask])**2)
    ss_tot = np.nansum(y[mask]**2)
    if ss_tot < 1e-12:
        return np.nan
    return (1.0 - ss_res / ss_tot) * 100.0

def bootstrap_r2(y_true, y_pred, n_samples=BOOTSTRAP_SAMPLES):
    """
    Bootstrap R2 calculation to estimate mean R2 and its standard error.
    Note: Have not been used in calculation but could be a back-up choice
    """
    if not USE_BOOTSTRAP:
        return calculate_r2(y_true, y_pred)
    arr_true = np.asarray(y_true).reshape(-1)
    arr_pred = np.asarray(y_pred).reshape(-1)
    mask = ~np.isnan(arr_true)
    arr_true_m = arr_true[mask]
    arr_pred_m = arr_pred[mask]
    if len(arr_true_m) == 0:
        return np.nan
    r2_vals = []
    n = len(arr_true_m)
    for _ in range(n_samples):
        idx = np.random.choice(n, n, replace=True)
        r2_vals.append(calculate_r2(arr_true_m[idx], arr_pred_m[idx]))
    return np.nanmean(r2_vals)

def pred_r2_using_lambda(actual_df, loadings, lambda_hat):
    """
    Predictive R2 percent, comparing mean actual returns
    to mean predicted returns.
    actual_df (T x N), loadings N x K, lambda_hat K
    """
    actual_means = actual_df.mean(axis=0).values
    L = np.asarray(loadings)
    lam = np.asarray(lambda_hat).reshape(-1)
    predicted_means = L.dot(lam)
    return calculate_r2(actual_means, predicted_means)

# ---------------------------
# ALS (alternating least square) for PCA with missing values
# NOTE: this is NOT the MATLAB IPCA algorithm in the paper
def pca_als(panel_centered, k, tol=ALS_TOL, max_iter=ALS_MAX_ITER, verbose=False):
    """
    panel_centered: DataFrame (T x N) already column-centered 
    (col means subtracted), may contain NaN.
    Returns: factors (DataFrame T x k), loadings 
    (DataFrame N x k), explained_var (k,), n_iter
    """
    panel_np = panel_centered.values.astype(float)
    mask = ~np.isnan(panel_np)
    T, N = panel_np.shape
    # initial imputation for SVD: fill missing with 0
    imputed = panel_np.copy()
    imputed[~mask] = 0.0
    try:
        U, s, Vt = svd(imputed, full_matrices=False)
    except Exception:
        # fallback random init
        U = np.random.randn(T, k)
        s = np.ones(k)
        Vt = np.random.randn(k, N)
    F = (U[:, :k] * s[:k])  # T x k
    L = Vt[:k, :].T         # N x k
    # ALS iterations--main part
    for it in range(max_iter):
        L_old = L.copy()
        # Update loadings L (N x k)
        #The code iterates through every single stock (for j in range(N)).
        #For each stock j, it uses the mask to find all the time periods 
            #(valid) where that stock has an actual, non-missing return.
        #It then runs a time-series regression: it regresses the stock's
            #observed returns (y) on the current estimate of the factors (X = F).
        #The resulting regression coefficients are the new, improved 
            #estimate for that stock's loadings, L[j, :].
        for j in range(N):
            valid = mask[:, j]
            if valid.sum() >= 1:
                X = F[valid, :]          
                y = panel_np[valid, j]  
                try:
                    L[j, :] = lstsq(X, y, rcond=None)[0]
                except Exception:
                    XTX = X.T @ X + 1e-8 * np.eye(X.shape[1])
                    L[j, :] = lstsq(XTX, X.T @ y, rcond=None)[0]
        # Update factors F (T x k)
        for i in range(T):
            valid = mask[i, :]
            if valid.sum() >= 1:
                X = L[valid, :]          
                y = panel_np[i, valid]
                try:
                    F[i, :] = lstsq(X, y, rcond=None)[0]
                except Exception:
                    XTX = X.T @ X + 1e-8 * np.eye(X.shape[1])
                    F[i, :] = lstsq(XTX, X.T @ y, rcond=None)[0]
        recon = F @ L.T
        num = np.linalg.norm((panel_np - recon)[mask])
        den = np.linalg.norm(panel_np[mask]) + 1e-12
        err = num / den
        if verbose and (it % 50 == 0 or it == 0):
            print(f" pca_als iter {it}: err={err:.3e}")
        if err < tol:
            break
        #Convergence Check
        if np.linalg.norm(L - L_old) / (np.linalg.norm(L_old) + 1e-12) < tol:
            break
    recon_final = F @ L.T
    explained_var = np.var(recon_final, axis=0)
    if explained_var.sum() > 0:
        explained_var = explained_var / explained_var.sum()
    else:
        explained_var = np.zeros(k)
    F_df = pd.DataFrame(F, index=panel_centered.index, columns=[
        f"F{i+1}" for i in range(k)])
    L_df = pd.DataFrame(L, index=panel_centered.columns, columns=[
        f"L{i+1}" for i in range(k)])
    return F_df, L_df, explained_var, it

# ---------------------------
# Optional: IPCA-style ALS (strict normal equations, slower)
def ipca_als(r_panel, Z_by_t, K, tol=1e-10, max_iter=500, ridge=1e-8, verbose=False):
    """
    r_panel: DataFrame T x N
    Z_by_t: dict of {date: Z_t DataFrame with index matching r_panel.columns}
    K: number of factors
    Returns Gamma (L x K), F (T x K DataFrame)
    NOTE: expensive but matches normal-equation approach in paper's appendix.
    """
    dates = list(r_panel.index)
    T = len(dates)
    N = r_panel.shape[1]
    L = next(iter(Z_by_t.values())).shape[1]
    # init F by PCA on mean-imputed r_panel
    r_np = r_panel.values.copy()
    mask_r = ~np.isnan(r_np)
    col_means = np.nanmean(r_np, axis=0)
    imp = r_np.copy()
    inds = np.where(~mask_r)
    if inds[0].size > 0:
        imp[inds] = np.take(col_means, inds[1])
    U, s, Vt = svd(imp, full_matrices=False)
    F = U[:, :K] * s[:K]
    Gamma = np.random.randn(L, K) * 1e-3
    # precompute Zt_list and rt_list
    Zt_list, rt_list = [], []
    for d in dates:
        Zt = Z_by_t[d].reindex(r_panel.columns).values
        rt = r_panel.loc[d].values
        Zt_list.append(Zt)
        rt_list.append(rt)
    for it in range(max_iter):
        Gamma_old = Gamma.copy()
        lhs = np.zeros((L*K, L*K))
        rhs = np.zeros(L*K)
        for t in range(T):
            ft = F[t, :]
            Zt = Zt_list[t]
            rt = rt_list[t]
            valid = (~np.isnan(Zt).any(axis=1)) & (~np.isnan(rt))
            if valid.sum() == 0:
                continue
            Ztv = Zt[valid, :]
            rtv = rt[valid]
            ZtZ = Ztv.T @ Ztv
            Ztr = Ztv.T @ rtv
            lhs += np.kron(np.outer(ft, ft), ZtZ)
            rhs += np.kron(ft, Ztr)
        lhs += ridge * np.eye(L*K)
        try:
            vecG = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            vecG = lstsq(lhs, rhs, rcond=None)[0]
        Gamma = vecG.reshape(L, K, order='F')
        # update F per t
        for t in range(T):
            Zt = Zt_list[t]
            rt = rt_list[t]
            valid = (~np.isnan(Zt).any(axis=1)) & (~np.isnan(rt))
            if valid.sum() == 0:
                F[t, :] = 0.0
                continue
            Ztv = Zt[valid, :]
            rtv = rt[valid]
            A = Gamma.T @ (Ztv.T @ Ztv) @ Gamma + ridge * np.eye(K)
            b = Gamma.T @ (Ztv.T @ rtv)
            try:
                F[t, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                F[t, :] = lstsq(A, b, rcond=None)[0]
        rel_change = np.linalg.norm(Gamma - Gamma_old) / (np.linalg.norm(
            Gamma_old) + 1e-12)
        if verbose:
            print(f" ipca_als iter {it}: rel_change {rel_change:.3e}")
        if rel_change < tol:
            break
    F_df = pd.DataFrame(F, index=r_panel.index, columns=[
        f"F{i+1}" for i in range(K)])
    return Gamma, F_df

# ---------------------------
# Load Fama-French data helper functions
def clean(df):
    """
    Clean FF data: index to datetime, columns to numeric, div by 100
    """
    df.index = df.index.astype(str).str.strip()
    #use index clean data-dates
    mask_len6 = df.index.str.len() == 6
    df = df.loc[mask_len6]
    # transform to datetime set end of month as timestamp
    df.index = pd.to_datetime(df.index, format='%Y%m').to_period('M').to_timestamp('M')
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).div(100)
    # clean column names
    df.columns = df.columns.str.strip()
    return df

def pca_svd_on_covariance(panel_df, k_max=6):
    """
    Performs PCA on a dense panel using SVD on the covariance matrix.
    This is a robust method suitable for the x_t (managed portfolios) panel,
    which should not have missing values but might have numerical issues.
    It resolves the 'SVD did not converge' error by ensuring a clean
    covariance matrix is used for decomposition.
    """
    print("\nRunning PCA on x_t panel (stable covariance method)...")
    results = []

    # Center the data
    col_means = panel_df.mean(axis=0, skipna=True)
    centered_df = panel_df.subtract(col_means, axis=1)

    if centered_df.empty:
        print("Warning: x_t panel is empty, skipping PCA.")
        return results
    # use pandas' built-in .cov() method, which handles NaNs correctly.
    cov_matrix = centered_df.cov()
    # as a final safeguard, fill any NaNs in the covariance matrix with 0.
    cov_matrix.fillna(0, inplace=True)
    
    try:
        # perform SVD on the clean covariance matrix to get eigenvectors(loadings)
        U, s, _ = svd(cov_matrix)
    except np.linalg.LinAlgError as e:
        print(f"SVD failed even after cleaning covariance matrix: {e}")
        return results

    for k in range(1, k_max + 1):
        # Loadings are the first k eigenvectors
        loadings_xt = U[:, :k]
        # Factors are projected from the data using the loadings
        factors_xt = centered_df.fillna(0).values @ loadings_xt
        # Reconstruct the data for Total R²
        recon_center = factors_xt @ loadings_xt.T
        recon_full = recon_center + col_means.values.reshape(1, -1)
        total_r2_xt = calculate_r2(panel_df.values, recon_full)
        # Calculate Predictive R²
        lambda_hat_xt = np.nanmean(factors_xt, axis=0)
        pred_r2_xt = pred_r2_using_lambda(panel_df, loadings_xt, lambda_hat_xt)
        T_xt, N_xt = panel_df.shape
        Np_xt = (N_xt * k) + (T_xt * k)
        results.append({
            'K': k, 'Total R²_xt': total_r2_xt, 'Pred. R²_xt': pred_r2_xt, 'Np_xt': Np_xt})
        print(
            f" x_t k={k}: Total R2={total_r2_xt:.4f}%  Pred R2={pred_r2_xt:.4f}%  Np={Np_xt}")
    return results

def load_ff_factors(y_index=None):
    ff5_fname, mom_fname = (
        'F-F_Research_Data_5_Factors_2x3.csv', 'F-F_Momentum_Factor dl.csv')
    if not (os.path.exists(ff5_fname) and os.path.exists(mom_fname)):
        print(f"ERROR: Missing factor files.")
        return pd.DataFrame()
    ff = pd.read_csv(ff5_fname, skiprows=3, index_col=0)
    mom = pd.read_csv(mom_fname, skiprows=13, index_col=0)
    ff, mom = clean(ff), clean(mom)
    ff['MOM'] = mom.iloc[:, 0]
    if y_index is not None:
        common_index = pd.to_datetime(sorted(set(y_index)))
        ff = ff.reindex(common_index)
    return ff


# ---------------------------
# Main script --- all calculations will be done here
def main():
    print_banner("TABLE II FULL REPLICATION SCRIPT START")
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

    # If some chara. not present, try compute or warn, keep what's available
    available_cols = [v for v in characteristic_mapping.values() if v in df.columns]
    reverse_map = {v:k for k,v in characteristic_mapping.items()}
    if len(available_cols) < len(characteristic_mapping):
        print("Warning: not all 36 characteristics present. Proceeding with available subset.")
    X = df[available_cols].copy()
    # rename columns to short names
    X.columns = [reverse_map[c] for c in available_cols]

    # winsorize optional
    if USE_WINSORIZE:
        print("Applying winsorization to returns...")
        y = y.groupby(level='eom').transform(winsorize_series)

    # rank-transform characteristics (per month cross-section)
    print("Rank-transforming characteristics (monthly cross-section to [-0.5,0.5]) ...")
    X_ranked = rank_transform(X)

    # final sample selection: 
    # only keep observations with y and all selected X non-missing
    # To be flexible:
    # we keep observations where y is present and
    # at least one characteristic present (but for strict match, require all)
    strict_all36 = False 
    if strict_all36:
        valid_mask = y.notna() & X_ranked.notna().all(axis=1)
    else:
        valid_mask = y.notna()  # keep those with returns; ALS will handle missing X
    y = y[valid_mask]
    X_ranked = X_ranked[valid_mask]

    print(f"Final sample observations: {len(y)}")

    # load FF factors
    ff_df = load_ff_factors()
    K_vals = [1, 3, 4, 5, 6]
    factor_sets = {
        1: ['Mkt-RF'],
        3: ['Mkt-RF', 'SMB', 'HML'],
        4: ['Mkt-RF', 'SMB', 'HML', 'MOM'],
        5: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
        6: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    }

    # ---------------------------
    # Panel B: Observable Factors (Static Betas)
    print_banner("Panel B: Observable Factors (Static Betas)")
    results_b = []
    if ff_df.empty:
        print("FF factors not loaded; skipping Panel B.")
    else:
        # prepare panel for time series regressions per asset
        panel_b = y.reset_index().rename(columns={'id':'permno','eom':'date'})
        panel_b['date'] = pd.to_datetime(panel_b[
            'date']).dt.to_period('M').dt.to_timestamp('M')
        # merge with FF full factors to count obs
        full_factors = factor_sets[6]
        panel_b = panel_b.merge(ff_df[
            full_factors], left_on='date', right_index=True, how='left')
        # compute per-asset valid counts (>=60)
        valid_counts = panel_b.dropna(subset=[
            'ret_exc_lead1m'] + full_factors).groupby('permno').size()
        valid_permnos = valid_counts[valid_counts >= 60].index.tolist()
        panel_b_clean = panel_b[panel_b['permno'].isin(valid_permnos)].copy()
        # cross-sectional demeaning for Panel B (returns only)
        panel_b_clean['ret_exc_lead1m'] = panel_b_clean[
            'ret_exc_lead1m'] - panel_b_clean.groupby('date')[
                'ret_exc_lead1m'].transform('mean')

        # estimate betas 
        betas = {}
        for permno, grp in tqdm(panel_b_clean.groupby('permno'), desc="Estimating betas Panel B"):
            grp2 = grp.dropna(subset=full_factors + ['ret_exc_lead1m'])
            if grp2.shape[0] < 60: 
                continue
            X_ts = grp2[full_factors].values
            y_ts = grp2['ret_exc_lead1m'].values
            try:
                b, *_ = lstsq(X_ts, y_ts, rcond=None)
                betas[permno] = pd.Series(b, index=full_factors)
            except Exception:
                continue
        betas_df = pd.DataFrame.from_dict(betas, orient='index')

        # for each K compute R2 on r_t and on x_t
        ids = y.index.get_level_values('id').unique()
        dates = y.index.get_level_values('eom').unique()
        # construct xt for each date for later use
        # xt_panel will be built below in a shared section
        for k in K_vals:
            factors_k = factor_sets[k]
            # merge betas for only factors_k columns
            if betas_df.empty:
                continue
            betas_k = betas_df.loc[:, betas_df.columns.intersection(factors_k)]
            # prepare merged data for predictions: join betas to each observation
            merged = panel_b_clean.merge(
                betas_k, left_on='permno', right_index=True, how='inner',
                suffixes=('','_beta'))
            beta_cols = [c + '_beta' for c in factors_k]
            merged = merged.dropna(subset=factors_k + beta_cols)
            y_true = merged['ret_exc_lead1m'].values
            beta_arr = merged[beta_cols].values
            fact_arr = merged[factors_k].values
            y_hat = np.sum(beta_arr * fact_arr, axis=1)
            total_r2_rt = calculate_r2(y_true, y_hat)
            # predictive: lambda_hat = time mean of factors
            lam = np.nanmean(fact_arr, axis=0)
            # stock-level average returns
            permno_vec = merged['permno'].values
            y_true_df = pd.DataFrame({'permno': permno_vec, 'ret': y_true})
            actual_rt_means = y_true_df.groupby('permno')['ret'].mean()
            
            # --- Predictive R² for r_t ---
            predicted_rt_means = betas_k.values @ lam
            predicted_rt_means_series = pd.Series(
                predicted_rt_means, index=betas_k.index)
            common_stocks = actual_rt_means.index.intersection(
                predicted_rt_means_series.index)
            pred_r2_rt = calculate_r2(actual_rt_means.loc[
                common_stocks], predicted_rt_means_series.loc[common_stocks])

            # xt: instrumented managed portfolios using X_ranked 
            results_b.append({
                'K':k, 'Total R²_rt': total_r2_rt, 'Pred. R²_rt': pred_r2_rt, 
                'Np_rt': len(betas_k)*k,
                'Total R²_xt': np.nan, 'Pred. R²_xt': np.nan, 'Np_xt': np.nan})

    # ---------------------------
    # Build xt (managed portfolios) used in Panels B (xt), C, D (xt)
    # xt_t = Z_t' * r_t  (Z_t are instruments for date t)
    print_banner("Constructing managed portfolios x_t")
    # join X_ranked with returns to compute per-date
    # reindex X_ranked to (id,eom) same as y
    X_ranked = X_ranked.sort_index()
    # make temp frame to group by date
    tmp = X_ranked.copy()
    # attach returns
    tmp = tmp.join(y.rename('ret'), how='left')
    tmp['const'] = 1.0
    instrument_cols = ['const'] + list(X_ranked.columns)
    dates_order = []
    xt_rows = []
    for date, grp in tmp.groupby(level='eom'):
        dates_order.append(date)
        Zt = grp[instrument_cols].values  # n_t x L
        rt = grp['ret'].values            # n_t
        mask_rt = ~np.isnan(rt)
        if mask_rt.sum() == 0:
            xt_rows.append(np.full((len(instrument_cols),), np.nan))
        else:
            Zt_clean = Zt[mask_rt, :]
            rt_clean = rt[mask_rt]
            xt_rows.append(Zt_clean.T @ rt_clean)
    xt_panel = pd.DataFrame(
        xt_rows, index=dates_order, columns=instrument_cols).sort_index()
    # center xt by column (ignore nulls)
    xt_col_means = xt_panel.mean(axis=0, skipna=True)
    xt_centered = xt_panel.subtract(xt_col_means, axis=1)
    # Now fill results_b xt entries (if any)
    if 'results_b' in locals() and len(results_b) > 0:
        min_obs = 60 # Define the minimum number of observation months for regression
        for entry in results_b:
            k = entry['K']
            if ff_df.empty:
                entry.update({'Total R²_xt': np.nan, 'Pred. R²_xt': np.nan, 'Np_xt': np.nan})
                continue
            factors_k = factor_sets[k]
            # Estimate betas for each instrument in xt_panel
            betas_xt_df = pd.DataFrame(
                index=instrument_cols, columns=factors_k, dtype=float)
            # Loop through each instrument to estimate its beta
            for inst in instrument_cols:
                series = xt_panel[inst].dropna()
                if len(series) < min_obs:
                    continue # Not enough data for this instrument
                ff_sub = ff_df.loc[series.index, factors_k]
                common_idx = series.index.intersection(ff_sub.index)
                if len(common_idx) < min_obs:
                    continue
                    
                X_ts = ff_sub.loc[common_idx].values
                y_ts = series.loc[common_idx].values
                
                try:
                    b, *_ = lstsq(X_ts, y_ts, rcond=None)
                    betas_xt_df.loc[inst] = b # Store the estimated betas
                except Exception:
                    continue
            
            # For R-squared calculation, drop regression failed instruments
            betas_xt_df_clean = betas_xt_df.dropna()
            
            if betas_xt_df_clean.empty:
                entry.update({'Total R²_xt': np.nan, 'Pred. R²_xt': np.nan})
            else:
                # --- Calculate Pred. R² for x_t ---
                lambda_hat = ff_df[factors_k].mean(axis=0).values
                pred_xt_mean = (betas_xt_df_clean.values @ lambda_hat)
                xt_mean_clean = xt_panel.mean(axis=0).reindex(
                    betas_xt_df_clean.index).values
                entry['Pred. R²_xt'] = calculate_r2(xt_mean_clean, pred_xt_mean)

                # --- Calculate Total R² for x_t ---
                obs_list, pred_list = [], []
                for date in xt_panel.index:
                    row_obs = xt_panel.loc[date].reindex(
                        betas_xt_df_clean.index)
                    if row_obs.isna().all() or date not in ff_df.index:
                        continue
                    
                    fvals = ff_df.loc[date, factors_k].values
                    row_pred = betas_xt_df_clean.values @ fvals
                    
                    valid_mask = ~np.isnan(row_obs.values)
                    if np.sum(valid_mask) > 0:
                        obs_list.append(row_obs.values[valid_mask])
                        pred_list.append(row_pred[valid_mask])

                if len(obs_list) == 0:
                    entry['Total R²_xt'] = np.nan
                else:
                    obs_arr = np.concatenate(obs_list)
                    pred_arr = np.concatenate(pred_list)
                    entry['Total R²_xt'] = calculate_r2(obs_arr, pred_arr)

            # (Critical Fix) Calculate Np from the number of instruments
            entry['Np_xt'] = len(instrument_cols) * k

    # ---------------------------
    # Panel C: Observable Factors with instruments (instrumented betas)
    print_banner("Panel C: Observable Factors (with instruments)")
    results_c = []
    if ff_df.empty:
        print("FF factors missing; skipping Panel C.")
    else:
        # Build panel data merged with instruments Z (const + X_ranked)
        panel_c = y.reset_index().rename(columns={'id':'permno','eom':'date'})
        panel_c['date'] = pd.to_datetime(
            panel_c['date']).dt.to_period('M').dt.to_timestamp('M')
        panel_c = panel_c.merge(
            ff_df, left_on='date', right_index=True, how='inner')
        # merge instruments
        Z_df = X_ranked.reset_index().rename(
            columns={'id':'permno','eom':'date'})
        Z_df['date'] = pd.to_datetime(
            Z_df['date']).dt.to_period('M').dt.to_timestamp('M')
        Z_df['const'] = 1.0
        panel_c = panel_c.merge(Z_df, on=['permno','date'], how='inner')
        if 'valid_permnos' in locals() and len(valid_permnos) > 0:
            print(f"Filtering Panel C sample to {len(valid_permnos)} stocks from Panel B.")
            panel_c = panel_c[panel_c['permno'].isin(valid_permnos)].copy()
        # cross-sectional demeaning returns
        panel_c['ret_exc_lead1m'] = panel_c[
            'ret_exc_lead1m'] - panel_c.groupby('date')[
                'ret_exc_lead1m'].transform('mean')
        instrument_cols = ['const'] + list(X_ranked.columns)
        for k in K_vals:
            factors_k = factor_sets[k]
            all_required_cols = ['ret_exc_lead1m'] + factors_k + instrument_cols
            data_k = panel_c.dropna(subset=all_required_cols).copy()
            if data_k.shape[0] == 0:
                continue
            M = len(factors_k)
            L = len(instrument_cols)
            # first stage: regress r_t on Z_t * factor_m for m=1..M
            # build design matrix D (N_obs x (L*M))
            Zmat = data_k[instrument_cols].values
            Gmat = data_k[factors_k].values
            y_vec = data_k['ret_exc_lead1m'].values
            D = np.hstack([np.nan_to_num(Zmat) * np.nan_to_num(
                Gmat[:, m].reshape(-1,1)) for m in range(M)])
            coef_vec, *_ = lstsq(D, np.nan_to_num(y_vec), rcond=None)
            Gamma_delta = coef_vec.reshape(M, L).T  # L x M
            # second stage: predict r_t_hat = D * coef_vec
            # evaluate Total R² and Pred. R² for r_t and x_t
            # Np = number of estimated parameters = L * M
            # Note: we do NOT re-estimate betas per stock as in Panel B
            # --- Total R² for r_t ---
            y_pred = D @ coef_vec
            total_r2_rt = calculate_r2(y_vec, y_pred)
            
            # --- Pred. R² for r_t ---
            lambda_hat = np.nanmean(Gmat, axis=0)
            y_pred_pred = (np.nan_to_num(Zmat) @ Gamma_delta) @ lambda_hat
            # Correctly compare the means for each stock
            permno_vec_c = data_k.reset_index()['permno'].values
            y_true_df_c = pd.DataFrame(
                {'permno': permno_vec_c, 'ret': y_vec, 'pred': y_pred_pred})
            actual_rt_means_c = y_true_df_c.groupby('permno')['ret'].mean()
            predicted_rt_means_c = y_true_df_c.groupby('permno')['pred'].mean()
            common_stocks_c = actual_rt_means_c.index.intersection(
                predicted_rt_means_c.index)
            pred_r2_rt = calculate_r2(
                actual_rt_means_c.loc[common_stocks_c], 
                predicted_rt_means_c.loc[common_stocks_c])
            
            # --- Build x_t time series ---
            y_pred_series = pd.Series(y_pred, index=data_k.index)
            y_pred_pred_series = pd.Series(y_pred_pred, index=data_k.index)
            xt_obs_list2, xt_pred_list2, xt_pred_pred_list2 = [], [], []

            for date, grp in data_k.groupby('date'):
                Zt = np.nan_to_num(grp[instrument_cols].values)
                rt1 = np.nan_to_num(grp['ret_exc_lead1m'].values)
                y_pred_sub = np.nan_to_num(y_pred_series.loc[grp.index].values)
                y_pred_pred_sub = np.nan_to_num(y_pred_pred_series.loc[grp.index].values)
                
                xt_obs_list2.append(Zt.T @ rt1)
                xt_pred_list2.append(Zt.T @ y_pred_sub)
                xt_pred_pred_list2.append(Zt.T @ y_pred_pred_sub)

            # --- Total and Pred. R² for x_t ---
            if len(xt_obs_list2) > 0:
                total_r2_xt = calculate_r2(
                    np.concatenate(xt_obs_list2), np.concatenate(xt_pred_list2))
                
                xt_obs_arr = np.array(xt_obs_list2)
                xt_pred_pred_arr = np.array(xt_pred_pred_list2)
                actual_xt_means = np.nanmean(xt_obs_arr, axis=0)
                predicted_xt_means = np.nanmean(xt_pred_pred_arr, axis=0)
                pred_r2_xt = calculate_r2(actual_xt_means, predicted_xt_means)
            else:
                total_r2_xt = np.nan
                pred_r2_xt = np.nan

            # Append results for Panel C
            results_c.append({'K':k, 'Total R²_rt':total_r2_rt,
                            'Pred. R²_rt':pred_r2_rt,'Np_rt':D.shape[1],
                            'Total R²_xt':total_r2_xt,'Pred. R²_xt':pred_r2_xt,
                            'Np_xt':D.shape[1]})

    # ---------------------------
    # Panel D: PCA on returns (r_t) and on xt
    print_banner("Panel D: PCA (ALS) on returns r_t and on managed portfolios x_t")
    # build returns panel (T x N)
    returns_panel = y.unstack(
        level='id').reindex(sorted(y.index.get_level_values('eom').unique()))
    returns_panel = returns_panel.sort_index(axis=0)
    T_rt, N_rt = returns_panel.shape
    results_d_rt = []
    for k in range(1,7):
        # column means ignoring NaN
        col_means = returns_panel.mean(axis=0, skipna=True)
        returns_centered = returns_panel.subtract(col_means, axis=1)
        fac_df, load_df, expl_var, nit = pca_als(
            returns_centered, k, tol=ALS_TOL, 
            max_iter=ALS_MAX_ITER, verbose=False)
        # reconstruct and add back column means
        recon_center = fac_df.values @ load_df.values.T
        recon_full = recon_center + col_means.values.reshape(1, -1)
        recon_df = pd.DataFrame(recon_full, index=returns_panel.index, 
                                columns=returns_panel.columns)
        total_r2_rt = calculate_r2(returns_panel.values, recon_df.values)
        lambda_hat = np.nanmean(fac_df.values, axis=0)
        pred_r2_rt = pred_r2_using_lambda(
            returns_panel, load_df.values, lambda_hat)
        Np_rt = (N_rt * k) + (T_rt * k)
        results_d_rt.append({'K':k, 'Total R²_rt': total_r2_rt, 
                            'Pred. R²_rt': pred_r2_rt, 'Np_rt':Np_rt,
                            'explained_var': expl_var.tolist()
                            if hasattr(expl_var,'tolist') 
                            else list(expl_var), 'n_iter':nit})
        print(f" r_t k={k}: Total R2={total_r2_rt:.4f}%  Pred R2={pred_r2_rt:.4f}%  Np={Np_rt}  iters={nit}")

    print("\nRunning PCA on x_t panel...")
    results_d_xt = []
    results_d_xt = pca_svd_on_covariance(xt_panel, k_max=6)

    # ---------------------------
    # Summarize Table II --- assemble results AND print
    print_banner("Assembling Table II")
    # Build DataFrames
    df_b = pd.DataFrame(results_b).set_index('K') if len(results_b)>0 else pd.DataFrame()
    df_c = pd.DataFrame(results_c).set_index('K') if len(results_c)>0 else pd.DataFrame()
    df_d_rt = pd.DataFrame(results_d_rt).set_index('K') if len(results_d_rt)>0 else pd.DataFrame()
    df_d_xt = pd.DataFrame(results_d_xt).set_index('K') if len(results_d_xt)>0 else pd.DataFrame()

    # MultiIndex table similar to paper (Model, Asset, Statistic)
    idx = pd.MultiIndex.from_tuples([], names=['Model','Asset','Statistic'])
    table2 = pd.DataFrame(index=idx)
    # add Panel B
    for k in K_vals:
        if k in df_b.index:
            table2.loc[('Obs Factors','r_t','Total R²'), k] = df_b.loc[k,'Total R²_rt']
            table2.loc[('Obs Factors','r_t','Pred. R²'), k] = df_b.loc[k,'Pred. R²_rt']
            table2.loc[('Obs Factors','r_t','Np'), k] = df_b.loc[k,'Np_rt']
            table2.loc[('Obs Factors','x_t','Total R²'), k] = df_b.loc[k,'Total R²_xt']
            table2.loc[('Obs Factors','x_t','Pred. R²'), k] = df_b.loc[k,'Pred. R²_xt']
            table2.loc[('Obs Factors','x_t','Np'), k] = df_b.loc[k,'Np_xt']
    # Panel C
    for k in K_vals:
        if k in df_c.index:
            table2.loc[('Obs Factors Instr','r_t','Total R²'), k] = df_c.loc[k,'Total R²_rt']
            table2.loc[('Obs Factors Instr','r_t','Pred. R²'), k] = df_c.loc[k,'Pred. R²_rt']
            table2.loc[('Obs Factors Instr','r_t','Np'), k] = df_c.loc[k,'Np_rt']
            table2.loc[('Obs Factors Instr','x_t','Total R²'), k] = df_c.loc[k,'Total R²_xt']
            table2.loc[('Obs Factors Instr','x_t','Pred. R²'), k] = df_c.loc[k,'Pred. R²_xt']
            table2.loc[('Obs Factors Instr','x_t','Np'), k] = df_c.loc[k,'Np_xt']
    # Panel D
    for k in range(1,7):
        if k in df_d_rt.index:
            table2.loc[('PCA','r_t','Total R²'), k] = df_d_rt.loc[k,'Total R²_rt']
            table2.loc[('PCA','r_t','Pred. R²'), k] = df_d_rt.loc[k,'Pred. R²_rt']
            table2.loc[('PCA','r_t','Np'), k] = df_d_rt.loc[k,'Np_rt']
        if k in df_d_xt.index:
            table2.loc[('PCA','x_t','Total R²'), k] = df_d_xt.loc[k,'Total R²_xt']
            table2.loc[('PCA','x_t','Pred. R²'), k] = df_d_xt.loc[k,'Pred. R²_xt']
            table2.loc[('PCA','x_t','Np'), k] = df_d_xt.loc[k,'Np_xt']

    # print nicely
    if ('Obs Factors' in table2.index.get_level_values(0)):
        print("\n--- Panel B: Observable Factors (no instruments) ---")
        try:
            print(table2.loc['Obs Factors'].dropna(axis=1, how='all').round(2))
        except Exception:
            pass
    if ('Obs Factors Instr' in table2.index.get_level_values(0)):
        print("\n--- Panel C: Observable Factors (with instruments) ---")
        try:
            print(table2.loc['Obs Factors Instr'].dropna(axis=1, how='all').round(2))
        except Exception:
            pass
    print("\n--- Panel D: PCA (returns r_t and x_t) ---")
    try:
        print(table2.loc['PCA'].dropna(axis=1, how='all').round(2))
    except Exception:
        pass

    print_banner("TABLE II REPLICATION COMPLETE")

if __name__ == "__main__":
    main()
