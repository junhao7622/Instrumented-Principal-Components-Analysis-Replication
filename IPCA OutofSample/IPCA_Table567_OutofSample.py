"""
Replication script on OutSample OOS IPCA-model,strictly follow methodology of Kelly, Pruitt & Su (2019)
2
Table V reports OOS predictive performance, comparing IPCA to benchmark models.
Metrics include OOS R square and mean squared error.
IPCA consistently delivers higher OOS R square, higher predictive frequency.

Table VI Panel A and B report in-sample and out-of-sample total
and predictive R square for subsamples of large and small stocks.

Table VII reports total and predictive R2 for large and small stock subsamples using parameters
estimated separately in each subsample. Rows correspond to the sample from which parameters are estimated
and columns represent the sample in which fits are evaluated.

AUTHOR: Junhao Gao
DATE: 2025-10
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats.mstats import winsorize
from refactored_Table1 import build_panel_matrices
from ipca import InstrumentedPCA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
# CONFIGURATION

CONFIG = {
    'Kmax': 6,
    'min_train_obs': 1000,
    'keep_dates_with_at_least': 1000,
    'standardize': 'rank',
    'include_intercept': False,
    'winsor_quantile': 0.005,
    'max_factor': 2,
    'start_date': '1963-07-01',
    'end_date': '2014-05-31',
    'initial_train_months': 120,
    'oos_start_date': '1987-01-01', # OOS starts use T>=2, suppose to be around 1984
    'n_jobs': -1,
    'verbose': True,
}

PAPER_CHARACTERISTICS = [
    'beta', 'a2me', 'at', 'ato', 'beme', 'c', 'cto', 'd2a', 'dpi2a', 'e2p',
    'fc2y', 'free_cf', 'idio_vol', 'investment', 'lev', 'size', 'lturnover',
    'noa', 'oa', 'ol', 'pcm', 'pm', 'prof', 'q', 'rel_high', 'rna', 'roa',
    'roe', 'mom_12_2', 'mom_12_7', 'mom_1_0', 'mom_36_13', 's2p', 'sga2m',
    'spread_mean', 'suv'
]

#-------------------------------------------------------------------------------
# CORE INSTRUMENT CONSTRUCTION (PAPER SECTION 4.1)

def build_instruments_insample(
    df: pd.DataFrame,
    char_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build In-Sample instruments following paper Section 4.1.
    
    For each characteristic c:
    1. Rank-transform cross-sectionally per date → c_ranked ∈ [-0.5, 0.5]
    2. Compute FULL-SAMPLE mean: c̄_i = mean over all t
    3. Compute deviation: dev = c_ranked - c̄_i
    4. Rank-transform both mean and deviation
    
    Returns z_{i,t} = [1, c̄_i (ranked), (c - c̄_i) (ranked)]
    Dimension: 2L+1 for L characteristics
    
    Used in: Table VI In-Sample, Table VII
    """
    df_work = df.copy().sort_values(['permno', 'date'])
    instrument_cols = []
    
    logger.info(f"Building IN-SAMPLE instruments for {len(char_cols)} characteristics...")
    
    for char in char_cols:
        if char not in df_work.columns:
            continue
        
        # Step 1: Rank-transform original characteristic
        df_work[f'{char}_ranked'] = (
            df_work.groupby('date')[char]
            .rank(method='average', na_option='keep') /
            df_work.groupby('date')[char].transform('count')
        ) - 0.5
        
        # Step 2: Compute FULL-SAMPLE mean (fixed for each stock)
        char_mean = df_work.groupby('permno')[f'{char}_ranked'].transform('mean')
        df_work[f'{char}_mean'] = char_mean
        
        # Step 3: Compute deviation
        df_work[f'{char}_dev'] = df_work[f'{char}_ranked'] - df_work[f'{char}_mean']
        
        # Step 4: Rank-transform mean and deviation cross-sectionally
        df_work[f'{char}_mean_rank'] = (
            df_work.groupby('date')[f'{char}_mean']
            .rank(method='average', na_option='keep') /
            df_work.groupby('date')[f'{char}_mean'].transform('count')
        ) - 0.5
        
        df_work[f'{char}_dev_rank'] = (
            df_work.groupby('date')[f'{char}_dev']
            .rank(method='average', na_option='keep') /
            df_work.groupby('date')[f'{char}_dev'].transform('count')
        ) - 0.5
        
        instrument_cols.extend([f'{char}_mean_rank', f'{char}_dev_rank'])
        
        # Cleanup intermediate columns
        df_work = df_work.drop([f'{char}_ranked', f'{char}_mean', f'{char}_dev'], axis=1)
    
    # Add constant
    df_work['const'] = 1.0
    instrument_cols.append('const')
    
    # Remove NaN
    df_work = df_work.dropna(subset=instrument_cols)
    
    logger.info(f"  Created {len(instrument_cols)} instruments (2L+1 structure)")
    
    return df_work, instrument_cols


def build_instruments_oosample(
    df: pd.DataFrame,
    char_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build Out-of-Sample instruments following paper Section 4.1.
    
    For each characteristic c:
    1. Rank-transform cross-sectionally per date → c_ranked ∈ [-0.5, 0.5]
    2. Compute EXPANDING mean: c̄_{i,t} = mean from τ=1 to t
    3. Compute deviation: dev = c_ranked - c̄_{i,t}
    4. Rank-transform both mean and deviation
    
    Returns z_{i,t} = [1, c̄_{i,t} (ranked), (c - c̄_{i,t}) (ranked)]
    Dimension: 2L+1 for L characteristics
    
    Used in: Table V, Table VI Out-of-Sample
    """
    df_work = df.copy().sort_values(['permno', 'date'])
    instrument_cols = []
    
    logger.info(f"Building OUT-OF-SAMPLE instruments for {len(char_cols)} characteristics...")
    
    for char in char_cols:
        if char not in df_work.columns:
            continue
        
        # Step 1: Rank-transform original characteristic
        df_work[f'{char}_ranked'] = (
            df_work.groupby('date')[char]
            .rank(method='average', na_option='keep') /
            df_work.groupby('date')[char].transform('count')
        ) - 0.5
        
        # Step 2: Compute EXPANDING mean (dynamic, changes each period)
        char_mean = (
            df_work.groupby('permno')[f'{char}_ranked']
            .expanding().mean()
            .reset_index(0, drop=True)
        )
        df_work[f'{char}_mean'] = char_mean
        
        # Step 3: Compute deviation
        df_work[f'{char}_dev'] = df_work[f'{char}_ranked'] - df_work[f'{char}_mean']
        
        # Step 4: Rank-transform mean and deviation cross-sectionally
        df_work[f'{char}_mean_rank'] = (
            df_work.groupby('date')[f'{char}_mean']
            .rank(method='average', na_option='keep') /
            df_work.groupby('date')[f'{char}_mean'].transform('count')
        ) - 0.5
        
        df_work[f'{char}_dev_rank'] = (
            df_work.groupby('date')[f'{char}_dev']
            .rank(method='average', na_option='keep') /
            df_work.groupby('date')[f'{char}_dev'].transform('count')
        ) - 0.5
        
        instrument_cols.extend([f'{char}_mean_rank', f'{char}_dev_rank'])
        
        # Cleanup intermediate columns
        df_work = df_work.drop([f'{char}_ranked', f'{char}_mean', f'{char}_dev'], axis=1)
    
    # Add constant
    df_work['const'] = 1.0
    instrument_cols.append('const')
    
    # Remove NaN
    df_work = df_work.dropna(subset=instrument_cols)
    
    logger.info(f"  Created {len(instrument_cols)} instruments (2L+1 structure)")
    
    return df_work, instrument_cols


def run_ipca_with_proper_instruments(
    df: pd.DataFrame,
    instrument_cols: List[str],
    n_factors: int,
    keep_dates_with_at_least: int = 1000
) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """
    Run IPCA with pre-constructed instruments.
    Returns: (total_R2, predictive_R2, Gamma_df, Factors_df)
    """
    # Build panel matrices
    X, y = build_panel_matrices(df, instrument_cols, keep_dates_with_at_least)
    
    if len(X) < 1000:
        raise ValueError("Insufficient data after filtering")
    
    # Get unique entities and dates
    entity_ids = X.index.get_level_values('permno').unique()
    dates = X.index.get_level_values('date').unique()
    
    # Create indices array for IPCA
    indices = np.column_stack([
        X.index.get_level_values('permno').map({e: i for i, e in enumerate(entity_ids)}),
        X.index.get_level_values('date').map({d: i for i, d in enumerate(dates)})
    ])
    
    # Fit IPCA
    ipca = InstrumentedPCA(
        n_factors=n_factors,
        intercept=False,
        max_iter=10000,
        iter_tol=1e-6
    )
    ipca.fit(X.values, y.values, indices=indices)
    
    # Extract results
    Gamma = ipca.Gamma
    Factors = ipca.Factors
    
    # Create DataFrames
    Gamma_df = pd.DataFrame(Gamma, index=instrument_cols, columns=[f'F{i+1}' for i in range(n_factors)])
    Factors_df = pd.DataFrame(Factors.T, index=dates, columns=[f'F{i+1}' for i in range(n_factors)])
    
    # Calculate R²
    y_pred = ipca.predict(X.values, indices=indices, mean_factor=False)
    y_actual = y.values
    
    # Total R²
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_actual)
    ss_res = np.sum((y_actual[valid_mask] - y_pred[valid_mask])**2)
    ss_tot = np.sum(y_actual[valid_mask]**2)
    total_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Predictive R²
    lambda_mean = Factors_df.mean().values
    Z = X.values
    Gamma_beta = Gamma[:, :n_factors]
    mu_pred = Z @ Gamma_beta @ lambda_mean
    
    ss_res_pred = np.sum((y_actual[valid_mask] - mu_pred[valid_mask])**2)
    pred_r2 = 1 - ss_res_pred / ss_tot if ss_tot > 0 else 0.0
    
    return total_r2, pred_r2, Gamma_df, Factors_df

#-------------------------------------------------------------------------------
# CORE UTILITIES

def load_ff_factors(
    ff5_csv: str = "F-F_Research_Data_5_Factors_2x3.csv",
    mom_csv: str = "F-F_Momentum_Factor dl.csv"
) -> pd.DataFrame:
    """Load and clean Fama-French 5 factors + Momentum."""
    
    def clean_ff(df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        
        idx = df.index.astype(str).str.strip()
        mask_monthly = idx.str.fullmatch(r"\d{6}")
        df = df[mask_monthly]
        
        df.index = pd.to_datetime(df.index, format="%Y%m", errors="coerce")
        df = df.dropna() / 100.0
        return df
    
    if not os.path.exists(ff5_csv) or not os.path.exists(mom_csv):
        logger.warning("FF factor files not found")
        return pd.DataFrame()
    
    ff5 = clean_ff(pd.read_csv(ff5_csv, index_col=0, skiprows=3))
    mom = clean_ff(pd.read_csv(mom_csv, index_col=0, skiprows=13))
    mom.columns = ['MOM']
    
    ff_factors = pd.concat([ff5, mom], axis=1)
    
    start = pd.to_datetime('1963-07-31')
    end = pd.to_datetime('2014-04-30')
    ff_factors = ff_factors[(ff_factors.index >= start) & (ff_factors.index <= end)]
    
    return ff_factors


def align_dates_to_month_period(
    factors_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Align factor dates to target dates using month-period matching."""
    if factors_df.empty or len(target_dates) == 0:
        raise ValueError("Empty input for date alignment")
    
    factors_period = factors_df.index.to_period('M')
    target_period = target_dates.to_period('M')
    
    common_periods = factors_period.intersection(target_period)
    
    if len(common_periods) == 0:
        raise ValueError(
            f"No overlapping periods!\n"
            f"  Factors: {factors_period[0]} to {factors_period[-1]}\n"
            f"  Target: {target_period[0]} to {target_period[-1]}"
        )
    
    factors_aligned = factors_df[factors_period.isin(common_periods)].copy()
    factors_aligned.index = common_periods.to_timestamp()
    
    return factors_aligned


def match_features(
    source_features: List[str],
    target_features: List[str],
    method: str = 'direct'
) -> List[str]:
    """Unified feature matching logic."""
    matched = [f for f in source_features if f in target_features]
    
    if method == 'direct':
        return matched
    
    if len(matched) > 0:
        return matched
    
    # Flexible matching (for rank-transformed features)
    target_base_map = {}
    for feat in target_features:
        base = feat.replace('_mean_rank', '').replace('_dev_rank', '').replace('_rank', '')
        if base not in target_base_map:
            target_base_map[base] = []
        target_base_map[base].append(feat)
    
    matched_flexible = []
    for source_feat in source_features:
        source_base = source_feat.replace('_mean_rank', '').replace('_dev_rank', '').replace('_rank', '')
        
        if source_feat in target_features:
            matched_flexible.append(source_feat)
            continue
        
        if source_base in target_base_map:
            if '_mean_rank' in source_feat:
                candidates = [f for f in target_base_map[source_base] if '_mean_rank' in f]
            elif '_dev_rank' in source_feat:
                candidates = [f for f in target_base_map[source_base] if '_dev_rank' in f]
            else:
                candidates = target_base_map[source_base]
            
            if candidates:
                matched_flexible.append(candidates[0])
    
    return list(set(matched_flexible))

#-------------------------------------------------------------------------------
# DATA LOADING & CLEANING

def load_data(file_path: str) -> pd.DataFrame:
    """Load characteristics data with proper formatting."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    return df


def clean_data(
        df: pd.DataFrame,
        apply_rank_transform: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete data cleaning process.

    Parameters
    ----------
    apply_rank_transform : If False, returns raw characteristics (for subsample re-ranking)

    Returns
    -------
    df_clean : Cleaned DataFrame
    char_cols : List of characteristic column names
    """
    # Column renaming (standardize names)
    col_alias = {
        'cum_return_12_2': 'mom_12_2',
        'cum_return_12_7': 'mom_12_7',
        'cum_return_1_0': 'mom_1_0',
        'cum_return_36_13': 'mom_36_13',
        'lme': 'size',
        'rel_to_high_price': 'rel_high'
    }
    df = df.rename(columns=col_alias)

    # Convert date and apply filters
    df['date'] = pd.to_datetime(df['date'])
    START_DATE = pd.to_datetime(CONFIG['start_date'])
    END_DATE = pd.to_datetime(CONFIG['end_date'])
    df = df.loc[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)].copy()

    # Check available characteristics
    available_cols = [
        col for col in df.columns
        if col not in ['permco', 'date', 'ret', 'ret_lead', 'gvkey', 'lpermno']
           and not col.startswith('Unnamed')
    ]
    char_cols = [col for col in PAPER_CHARACTERISTICS if col in available_cols]

    logger.info(f"Available characteristics: {len(char_cols)}/{len(PAPER_CHARACTERISTICS)}")

    # Create future returns
    df = df.sort_values(['permno', 'date'])
    df['ret_lead'] = df.groupby('permno')['ret'].shift(-1)

    # Convert to excess returns
    ff_factors = load_ff_factors()
    if not ff_factors.empty:
        rf = ff_factors[['RF']].reset_index().rename(columns={'index': 'date_ff'})
        rf['year_month'] = rf['date_ff'].dt.to_period('M').dt.to_timestamp()
        df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df = df.merge(rf[['year_month', 'RF']], on='year_month', how='left')

        df['RF'] = df['RF'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        df['ret_lead'] = df['ret_lead'] - df['RF']
        df = df.drop(columns=['year_month', 'RF'])

    # Apply winsorization
    if CONFIG.get('winsor_quantile', 0) > 0:
        for col in char_cols:
            if col in df.columns:
                df[col] = winsorize(
                    df[col],
                    limits=[CONFIG['winsor_quantile'], CONFIG['winsor_quantile']]
                )

    # Apply rank transformation (if requested)
    if apply_rank_transform:
        pass

    # Clean final dataset
    df = df.dropna(subset=['ret_lead'])
    char_cols = [col for col in char_cols if col in df.columns]

    # Filter rows with complete characteristics
    mask_valid = df[char_cols].notna().all(axis=1)
    df = df.loc[mask_valid].copy()

    # Filter dates with sufficient observations
    count_by_date = df.groupby('date').size()
    valid_dates = count_by_date[
        count_by_date >= CONFIG['keep_dates_with_at_least']
        ].index
    df = df[df['date'].isin(valid_dates)].copy()

    # Select essential columns
    essential_cols = ['permno', 'date', 'ret_lead'] + char_cols
    df = df[essential_cols]

    logger.info(
        f"Cleaned data: {len(df):,} rows, {df['date'].nunique()} dates, "
        f"{len(char_cols)} characteristics"
    )

    return df, char_cols


def apply_rank_standardization(df: pd.DataFrame, char_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply simple rank-based standardization (legacy function for compatibility).
    """
    df_ranked = df.copy()
    
    for char in char_cols:
        if char not in df_ranked.columns:
            continue
        
        df_ranked[f'{char}_rank'] = (
            df_ranked.groupby('date')[char]
            .rank(method='average', na_option='keep') /
            df_ranked.groupby('date')[char].transform('count')
        ) - 0.5
    
    rank_cols = [f'{c}_rank' for c in char_cols if f'{c}_rank' in df_ranked.columns]
    return df_ranked, rank_cols

#-------------------------------------------------------------------------------
# SHARPE RATIO CALCULATION

def get_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 12:
        return np.nan
    
    monthly_mean = returns.mean()
    monthly_std = returns.std()
    
    if monthly_std == 0 or np.isnan(monthly_std):
        return np.nan
    
    annual_mean = monthly_mean * 12
    annual_std = monthly_std * np.sqrt(12)
    
    return annual_mean / annual_std


def get_tangency_sharpe_rolling(factor_returns: pd.DataFrame) -> float:
    """
    Calculate rolling tangency portfolio Sharpe ratio.
    
    Paper methodology: Construct portfolio using mean and covariance through t,
    then track t+1 return out-of-sample.
    """
    if factor_returns.empty or len(factor_returns) < 24:
        return np.nan
    
    # K=1 special case
    if factor_returns.shape[1] == 1:
        return get_sharpe(factor_returns.iloc[:, 0])
    
    try:
        portfolio_returns = []
        dates = factor_returns.index.tolist()
        
        # Start rolling from period 24
        for t in range(23, len(dates)):
            hist_data = factor_returns.iloc[:t+1]
            
            if len(hist_data) < 12:
                continue
            
            # Calculate expanding mean and covariance through period t
            mu_t = hist_data.mean().values
            Sigma_t = hist_data.cov().values
            
            # Handle singular covariance
            try:
                Sigma_inv_t = np.linalg.inv(Sigma_t)
            except np.linalg.LinAlgError:
                Sigma_inv_t = np.linalg.pinv(Sigma_t)
            
            # Calculate optimal tangency weights
            if np.all(np.isfinite(mu_t)) and np.all(np.isfinite(Sigma_inv_t)):
                w_raw = Sigma_inv_t @ mu_t
                ex_ante_variance = mu_t.T @ Sigma_inv_t @ mu_t
                
                portfolio_ret_t_plus_1 = 0.0
                
                if ex_ante_variance > 0:
                    # Target 1% monthly volatility
                    target_vol = 0.01
                    ex_ante_vol = np.sqrt(ex_ante_variance)
                    
                    if ex_ante_vol > 1e-8:
                        scaling_factor = target_vol / ex_ante_vol
                        w_scaled = scaling_factor * w_raw
                        
                        # Apply to t+1 returns
                        if t + 1 < len(dates):
                            F_t_plus_1 = factor_returns.iloc[t+1].values
                            if np.all(np.isfinite(F_t_plus_1)):
                                portfolio_ret_t_plus_1 = w_scaled.T @ F_t_plus_1
                
                if t+1 < len(dates):
                    portfolio_returns.append(portfolio_ret_t_plus_1)
        
        if len(portfolio_returns) < 12:
            return np.nan
        
        # Calculate Sharpe ratio
        portfolio_returns = np.array(portfolio_returns)
        mean_ret = np.mean(portfolio_returns)
        std_ret = np.std(portfolio_returns)
        
        if std_ret == 0:
            return np.nan
        
        annual_mean = mean_ret * 12
        annual_std = std_ret * np.sqrt(12)
        
        return annual_mean / annual_std
        
    except Exception as e:
        logger.warning(f"Error calculating tangency Sharpe: {e}")
        return np.nan

#-------------------------------------------------------------------------------
# OUT-OF-SAMPLE FACTOR CALCULATION

def compute_oos_factor(
    Gamma_df: pd.DataFrame,
    Z_df: pd.DataFrame,
    r_series: pd.Series,
    ridge: float = 1e-8
) -> Optional[np.ndarray]:
    """
    Compute out-of-sample factor using Equation (6) from the paper.
    
    Formula: f_{t+1,t} = (Γ'Z'ZΓ + λI)^{-1} Γ'Z'r_{t+1}
    """
    # Find common observations
    common_idx = Z_df.index.intersection(r_series.index)
    if len(common_idx) < Gamma_df.shape[1]:
        return None
    
    # Align characteristics
    common_chars = match_features(
        source_features=Gamma_df.index.tolist(),
        target_features=Z_df.columns.tolist(),
        method='flexible'
    )
    
    if len(common_chars) == 0:
        return None
    
    # Extract matched data
    Z = Z_df.loc[common_idx, common_chars].values
    r = r_series.loc[common_idx].values.reshape(-1, 1)
    Gamma = Gamma_df.loc[common_chars].values
    
    # Check for invalid values
    if (np.any(np.isnan(Z)) or np.any(np.isnan(r)) or np.any(np.isnan(Gamma)) or
        np.std(r) < 1e-6 or np.any(np.isinf(Z)) or np.any(np.isinf(r))):
        return None
    
    # Transform: ZΓ
    ZG = Z @ Gamma
    
    # Check rank
    K = Gamma.shape[1]
    if np.linalg.matrix_rank(ZG) < K:
        return None
    
    # Solve with tiny ridge regularization for numerical stability
    numerator = ZG.T @ r.flatten()
    denominator = ZG.T @ ZG + ridge * np.eye(K)
    
    # Check condition number
    if np.linalg.cond(denominator) > 1e10:
        return None
    
    try:
        f_oos = np.linalg.solve(denominator, numerator)
    except np.linalg.LinAlgError:
        try:
            f_oos = np.linalg.lstsq(denominator, numerator, rcond=None)[0]
        except:
            return None
    
    # Sanity check
    if np.any(np.isnan(f_oos)) or np.any(np.isinf(f_oos)) or np.any(np.abs(f_oos) > 10):
        return None
    
    return f_oos.flatten()

#-------------------------------------------------------------------------------
# R-SQUARED CALCULATION ON SUBSETS

def calculate_r_squared_on_subset(
    df_subset: pd.DataFrame,
    char_cols: List[str],
    Gamma_df: pd.DataFrame,
    Factors_df: pd.DataFrame,
    char_mapping: Optional[Dict[str, str]] = None
) -> Tuple[float, float]:
    """
    Calculate Total R² and Predictive R² on a subset using unified parameters.
    
    This implements the "unified parameter, separate evaluation" logic for Table VI.
    """
    try:
        # Build panel
        X_subset, y_subset = build_panel_matrices(df_subset, char_cols, 100)
        
        if len(X_subset) < 100:
            logger.warning("Insufficient observations in subset")
            return np.nan, np.nan
        
        # Match features
        if char_mapping:
            # Cross-validation mode: map features between Large/Small
            gamma_features = Gamma_df.index.tolist()
            eval_features_mapped = []
            train_features_filtered = []
            
            for train_feat in gamma_features:
                eval_feat = char_mapping.get(train_feat, train_feat)
                if eval_feat in X_subset.columns:
                    eval_features_mapped.append(eval_feat)
                    train_features_filtered.append(train_feat)
            
            if len(eval_features_mapped) == 0:
                logger.warning("No features matched with mapping")
                return np.nan, np.nan
            
            Z_matched = X_subset[eval_features_mapped].values
            Gamma_matched = Gamma_df.loc[train_features_filtered].values
            
        else:
            # Same-sample mode: flexible matching
            matched_chars = match_features(
                source_features=Gamma_df.index.tolist(),
                target_features=X_subset.columns.tolist(),
                method='flexible'
            )
            
            if len(matched_chars) == 0:
                logger.warning("No matching characteristics")
                return np.nan, np.nan
            
            Z_matched = X_subset[matched_chars].values
            Gamma_matched = Gamma_df.loc[matched_chars].values
        
        # Verify dimensions
        if Z_matched.shape[1] != Gamma_matched.shape[0]:
            logger.error(f"Dimension mismatch: Z={Z_matched.shape}, Gamma={Gamma_matched.shape}")
            return np.nan, np.nan
        
        # Calculate factor loadings
        factor_loadings = Z_matched @ Gamma_matched  # [N*T, K]
        
        # Align dates
        eval_dates = sorted(df_subset['date'].unique())
        eval_periods = pd.DatetimeIndex(eval_dates).to_period('M')
        factor_periods = Factors_df.index.to_period('M')
        
        # Find common periods
        common_periods = eval_periods.intersection(factor_periods)
        
        if len(common_periods) == 0:
            logger.error("No date overlap!")
            return np.nan, np.nan
        
        # TOTAL R²: Use actual factor values f_t
        y_pred_total = []
        y_actual = []
        
        for period in common_periods:
            # Find corresponding date in eval data
            eval_date_mask = eval_periods == period
            if not eval_date_mask.any():
                continue
            eval_date = eval_dates[eval_date_mask.argmax()]
            
            # Find corresponding factor index
            factor_idx_mask = factor_periods == period
            if not factor_idx_mask.any():
                continue
            factor_idx = factor_idx_mask.argmax()
            
            F_t = Factors_df.iloc[factor_idx].values  # [K,]
            
            # Get observations for this date
            date_obs_mask = X_subset.index.get_level_values('date') == eval_date
            if date_obs_mask.sum() == 0:
                continue
            
            date_idx_positions = np.where(date_obs_mask)[0]
            ZG_t = factor_loadings[date_idx_positions, :]  # [N_t, K]
            
            # Predict: ŷ = (Z × Γ) × f_t
            y_pred_t = ZG_t @ F_t
            y_actual_t = y_subset.iloc[date_idx_positions].values
            
            y_pred_total.extend(y_pred_t)
            y_actual.extend(y_actual_t)
        
        if len(y_pred_total) == 0:
            return np.nan, np.nan
        
        y_pred_total = np.array(y_pred_total)
        y_actual = np.array(y_actual)
        
        # Calculate Total R²
        valid_mask = np.isfinite(y_pred_total) & np.isfinite(y_actual)
        if valid_mask.sum() < 10:
            return np.nan, np.nan
        
        ss_res = np.sum((y_actual[valid_mask] - y_pred_total[valid_mask])**2)
        ss_tot = np.sum(y_actual[valid_mask]**2)
        total_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        # PREDICTIVE R²: Use factor mean λ̄
        lambda_mean = Factors_df.iloc[factor_periods.isin(common_periods)].mean().values
        
        y_pred_predictive = []
        for period in common_periods:
            eval_date_mask = eval_periods == period
            if not eval_date_mask.any():
                continue
            eval_date = eval_dates[eval_date_mask.argmax()]
            
            date_obs_mask = X_subset.index.get_level_values('date') == eval_date
            if date_obs_mask.sum() == 0:
                continue
            
            date_idx_positions = np.where(date_obs_mask)[0]
            ZG_t = factor_loadings[date_idx_positions, :]
            
            y_pred_t = ZG_t @ lambda_mean
            y_pred_predictive.extend(y_pred_t)
        
        if len(y_pred_predictive) != len(y_actual):
            return total_r2, np.nan
        
        y_pred_predictive = np.array(y_pred_predictive)
        
        valid_mask_pred = np.isfinite(y_pred_predictive) & np.isfinite(y_actual)
        if valid_mask_pred.sum() < 10:
            return total_r2, np.nan
        
        ss_res_pred = np.sum((y_actual[valid_mask_pred] - y_pred_predictive[valid_mask_pred])**2)
        ss_tot_pred = np.sum(y_actual[valid_mask_pred]**2)
        pred_r2 = 1 - ss_res_pred / ss_tot_pred if ss_tot_pred > 0 else np.nan
        
        return float(total_r2), float(pred_r2)
        
    except Exception as e:
        logger.error(f"R² calculation error: {e}", exc_info=True)
        return np.nan, np.nan

#-------------------------------------------------------------------------------
# TABLE V: OUT-OF-SAMPLE SHARPE RATIOS

def replicate_table_v_oos_single_k(
    df: pd.DataFrame,
    char_cols: List[str],
    K: int = 4,
    min_train_periods: int = 120
) -> Dict:
    """
    Replicate Table V: Out-of-sample Sharpe ratios using T/2 split methodology.
    Uses OOS instruments (expanding mean).
    """
    df_sorted = df.sort_values('date')
    unique_dates = sorted(df_sorted['date'].unique())
    
    # Calculate T/2 split point
    total_periods = len(unique_dates)
    t_half_idx = total_periods // 2
    
    # Find date closest to target OOS start
    oos_start_target = pd.to_datetime(CONFIG['oos_start_date'])
    oos_start_idx = None
    for i, date in enumerate(unique_dates):
        if date >= oos_start_target:
            oos_start_idx = i
            break
    
    # Use T/2 or closest to target
    if oos_start_idx is None:
        oos_start_idx = t_half_idx
    else:
        if abs(oos_start_idx - t_half_idx) > abs(t_half_idx - oos_start_idx):
            oos_start_idx = t_half_idx
    
    if oos_start_idx >= len(unique_dates) - 1:
        raise ValueError("Insufficient data for T/2 split")
    
    logger.info(
        f"OOS period: {unique_dates[oos_start_idx].strftime('%Y-%m')} to "
        f"{unique_dates[-1].strftime('%Y-%m')} ({len(unique_dates) - oos_start_idx} months)"
    )
    
    # Storage for OOS results
    oos_factors = []
    oos_dates = []
    
    # Rolling estimation loop
    for t in range(oos_start_idx, len(unique_dates) - 1):
        current_date = unique_dates[t]
        next_date = unique_dates[t + 1]
        
        # Training data (expanding window) with OOS instruments
        train_df = df[df['date'] <= current_date].copy()
        if len(train_df) < CONFIG['min_train_obs']:
            continue
        
        # Build OOS instruments (expanding mean)
        train_df_instr, instr_cols = build_instruments_oosample(train_df, char_cols)
        if len(instr_cols) == 0:
            continue
        
        # Estimate IPCA model
        try:
            _, _, Gamma_df_t, _ = run_ipca_with_proper_instruments(
                train_df_instr, instr_cols, K, CONFIG['keep_dates_with_at_least']
            )
        except Exception as e:
            logger.debug(f"IPCA training failed at t={t}: {e}")
            continue
        
        # Prepare test data
        test_current = df[df['date'] == current_date].copy()
        test_next = df[df['date'] == next_date].copy()
        
        if test_current.empty or test_next.empty:
            continue
        
        # Get common stocks (SORTED for alignment)
        common_stocks = sorted(
            set(test_current['permno']).intersection(set(test_next['permno']))
        )
        if len(common_stocks) < 10:
            continue
        
        # Build test instruments using same OOS logic
        test_current_with_hist = df[df['date'] <= current_date].copy()
        test_current_instr, test_instr_cols = build_instruments_oosample(
            test_current_with_hist, char_cols
        )
        test_current_t = test_current_instr[test_current_instr['date'] == current_date]
        
        if test_current_t.empty:
            continue
        
        # Force alignment by permno
        test_current_t_aligned = (
            test_current_t[test_current_t['permno'].isin(common_stocks)]
            .set_index('permno')
            .loc[common_stocks]
        )
        test_next_aligned = (
            test_next[test_next['permno'].isin(common_stocks)]
            .set_index('permno')
            .loc[common_stocks]
        )
        
        # Verify alignment
        if not test_current_t_aligned.index.equals(test_next_aligned.index):
            continue
        
        # Match instrument columns
        matched_instr_cols = [
            col for col in test_instr_cols
            if col in instr_cols and col in test_current_t_aligned.columns
        ]
        if len(matched_instr_cols) == 0:
            continue
        
        Z_t = test_current_t_aligned[matched_instr_cols]
        r_t_plus_1 = test_next_aligned['ret_lead']
        
        # Compute OOS factor
        Gamma_df_matched = Gamma_df_t.reindex(matched_instr_cols).dropna()
        if len(Gamma_df_matched) == 0:
            continue
        
        f_oos = compute_oos_factor(Gamma_df_matched, Z_t, r_t_plus_1)
        
        if f_oos is not None and not np.any(np.isnan(f_oos)):
            oos_factors.append(f_oos)
            oos_dates.append(next_date)
    
    if len(oos_factors) == 0:
        return {'status': 'error', 'message': 'No valid OOS factors computed'}
    
    # Convert to DataFrame
    oos_factors_df = pd.DataFrame(
        oos_factors,
        index=oos_dates,
        columns=[f'Factor_{i+1}' for i in range(K)]
    )
    
    logger.info(f"K={K}: Computed {len(oos_factors)} OOS factor observations")
    
    # Calculate performance metrics
    results = {
        'n_periods': len(oos_factors),
        'date_range': f"{oos_dates[0].strftime('%Y-%m')} to {oos_dates[-1].strftime('%Y-%m')}",
        'factor_sharpes': {
            f'Factor_{i+1}': get_sharpe(oos_factors_df[f'Factor_{i+1}'])
            for i in range(K)
        },
        'tangency_sharpe': get_tangency_sharpe_rolling(oos_factors_df),
        'oos_factors_df': oos_factors_df,
        'status': 'success'
    }
    
    return results


def replicate_table_v_oos(
    df: pd.DataFrame,
    char_cols: List[str],
    Kmax: int = 6
) -> pd.DataFrame:
    """Replicate Table V: Out-of-sample Sharpe ratios for K=1 to Kmax."""
    logger.info(f"\n{'='*80}\nTABLE V: OUT-OF-SAMPLE SHARPE RATIOS (IPCA)\n{'='*80}")
    
    univariate_sharpes = []
    tangency_sharpes = []
    
    for K in range(1, Kmax + 1):
        logger.info(f"\nProcessing K={K}...")
        result_k = replicate_table_v_oos_single_k(df, char_cols, K=K)
        
        if result_k['status'] == 'success':
            factor_sharpes = list(result_k['factor_sharpes'].values())
            
            # Kth factor (last one added)
            if len(factor_sharpes) != K:
                univariate_sharpe = np.nan
            elif np.isnan(factor_sharpes[K-1]):
                univariate_sharpe = np.nan
            else:
                univariate_sharpe = factor_sharpes[K-1]
            
            tangency_sharpe = result_k['tangency_sharpe']
            
            logger.info(
                f"  K={K}: Univariate={univariate_sharpe:.2f}, "
                f"Tangency={tangency_sharpe:.2f}"
            )
        else:
            univariate_sharpe = np.nan
            tangency_sharpe = np.nan
            logger.warning(f"  K={K}: Failed")
        
        univariate_sharpes.append(univariate_sharpe)
        tangency_sharpes.append(tangency_sharpe)
    
    results = pd.DataFrame({
        'K': list(range(1, Kmax + 1)),
        'Univariate': univariate_sharpes,
        'Tangency': tangency_sharpes
    })
    
    return results


def replicate_table_v_panel_b_observable_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate Table V Panel B: Observable Factors (Benchmark Models)."""
    logger.info("\n--- Panel B: Observable Factors ---")
    
    ff_factors = load_ff_factors()
    if ff_factors.empty:
        logger.warning("FF factors not loaded")
        return pd.DataFrame()
    
    # Define factor models
    factor_models = {
        1: {'name': 'CAPM', 'factors': ['Mkt-RF'], 'kth_factor': 'Mkt-RF'},
        2: {'name': 'FF2', 'factors': ['Mkt-RF', 'SMB'], 'kth_factor': 'SMB'},
        3: {'name': 'FF3', 'factors': ['Mkt-RF', 'SMB', 'HML'], 'kth_factor': 'HML'},
        4: {'name': 'FFC4', 'factors': ['Mkt-RF', 'SMB', 'HML', 'MOM'], 'kth_factor': 'MOM'},
        5: {'name': 'FF5', 'factors': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], 'kth_factor': 'CMA'},
        6: {'name': 'FFC6', 'factors': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'], 'kth_factor': 'MOM'},
    }
    
    # T/2 split
    total_periods = len(ff_factors)
    t_half_idx = total_periods // 2
    
    results_dict = {}
    
    for K in [1, 2, 3, 4, 5, 6]:
        model = factor_models[K]
        
        try:
            model_factors = ff_factors[model['factors']].copy()
        except KeyError as e:
            logger.warning(f"K={K}: Missing factors {e}")
            results_dict[K] = {'univariate': np.nan, 'tangency': np.nan}
            continue
        
        # OOS period
        oos_factors = model_factors.iloc[t_half_idx:].copy()
        
        if oos_factors.empty or len(oos_factors) < 24:
            results_dict[K] = {'univariate': np.nan, 'tangency': np.nan}
            continue
        
        # Univariate Sharpe
        kth_factor_name = model['kth_factor']
        kth_factor_returns = oos_factors[kth_factor_name]
        univariate_sharpe = get_sharpe(kth_factor_returns)
        
        # Tangency Sharpe
        tangency_sharpe = get_tangency_sharpe_rolling(oos_factors)
        
        results_dict[K] = {
            'univariate': univariate_sharpe,
            'tangency': tangency_sharpe
        }
        
        logger.info(
            f"  {model['name']}: Univariate={univariate_sharpe:.2f}, "
            f"Tangency={tangency_sharpe:.2f}"
        )
    
    results = pd.DataFrame({
        'K': [1, 2, 3, 4, 5, 6],
        'Model': [factor_models[k]['name'] for k in [1, 2, 3, 4, 5, 6]],
        'Univariate': [results_dict[k]['univariate'] for k in [1, 2, 3, 4, 5, 6]],
        'Tangency': [results_dict[k]['tangency'] for k in [1, 2, 3, 4, 5, 6]]
    })
    
    return results


def replicate_table_v_complete(
    df: pd.DataFrame,
    char_cols: List[str],
    Kmax: int = 6
) -> Dict[str, pd.DataFrame]:
    """Complete Table V replication."""
    results = {}
    
    # Panel A: IPCA
    logger.info("\n=== PANEL A: IPCA ===")
    panel_a = replicate_table_v_oos(df, char_cols, Kmax=Kmax)
    results['panel_a'] = panel_a
    
    # Print Panel A
    print(f"\n{'='*80}")
    print("TABLE V - PANEL A: IPCA OUT-OF-SAMPLE SHARPE RATIOS")
    print('='*80)
    print(panel_a.to_string(index=False))
    
    # Panel B: Observable Factors
    logger.info("\n=== PANEL B: OBSERVABLE FACTORS ===")
    panel_b = replicate_table_v_panel_b_observable_factors(df)
    results['panel_b'] = panel_b
    
    if not panel_b.empty:
        print(f"\n{'='*80}")
        print("TABLE V - PANEL B: OBSERVABLE FACTORS OUT-OF-SAMPLE SHARPE RATIOS")
        print('='*80)
        print(panel_b.to_string(index=False))
    
    return results

#-------------------------------------------------------------------------------
# TABLE VI: SIZE SPLIT ANALYSIS (FIXED VERSION)

def replicate_table_vi_size_split(
    df: pd.DataFrame,
    char_cols: List[str],
    Kmax: int = 6
) -> pd.DataFrame:
    """
    Replicate Table VI: R² by market cap (Large vs Small stocks).
    
    FIXED: Now uses proper IS instruments (full-sample mean + deviation).
    """
    logger.info(f"\n{'='*80}\nTABLE VI: R² BY SIZE\n{'='*80}")
    
    mcap_col = 'lme' if 'lme' in df.columns else ('me' if 'me' in df.columns else 'size')
    if mcap_col not in df.columns:
        raise ValueError(f"Market cap column not found")
    
    results = []
    
    for K in range(1, Kmax + 1):
        logger.info(f"\n{'='*10} K={K} {'='*10}")
        
        # IN-SAMPLE: Build IS instruments (full-sample mean)
        logger.info("Building IN-SAMPLE instruments...")
        df_work_is, inst_cols_is = build_instruments_insample(df, char_cols)
        
        # Train unified model
        try:
            total_r2_full, pred_r2_full, Gamma_df_is, Factors_df_is = run_ipca_with_proper_instruments(
                df_work_is, inst_cols_is, K, CONFIG['keep_dates_with_at_least']
            )
            logger.info(f"  Unified model: Total R²={total_r2_full*100:.2f}%, Pred R²={pred_r2_full*100:.2f}%")
        except Exception as e:
            logger.warning(f"K={K}: Unified IPCA error: {e}")
            continue
        
        # Split by market cap
        df_work_is['mcap_rank'] = df_work_is.groupby('date')[mcap_col].transform(
            lambda x: x.rank(ascending=False, method='first', na_option='bottom')
        )
        df_large_is = df_work_is[df_work_is['mcap_rank'] <= 1000].drop(columns=['mcap_rank'])
        df_small_is = df_work_is[df_work_is['mcap_rank'] > 1000].drop(columns=['mcap_rank'])
        
        logger.info(f"  Subsamples: {len(df_large_is):,} large, {len(df_small_is):,} small obs")
        
        # IN-SAMPLE R² on subsets
        logger.info("  Evaluating IS Large...")
        total_r2_large_is, pred_r2_large_is = calculate_r_squared_on_subset(
            df_large_is, inst_cols_is, Gamma_df_is, Factors_df_is
        )
        
        logger.info("  Evaluating IS Small...")
        total_r2_small_is, pred_r2_small_is = calculate_r_squared_on_subset(
            df_small_is, inst_cols_is, Gamma_df_is, Factors_df_is
        )
        
        logger.info(f"  IS Large: Total={total_r2_large_is*100:.2f}%, Pred={pred_r2_large_is*100:.2f}%")
        logger.info(f"  IS Small: Total={total_r2_small_is*100:.2f}%, Pred={pred_r2_small_is*100:.2f}%")
        
        # Store results
        results.append({
            'K': K,
            'Total_R2_Full': total_r2_full * 100,
            'Pred_R2_Full': pred_r2_full * 100,
            'IS_Total_R2_Large': total_r2_large_is * 100 if not np.isnan(total_r2_large_is) else np.nan,
            'IS_Pred_R2_Large': pred_r2_large_is * 100 if not np.isnan(pred_r2_large_is) else np.nan,
            'IS_Total_R2_Small': total_r2_small_is * 100 if not np.isnan(total_r2_small_is) else np.nan,
            'IS_Pred_R2_Small': pred_r2_small_is * 100 if not np.isnan(pred_r2_small_is) else np.nan,
        })
    
    results_df = pd.DataFrame(results)
    
    # Print results
    if not results_df.empty:
        print(f"\n{'='*80}")
        print("TABLE VI: R² BY SIZE (IN-SAMPLE)")
        print('='*80)
        print("\nPanel A: Large Stocks")
        print(results_df[['K', 'IS_Total_R2_Large', 'IS_Pred_R2_Large']].to_string(index=False))
        print("\nPanel B: Small Stocks")
        print(results_df[['K', 'IS_Total_R2_Small', 'IS_Pred_R2_Small']].to_string(index=False))
        print('='*80)
    
    return results_df

def replicate_table_vi_oos_size_split(
    df: pd.DataFrame,
    char_cols: List[str],
    K: int = 4,
    oos_start_date: str = '1987-01-01',
    min_train_months: int = 120,
    mcap_cutoff: int = 1000
) -> Dict[str, float]:
    """
    Replicate Table VI OOS: Out-of-Sample R² by market cap (Large vs Small).
    
    Implements expanding window methodology:
    - For each month t in OOS period:
        1. Train IPCA on data [start, t] using OOS instruments (expanding mean)
        2. Predict returns for t+1
        3. Split predictions by market cap
        4. Accumulate predictions and actuals
    - Calculate panel R² across all OOS periods
    
    Parameters
    ----------
    df : pd.DataFrame
        Full sample data with characteristics and returns
    char_cols : List[str]
        List of characteristic names
    K : int
        Number of factors
    oos_start_date : str
        Start date for OOS period
    min_train_months : int
        Minimum number of months for initial training
    mcap_cutoff : int
        Market cap rank cutoff (stocks ranked 1-1000 = Large, >1000 = Small)
    
    Returns
    -------
    Dict[str, float]
        Dictionary with Total and Predictive R² for Large and Small stocks
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TABLE VI OOS: K={K} factors")
    logger.info(f"OOS Start: {oos_start_date}, Min train: {min_train_months} months")
    logger.info('='*80)
    
    # Identify market cap column
    mcap_col = 'lme' if 'lme' in df.columns else ('me' if 'me' in df.columns else 'size')
    if mcap_col not in df.columns:
        raise ValueError("Market cap column not found")
    
    # Sort by date
    df = df.sort_values(['permno', 'date']).copy()
    unique_dates = sorted(df['date'].unique())
    oos_start = pd.to_datetime(oos_start_date)
    
    # Find OOS start index
    oos_dates = [d for d in unique_dates if d >= oos_start]
    if len(oos_dates) < 2:
        raise ValueError("Insufficient OOS periods")
    
    # Ensure minimum training period
    train_dates = [d for d in unique_dates if d < oos_start]
    if len(train_dates) < min_train_months:
        raise ValueError(f"Insufficient training periods: {len(train_dates)} < {min_train_months}")
    
    logger.info(f"OOS periods: {len(oos_dates)} months ({oos_dates[0]} to {oos_dates[-1]})")
    logger.info(f"Initial training: {len(train_dates)} months")
    
    # Storage for predictions and actuals (by group)
    predictions_total_large = []  # Using realized factors
    predictions_total_small = []
    predictions_pred_large = []   # Using mean factors
    predictions_pred_small = []
    actuals_large = []
    actuals_small = []

    # MAIN EXPANDING WINDOW LOOP
    
    for t_idx, t_date in enumerate(oos_dates[:-1]):  # Predict up to T-1
        t_plus_1_date = oos_dates[t_idx + 1]
        
        logger.info(f"\n--- Period {t_idx+1}/{len(oos_dates)-1}: Training on [start, {t_date.strftime('%Y-%m')}], "
                   f"Predicting {t_plus_1_date.strftime('%Y-%m')} ---")
        # STEP 3a: TRAINING on [start, t]
        
        # Extract training data (all data up to and including t)
        df_train = df[df['date'] <= t_date].copy()
        
        if len(df_train) < min_train_months * 100:  # Heuristic check
            logger.warning(f"  Skipping: insufficient training data ({len(df_train)} obs)")
            continue
        
        # Build OOS instruments (expanding mean)
        try:
            df_train_instr, inst_cols = build_instruments_oosample(df_train, char_cols)
        except Exception as e:
            logger.warning(f"  Instrument construction failed: {e}")
            continue
        
        if df_train_instr.empty:
            logger.warning("  Skipping: empty instrument data")
            continue
        
        # Train IPCA model
        try:
            _, _, Gamma_t, Factors_t = run_ipca_with_proper_instruments(
                df_train_instr, inst_cols, K, keep_dates_with_at_least=100
            )
        except Exception as e:
            logger.warning(f"  IPCA training failed: {e}")
            continue
        
        # Calculate mean factor (lambda_bar) for Predictive R²
        lambda_bar_t = Factors_t.mean().values  # [K,]
        
        logger.info(f"  Trained: Gamma={Gamma_t.shape}, Factors={Factors_t.shape}")

        # STEP 3b: PREDICTION for t+1
        
        # Extract t and t+1 data
        df_t = df[df['date'] == t_date].copy()
        df_t_plus_1 = df[df['date'] == t_plus_1_date].copy()
        
        if df_t.empty or df_t_plus_1.empty:
            logger.warning("  Skipping: missing t or t+1 data")
            continue
        
        # Align stocks (only predict for stocks with characteristics at t and returns at t+1)
        common_stocks = set(df_t['permno']).intersection(set(df_t_plus_1['permno']))
        if len(common_stocks) < 50:
            logger.warning(f"  Skipping: only {len(common_stocks)} common stocks")
            continue
        
        df_t = df_t[df_t['permno'].isin(common_stocks)].copy()
        df_t_plus_1 = df_t_plus_1[df_t_plus_1['permno'].isin(common_stocks)].copy()
        
        # Build Z_t (instruments at time t from OOS-constructed instruments)
        # We need to extract Z_t from the trained data
        df_t_instr = df_train_instr[df_train_instr['date'] == t_date].copy()
        
        if df_t_instr.empty:
            logger.warning("  Skipping: no instruments for time t")
            continue
        
        # Match stocks
        df_t_instr = df_t_instr[df_t_instr['permno'].isin(common_stocks)]
        df_t_plus_1 = df_t_plus_1[df_t_plus_1['permno'].isin(df_t_instr['permno'])]
        
        # Align by permno
        df_t_instr = df_t_instr.sort_values('permno').set_index('permno')
        df_t_plus_1 = df_t_plus_1.sort_values('permno').set_index('permno')
        
        # Verify alignment
        if not df_t_instr.index.equals(df_t_plus_1.index):
            logger.warning("  Skipping: permno alignment failed")
            continue
        
        # Extract Z_t (instrument matrix)
        Z_t = df_t_instr[inst_cols]
        
        # Extract r_{t+1} (actual returns)
        r_t_plus_1 = df_t_plus_1['ret_lead']
        
        # Get market cap at time t for grouping
        mcap_t = df_t.set_index('permno').loc[Z_t.index, mcap_col]
        
        # Rank market cap
        mcap_rank = mcap_t.rank(ascending=False, method='first')
        
        # Split into Large (rank <= 1000) and Small (rank > 1000)
        large_mask = mcap_rank <= mcap_cutoff
        small_mask = mcap_rank > mcap_cutoff
        
        n_large = large_mask.sum()
        n_small = small_mask.sum()
        
        if n_large < 10 or n_small < 10:
            logger.warning(f"  Skipping: insufficient stocks (Large={n_large}, Small={n_small})")
            continue
        
        Z_t_large = Z_t[large_mask]
        Z_t_small = Z_t[small_mask]
        r_t_plus_1_large = r_t_plus_1[large_mask]
        r_t_plus_1_small = r_t_plus_1[small_mask]

        # Compute REALIZED factor f_{t+1,t}
        # Using Equation (6): f_{t+1,t} = (Γ'Z'ZΓ + λI)^{-1} Γ'Z'r_{t+1}
        
        # Compute for FULL sample (needed for factor extraction)
        f_t_plus_1_realized = compute_oos_factor(Gamma_t, Z_t, r_t_plus_1)
        
        if f_t_plus_1_realized is None or len(f_t_plus_1_realized) != K:
            logger.warning("  Skipping: OOS factor computation failed")
            continue
        
        logger.info(f"  f_{{t+1,t}} computed: {f_t_plus_1_realized}")
        # Compute predictions for LARGE stock
        
        # Match features between Gamma and Z
        common_features_large = [f for f in Gamma_t.index if f in Z_t_large.columns]
        if len(common_features_large) < Gamma_t.shape[0] * 0.5:
            logger.warning(f"  Large: Only {len(common_features_large)}/{Gamma_t.shape[0]} features matched")
        
        Z_large_matched = Z_t_large[common_features_large].values
        Gamma_matched = Gamma_t.loc[common_features_large].values
        
        # Factor loadings: β_i = Z_i × Γ
        beta_large = Z_large_matched @ Gamma_matched  # [N_large, K]
        
        # Total R² prediction: ŷ = β × f_{t+1,t}
        y_pred_total_large = beta_large @ f_t_plus_1_realized
        
        # Predictive R² prediction: ŷ = β × λ̄
        y_pred_pred_large = beta_large @ lambda_bar_t

        # Compute predictions for SMALL stocks
        
        common_features_small = [f for f in Gamma_t.index if f in Z_t_small.columns]
        Z_small_matched = Z_t_small[common_features_small].values
        Gamma_matched_small = Gamma_t.loc[common_features_small].values
        
        beta_small = Z_small_matched @ Gamma_matched_small  # [N_small, K]
        
        y_pred_total_small = beta_small @ f_t_plus_1_realized
        y_pred_pred_small = beta_small @ lambda_bar_t
        # Store results
        predictions_total_large.extend(y_pred_total_large)
        predictions_total_small.extend(y_pred_total_small)
        predictions_pred_large.extend(y_pred_pred_large)
        predictions_pred_small.extend(y_pred_pred_small)
        
        actuals_large.extend(r_t_plus_1_large.values)
        actuals_small.extend(r_t_plus_1_small.values)
        
        logger.info(f"  Stored: {len(y_pred_total_large)} large, {len(y_pred_total_small)} small predictions")
    # CALCULATE PANEL R²
    
    logger.info(f"\n{'='*80}")
    logger.info("CALCULATING OOS R²")
    logger.info('='*80)
    
    # Convert to arrays
    pred_total_large = np.array(predictions_total_large)
    pred_total_small = np.array(predictions_total_small)
    pred_pred_large = np.array(predictions_pred_large)
    pred_pred_small = np.array(predictions_pred_small)
    act_large = np.array(actuals_large)
    act_small = np.array(actuals_small)
    
    logger.info(f"Total predictions collected:")
    logger.info(f"  Large: {len(act_large):,} observations")
    logger.info(f"  Small: {len(act_small):,} observations")
    
    # Calculate R² for Large stocks
    def calc_r2(y_pred, y_actual):
        """Calculate panel R²"""
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_actual)
        if valid_mask.sum() < 10:
            return np.nan
        
        y_p = y_pred[valid_mask]
        y_a = y_actual[valid_mask]
        
        ss_res = np.sum((y_a - y_p) ** 2)
        ss_tot = np.sum(y_a ** 2)
        
        if ss_tot <= 0:
            return np.nan
        
        return 1 - ss_res / ss_tot
    
    # Large stocks
    total_r2_large = calc_r2(pred_total_large, act_large)
    pred_r2_large = calc_r2(pred_pred_large, act_large)
    
    # Small stocks
    total_r2_small = calc_r2(pred_total_small, act_small)
    pred_r2_small = calc_r2(pred_pred_small, act_small)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TABLE VI OOS RESULTS (K={K})")
    print('='*80)
    print(f"\n{'Sample':<15} {'Total R²':<15} {'Predictive R²':<15}")
    print('-' * 45)
    print(f"{'Large':<15} {total_r2_large*100:>13.2f}% {pred_r2_large*100:>13.2f}%")
    print(f"{'Small':<15} {total_r2_small*100:>13.2f}% {pred_r2_small*100:>13.2f}%")
    print('='*80)
    
    return {
        'K': K,
        'OOS_Total_R2_Large': total_r2_large * 100,
        'OOS_Pred_R2_Large': pred_r2_large * 100,
        'OOS_Total_R2_Small': total_r2_small * 100,
        'OOS_Pred_R2_Small': pred_r2_small * 100,
        'N_Large': len(act_large),
        'N_Small': len(act_small)
    }


def replicate_table_vi_complete(
    df: pd.DataFrame,
    char_cols: List[str],
    Kmax: int = 6,
    oos_start_date: str = '1987-01-01'
) -> pd.DataFrame:
    """
    Replicate COMPLETE Table VI: Both In-Sample and Out-of-Sample R².
    
    This combines:
    1. IS logic (from existing replicate_table_vi_size_split)
    2. OOS logic (from new replicate_table_vi_oos_size_split)
    
    Parameters
    ----------
    df : pd.DataFrame
        Full sample data
    char_cols : List[str]
        Characteristic names
    Kmax : int
        Maximum number of factors
    oos_start_date : str
        OOS start date
    
    Returns
    -------
    pd.DataFrame
        Combined results with IS and OOS R² for K=1 to Kmax
    """
    logger.info(f"\n{'='*80}")
    logger.info("TABLE VI COMPLETE: IN-SAMPLE + OUT-OF-SAMPLE")
    logger.info('='*80)
    
    results = []
    
    for K in range(1, Kmax + 1):
        logger.info(f"\n{'='*10} K={K} {'='*10}")
        
        row = {'K': K}
        # PART 1: IN-SAMPLE
        logger.info("\n--- IN-SAMPLE ---")
        
        try:
            # Build IS instruments
            df_is, inst_cols_is = build_instruments_insample(df, char_cols)
            
            # Train unified model
            total_full, pred_full, Gamma_is, Factors_is = run_ipca_with_proper_instruments(
                df_is, inst_cols_is, K, keep_dates_with_at_least=1000
            )
            
            # Split by size
            mcap_col = 'lme' if 'lme' in df_is.columns else 'size'
            df_is['mcap_rank'] = df_is.groupby('date')[mcap_col].transform(
                lambda x: x.rank(ascending=False, method='first')
            )
            
            df_large_is = df_is[df_is['mcap_rank'] <= 1000]
            df_small_is = df_is[df_is['mcap_rank'] > 1000]
            
            # Calculate IS R²
            total_large_is, pred_large_is = calculate_r_squared_on_subset(
                df_large_is, inst_cols_is, Gamma_is, Factors_is
            )
            total_small_is, pred_small_is = calculate_r_squared_on_subset(
                df_small_is, inst_cols_is, Gamma_is, Factors_is
            )
            
            row['IS_Total_R2_Large'] = total_large_is * 100
            row['IS_Pred_R2_Large'] = pred_large_is * 100
            row['IS_Total_R2_Small'] = total_small_is * 100
            row['IS_Pred_R2_Small'] = pred_small_is * 100
            
            logger.info(f"IS Large: Total={total_large_is*100:.2f}%, Pred={pred_large_is*100:.2f}%")
            logger.info(f"IS Small: Total={total_small_is*100:.2f}%, Pred={pred_small_is*100:.2f}%")
            
        except Exception as e:
            logger.error(f"IS failed for K={K}: {e}")
            row['IS_Total_R2_Large'] = np.nan
            row['IS_Pred_R2_Large'] = np.nan
            row['IS_Total_R2_Small'] = np.nan
            row['IS_Pred_R2_Small'] = np.nan

        # PART 2: OUT-OF-SAMPLE (new expanding window logic)
        logger.info("\n--- OUT-OF-SAMPLE ---")
        try:
            oos_results = replicate_table_vi_oos_size_split(
                df, char_cols, K=K, oos_start_date=oos_start_date
            )
            
            row['OOS_Total_R2_Large'] = oos_results['OOS_Total_R2_Large']
            row['OOS_Pred_R2_Large'] = oos_results['OOS_Pred_R2_Large']
            row['OOS_Total_R2_Small'] = oos_results['OOS_Total_R2_Small']
            row['OOS_Pred_R2_Small'] = oos_results['OOS_Pred_R2_Small']
            
        except Exception as e:
            logger.error(f"OOS failed for K={K}: {e}")
            row['OOS_Total_R2_Large'] = np.nan
            row['OOS_Pred_R2_Large'] = np.nan
            row['OOS_Total_R2_Small'] = np.nan
            row['OOS_Pred_R2_Small'] = np.nan
        
        results.append(row)

    #final table
    results_df = pd.DataFrame(results)
    
    # Print formatted table
    print(f"\n{'='*100}")
    print("TABLE VI COMPLETE: IN-SAMPLE AND OUT-OF-SAMPLE R²")
    print('='*100)
    print("\nPanel A: LARGE STOCKS")
    print(f"{'K':<5} {'IS Total R²':<15} {'IS Pred R²':<15} {'OOS Total R²':<15} {'OOS Pred R²':<15}")
    print('-' * 70)
    
    for _, row in results_df.iterrows():
        print(f"{int(row['K']):<5} "
              f"{row['IS_Total_R2_Large']:>13.2f}% "
              f"{row['IS_Pred_R2_Large']:>13.2f}% "
              f"{row['OOS_Total_R2_Large']:>13.2f}% "
              f"{row['OOS_Pred_R2_Large']:>13.2f}%")
    
    print("\nPanel B: SMALL STOCKS")
    print(f"{'K':<5} {'IS Total R²':<15} {'IS Pred R²':<15} {'OOS Total R²':<15} {'OOS Pred R²':<15}")
    print('-' * 70)
    
    for _, row in results_df.iterrows():
        print(f"{int(row['K']):<5} "
              f"{row['IS_Total_R2_Small']:>13.2f}% "
              f"{row['IS_Pred_R2_Small']:>13.2f}% "
              f"{row['OOS_Total_R2_Small']:>13.2f}% "
              f"{row['OOS_Pred_R2_Small']:>13.2f}%")
    
    print('='*100)
    
    return results_df

#-------------------------------------------------------------------------------
# TABLE VII: CROSS-VALIDATION

def replicate_table_vii_cross_validation(
    df_raw: pd.DataFrame,
    char_cols: List[str],
    K: int = 4
) -> pd.DataFrame:
    """
    Replicate Table VII: Cross-validation between Large and Small models.
    
    FIXED: Now uses proper IS instruments (full-sample mean + deviation).
    """
    logger.info(f"\n{'='*80}\nTABLE VII: CROSS-VALIDATION ANALYSIS (K={K})\n{'='*80}")
    
    # Identify market cap column
    mcap_col = 'lme' if 'lme' in df_raw.columns else ('me' if 'me' in df_raw.columns else 'size')
    if mcap_col not in df_raw.columns:
        raise ValueError(f"Market cap column not found")
    
    # STEP 1: Split RAW data by size BEFORE any transformation
    df_raw = df_raw.copy()
    df_raw['mcap_rank'] = df_raw.groupby('date')[mcap_col].transform(
        lambda x: x.rank(ascending=False, method='first', na_option='bottom')
    )
    
    df_large_raw = df_raw[df_raw['mcap_rank'] <= 1000].copy().drop(columns=['mcap_rank'])
    df_small_raw = df_raw[df_raw['mcap_rank'] > 1000].copy().drop(columns=['mcap_rank'])
    
    logger.info(f"Split: {len(df_large_raw):,} large, {len(df_small_raw):,} small obs")
    
    # STEP 2: Clean separately
    df_large_clean, char_cols_large = clean_data(df_large_raw, apply_rank_transform=False)
    df_small_clean, char_cols_small = clean_data(df_small_raw, apply_rank_transform=False)
    
    logger.info(f"After cleaning: Large={len(df_large_clean):,}, Small={len(df_small_clean):,}")
    
    # STEP 3: Build IS instruments separately (full-sample mean within each subset)
    logger.info("\nBuilding Large IS instruments...")
    df_large_instr, inst_cols_large = build_instruments_insample(df_large_clean, char_cols_large)
    
    logger.info("Building Small IS instruments...")
    df_small_instr, inst_cols_small = build_instruments_insample(df_small_clean, char_cols_small)
    
    # Find common instrument base features for cross-validation
    def extract_base(char_name):
        return char_name.replace('_mean_rank', '').replace('_dev_rank', '').replace('_rank', '')
    
    large_bases = set([extract_base(c) for c in inst_cols_large if c != 'const'])
    small_bases = set([extract_base(c) for c in inst_cols_small if c != 'const'])
    common_bases = large_bases & small_bases
    
    logger.info(f"Common base characteristics: {len(common_bases)}")
    
    # Build feature mapping
    large_to_small_map = {}
    small_to_large_map = {}
    
    for base in common_bases:
        large_mean = f'{base}_mean_rank'
        large_dev = f'{base}_dev_rank'
        small_mean = f'{base}_mean_rank'
        small_dev = f'{base}_dev_rank'
        
        if large_mean in inst_cols_large and small_mean in inst_cols_small:
            large_to_small_map[large_mean] = small_mean
            small_to_large_map[small_mean] = large_mean
        
        if large_dev in inst_cols_large and small_dev in inst_cols_small:
            large_to_small_map[large_dev] = small_dev
            small_to_large_map[small_dev] = large_dev
    
    # Add const mapping
    large_to_small_map['const'] = 'const'
    small_to_large_map['const'] = 'const'
    
    logger.info(f"Feature mappings: {len(large_to_small_map)} features")
    
    # STEP 4: Train Large model
    logger.info("\nTraining Large model...")
    try:
        _, _, Gamma_large, Factors_large = run_ipca_with_proper_instruments(
            df_large_instr, inst_cols_large, K, CONFIG['keep_dates_with_at_least']
        )
        logger.info(f"  Large model: Gamma={Gamma_large.shape}, Factors={Factors_large.shape}")
    except Exception as e:
        logger.error(f"Large model failed: {e}")
        return pd.DataFrame()
    
    # STEP 5: Train Small model
    logger.info("\nTraining Small model...")
    try:
        _, _, Gamma_small, Factors_small = run_ipca_with_proper_instruments(
            df_small_instr, inst_cols_small, K, CONFIG['keep_dates_with_at_least']
        )
        logger.info(f"  Small model: Gamma={Gamma_small.shape}, Factors={Factors_small.shape}")
    except Exception as e:
        logger.error(f"Small model failed: {e}")
        return pd.DataFrame()
    
    # STEP 6: Cross-validation evaluations
    logger.info("\nCross-validation:")
    results = {}
    
    # L-on-L (Large parameters on Large data)
    logger.info("  Evaluating L-on-L...")
    total_L_L, pred_L_L = calculate_r_squared_on_subset(
        df_large_instr, inst_cols_large, Gamma_large, Factors_large
    )
    results['L_on_L'] = (total_L_L, pred_L_L)
    
    # S-on-S (Small parameters on Small data)
    logger.info("  Evaluating S-on-S...")
    total_S_S, pred_S_S = calculate_r_squared_on_subset(
        df_small_instr, inst_cols_small, Gamma_small, Factors_small
    )
    results['S_on_S'] = (total_S_S, pred_S_S)
    
    # L-on-S (Large parameters on Small data - Cross-validation)
    logger.info("  Evaluating L-on-S (cross-validation)...")
    total_L_S, pred_L_S = calculate_r_squared_on_subset(
        df_small_instr, inst_cols_small, Gamma_large, Factors_large,
        char_mapping=large_to_small_map
    )
    results['L_on_S'] = (total_L_S, pred_L_S)
    
    # S-on-L (Small parameters on Large data - Cross-validation)
    logger.info("  Evaluating S-on-L (cross-validation)...")
    total_S_L, pred_S_L = calculate_r_squared_on_subset(
        df_large_instr, inst_cols_large, Gamma_small, Factors_small,
        char_mapping=small_to_large_map
    )
    results['S_on_L'] = (total_S_L, pred_S_L)
    
    # Print results
    print(f"\n{'='*80}")
    print("TABLE VII: CROSS-VALIDATION RESULTS")
    print('='*80)
    print("\nEstimation Sample → Evaluation Sample")
    print(f"{'Sample':<12} {'Total R²':<25} {'Predictive R²':<25}")
    print(f"{'':12} {'Large':<12} {'Small':<12} {'Large':<12} {'Small':<12}")
    print("-" * 73)
    
    def fmt(val):
        return f"{val*100:>10.2f}" if not np.isnan(val) else f"{'FAIL':>10}"
    
    total_L_L, pred_L_L = results['L_on_L']
    total_L_S, pred_L_S = results['L_on_S']
    total_S_L, pred_S_L = results['S_on_L']
    total_S_S, pred_S_S = results['S_on_S']
    
    print(
        f"{'Large':<12} {fmt(total_L_L)} {fmt(total_L_S)} "
        f"{fmt(pred_L_L)} {fmt(pred_L_S)}"
    )
    print(
        f"{'Small':<12} {fmt(total_S_L)} {fmt(total_S_S)} "
        f"{fmt(pred_S_L)} {fmt(pred_S_S)}"
    )
    print('='*80)
    
    # Return DataFrame
    results_df = pd.DataFrame({
        'Estimation': ['Large', 'Large', 'Small', 'Small'],
        'Evaluation': ['Large', 'Small', 'Large', 'Small'],
        'Total_R2': [total_L_L*100, total_L_S*100, total_S_L*100, total_S_S*100],
        'Pred_R2': [pred_L_L*100, pred_L_S*100, pred_S_L*100, pred_S_S*100]
    })
    
    return results_df

#-------------------------------------------------------------------------------
# MAIN EXECUTION

def main():
    """Main execution function."""
    print("=" * 80)
    print("IPCA REPLICATION: TABLES V, VI, VII")
    print("=" * 80)
    
    # Check for required data file
    data_file = "characteristics_data_feb2017.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    try:
        # Step 1: Load and clean data
        logger.info("\n--- STEP 1: DATA LOADING & CLEANING ---")
        df_raw = load_data(data_file)
        df_clean, char_cols = clean_data(df_raw, apply_rank_transform=False)
        
        if df_clean.empty:
            logger.error("No data loaded")
            return
        
        # Configuration summary
        logger.info(f"\nConfiguration:")
        logger.info(f"  Kmax: {CONFIG['Kmax']}")
        logger.info(f"  Sample period: {CONFIG['start_date']} to {CONFIG['end_date']}")
        logger.info(f"  OOS start: {CONFIG['oos_start_date']}")
        logger.info(f"  Characteristics: {len(char_cols)}")
        
        # Create output directory
        OUTPUT_DIR = "./results_tableV_VI_VII"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Step 2: Replicate Table V
        logger.info("\n--- STEP 2: TABLE V (OUT-OF-SAMPLE SHARPE RATIOS) ---")
        table_v_results = replicate_table_v_complete(df_clean, char_cols, Kmax=CONFIG['Kmax'])
        
        if table_v_results and 'panel_a' in table_v_results:
            if not table_v_results['panel_a'].empty:
                output_file_a = f"{OUTPUT_DIR}/tableV_panelA.csv"
                table_v_results['panel_a'].to_csv(output_file_a, index=False)
                logger.info(f"Saved: {output_file_a}")
            
            if 'panel_b' in table_v_results and not table_v_results['panel_b'].empty:
                output_file_b = f"{OUTPUT_DIR}/tableV_panelB.csv"
                table_v_results['panel_b'].to_csv(output_file_b, index=False)
                logger.info(f"Saved: {output_file_b}")
        
        # Step 3: Replicate Table VI
        logger.info("Running COMPLETE Table VI (IS + OOS)...")
        table_vi_complete = replicate_table_vi_complete(
            df_clean, 
            char_cols, 
            Kmax=CONFIG['Kmax'],
            oos_start_date=CONFIG['oos_start_date']
        )
        
        if not table_vi_complete.empty:
            output_file = f"{OUTPUT_DIR}/tableVI_complete.csv"
            table_vi_complete.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
            
            # Panel A: Large stocks
            panel_a = table_vi_complete[['K', 'IS_Total_R2_Large', 'IS_Pred_R2_Large', 
                                         'OOS_Total_R2_Large', 'OOS_Pred_R2_Large']]
            panel_a.to_csv(f"{OUTPUT_DIR}/tableVI_panelA_large.csv", index=False)
            
            # Panel B: Small stocks
            panel_b = table_vi_complete[['K', 'IS_Total_R2_Small', 'IS_Pred_R2_Small',
                                         'OOS_Total_R2_Small', 'OOS_Pred_R2_Small']]
            panel_b.to_csv(f"{OUTPUT_DIR}/tableVI_panelB_small.csv", index=False)
            
            logger.info(f"Saved Panel A and B separately")
        
        
        # Step 4: Replicate Table VII
        logger.info("\n--- STEP 4: TABLE VII (CROSS-VALIDATION) ---")
        table_vii_results = replicate_table_vii_cross_validation(df_raw, char_cols, K=4)
        
        if not table_vii_results.empty:
            output_file = f"{OUTPUT_DIR}/tableVII.csv"
            table_vii_results.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
    except Exception as e:
        logger.error(f"\nERROR: {e}", exc_info=True)

if __name__ == "__main__":
    main()