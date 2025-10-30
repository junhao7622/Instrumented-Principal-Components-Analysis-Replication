import numpy as np
import pandas as pd
from ipca import InstrumentedPCA
from DataDefinitions.datadefinition import dd
import matplotlib.pyplot as plt
import seaborn as sns

def get_excess_return(df):
    """
    Main function to add excess returns to your dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Your data with 'date' and 'ret' columns
    use_actual_rf : bool
        If True, tries to download actual Fama-French RF data
        If False or download fails, uses approximation

    Returns:
    --------
    pd.DataFrame with 'xret' column added
    """
    df = df.copy()
    #df = data
    factors = dd('famafrench', item='F-F_Research_Data_Factors', start='1900-01-01', end=None)
    data_ff3 = factors.extract()
    rf_series = data_ff3['RF'] / 100

    # Ensure date column is datetime
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    # Create year-month for merging - align to beginning of month
    df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    # Fix the RF dataframe - convert the index to datetime
    rf_df = rf_series.reset_index()
    rf_df.columns = ['year_month', 'rf']

    # Convert rf_df year_month to datetime - handle Period type
    if hasattr(rf_df['year_month'].dtype, 'name') and 'period' in str(rf_df['year_month'].dtype):
        rf_df['year_month'] = rf_df['year_month'].dt.to_timestamp()
    elif rf_df['year_month'].dtype == 'object' or str(rf_df['year_month'].dtype) == 'object':
        rf_df['year_month'] = pd.to_datetime(rf_df['year_month'])

    # Now merge
    df = df.merge(rf_df, on='year_month', how='left')

    # Calculate excess returns
    df['xret'] = df['ret'] - df['rf']

    # Clean up
    df = df.drop(columns=['year_month', 'rf'])

    print(f"Successfully merged Fama-French RF data")
    print(f"Missing RF values: {df['xret'].isna().sum()}")

    return df

# ---------- helpers ----------
def winsorize_series(x: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lower=lo, upper=hi)

def cs_clean_zscore(group: pd.DataFrame, char_cols: list[str],
                    winsor: bool, w_pctl: float) -> pd.DataFrame:
    g = group.copy()
    if winsor:
        for c in char_cols:
            g[c] = winsorize_series(g[c], w_pctl)
    for c in char_cols:
        std = g[c].std(ddof=0)
        g[c] = np.where((std == 0) | np.isnan(std), np.nan, (g[c] - g[c].mean()) / std)
    return g

def make_rank_instruments(df: pd.DataFrame,
                          char_cols: list[str],
                          add_const: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    RNKDMN instruments (paper/Matlab baseline):
      - For each date and each characteristic c: compute cross-sectional ranks,
        then map to [-0.5, 0.5] via (rank-1)/(N-1) - 0.5 using only non-missing obs.
      - (Optional) append a constant instrument as the last column.
    Returns (df_with_instruments, instrument_cols).
    """
    df = df.sort_values(["permno", "date"]).copy()

    def _rank_std(g: pd.DataFrame) -> pd.DataFrame:
        r = g[char_cols].rank(method="average")           # ranks start at 1
        n = g[char_cols].notna().sum()                    # N_t per column at this date
        for c in char_cols:
            nn = int(n[c])
            if nn > 1:
                r[c] = (r[c] - 1) / (nn - 1) - 0.5
            else:
                r[c] = np.nan                              # not enough data this month
        return r

    ranked = (df.groupby("date", group_keys=False)
                .apply(_rank_std))

    df[char_cols] = ranked

    inst_cols = char_cols.copy()
    if add_const:
        df["const"] = 1.0
        inst_cols.append("const")

    return df, inst_cols


def build_panel_matrices(df: pd.DataFrame, x_cols: list[str],
                                keep_dates_with_at_least: int) -> tuple[pd.DataFrame, pd.Series]:
    # Paper requirement: restrict to non-missing observations only (no winsorization/trimming)
    panel = df.set_index(["permno", "date"]).sort_index()

    # Strict: require ALL x_cols to be non-missing (aligns with paper's "non-missing" filter)
    mask = panel[x_cols].notna().all(axis=1)
    panel = panel[mask]

    # Then check date requirements
    counts = panel.groupby("date").size()
    good_months = counts[counts >= keep_dates_with_at_least].index
    panel = panel[panel.index.get_level_values('date').isin(good_months)]

    # Finally ensure we have returns
    panel = panel.dropna(subset=["ret_lead"])

    X = panel[x_cols]
    y = panel["ret_lead"]

    return X, y

def compute_predictive_R2(X: pd.DataFrame, y: pd.Series, regr: InstrumentedPCA) -> float:
    Z = X.to_numpy()
    y_true = y.to_numpy()
    G = regr.Gamma
    K = regr.n_factors

    # Pull factor realizations from statsmodels-ipca
    F = regr.Factors if hasattr(regr, "Factors") else regr.get_factors(label_ind=False)[1].to_numpy()

    if F.shape[0] == K + 1 and np.allclose(F[-1, :], 1.0):
        lam = F[:K, :].mean(axis=1)
    elif F.shape[1] == K + 1 and np.allclose(F[:, -1], 1.0):
        lam = F[:, :K].mean(axis=0)
    elif F.shape[0] == K:
        lam = F.mean(axis=1)
    elif F.shape[1] == K:
        lam = F.mean(axis=0)
    else:
        raise ValueError(f"Unexpected factor shape {F.shape} for K={K}")

    c_beta = G[:, :K] @ lam
    mu = Z @ c_beta
    if G.shape[1] > K:
        mu += Z @ G[:, K]
    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1.0 - ss_res / ss_tot


def R2_total_matlab_style_from_Xy(X, y, regr):
    """
    Total R^2 that mirrors MATLAB:
      y_tplus1  vs  yhat_tplus1,  where yhat_tplus1 = Z_t Γ F_t
    Assumes `y` is already r_{i,t+1} stored on row (i,t).
    """
    # Predict using time-varying factors on THIS sample
    # (keeps your X/y mask and ordering)
    yhat = regr.predict(X=X, indices=None, mean_factor=False, data_type="panel")

    # Align to the same rows used inside predict()
    y_aligned = y.loc[X.index].to_numpy()

    m = np.isfinite(yhat) & np.isfinite(y_aligned)
    ss_res = np.nansum((y_aligned[m] - yhat[m])**2)
    ss_tot = np.nansum(y_aligned[m]**2)
    return 1.0 - ss_res/ss_tot


def IPCA_run(df: pd.DataFrame,
             char_cols: list[str],
             n_factors: int = 4,
             keep_dates_with_at_least: int = 10,
             include_intercept: bool = False,
             # choose standardization:
             standardize: str = "zscore",  # "zscore" or "rank"
             winsor: bool = False,  # used only when standardize="zscore"
             w_pctl: float = 0.01,  # used only when standardize="zscore"
             # significance testing parameters:
             test_significance: bool = False,  # whether to run bootstrap tests
             test_chars: list = None,  # specific characteristics to test (if None, tests all)
             ndraws: int = 1000,  # number of bootstrap draws
             n_jobs: int = 1,  # parallel jobs for bootstrap
             backend: str = 'loky',  # backend for parallel processing
             # SDF Sharpe ratio:
             compute_sdf_sharpe: bool = False  # whether to compute in-sample SDF Sharpe ratio (paper uses tangency Sharpe)
             ):
    """
    Returns: total_R2, predictive_R2, Gamma_df, Factors_df, [significance_results], [sdf_sharpe_ratio]

    standardize="rank": paper-style instruments
      - per char: [mean over time, deviation from mean]
      - per date: rank / N - 0.5 across stocks
      - set mean_type="historical" for OOS-style expanding means

    test_chars: list of characteristic names to test for significance
      - If None, tests all characteristics
      - Must be subset of char_cols used in model
      - Only used when test_significance=True and include_intercept=False

    compute_sdf_sharpe: bool = False      # whether to compute in-sample SDF Sharpe ratio

    If test_significance=True, also returns a dictionary with significance test results:
      - alpha_pval: p-value for H0: intercept = 0 (only if include_intercept=True)
      - char_pvals: dict mapping characteristic names to p-values for H0: loading = 0

    If compute_sdf_sharpe=True, also returns the in-sample SDF Sharpe ratio:
      - sdf_sharpe_ratio: scalar value of the SDF Sharpe ratio (lambda' * Sigma^-1 * lambda)
    """
    if standardize == "rank":
        df_inst, x_cols = make_rank_instruments(df, char_cols)
        X, y = build_panel_matrices(df_inst, x_cols, keep_dates_with_at_least)
    elif standardize == "zscore":
        df_z = (df.groupby("date", group_keys=False)
                .apply(cs_clean_zscore, char_cols=char_cols, winsor=winsor, w_pctl=w_pctl))
        X, y = build_panel_matrices(df_z, char_cols, keep_dates_with_at_least)
    else:
        raise ValueError('standardize must be "zscore" or "rank"')

    regr = InstrumentedPCA(n_factors=n_factors, intercept=include_intercept).fit(X=X, y=y)

    total_R2 = R2_total_matlab_style_from_Xy(X, y, regr)

    Gamma_df, Factors_df = regr.get_factors(label_ind=True)

    cols = [f"f{k + 1}" for k in range(regr.n_factors)]
    if regr.Gamma.shape[1] > regr.n_factors:
        cols += ["alpha"]
    Gamma_df.columns = cols

    predictive_R2 = compute_predictive_R2(X, y, regr)

    # Compute SDF Sharpe ratio if requested
    sdf_sharpe_ratio = None
    if compute_sdf_sharpe:
        # Extract the latent factors (exclude intercept if present)
        if include_intercept:
            factors_for_sdf = regr.Factors[:-1, :]  # Exclude intercept factor
        else:
            factors_for_sdf = regr.Factors

        # For K=1, this is much simpler
        if n_factors == 1:
            factor_returns = factors_for_sdf.flatten()  # Convert to 1D array
            sdf_sharpe_ratio = np.mean(factor_returns) / np.std(factor_returns)
        else:
            # Multi-factor case
            lambda_vec = np.mean(factors_for_sdf, axis=1).reshape(-1, 1)
            Sigma = np.cov(factors_for_sdf)
            try:
                Sigma_inv = np.linalg.inv(Sigma)
                sdf_sharpe_ratio = float(lambda_vec.T @ Sigma_inv @ lambda_vec) ** 0.5
            except np.linalg.LinAlgError:
                print("Warning: Factor covariance matrix is singular.")
                sdf_sharpe_ratio = None

    # Run significance tests if requested
    if test_significance:
        significance_results = {}

        # Test intercept significance (alpha = 0)
        if include_intercept:
            print("Testing intercept significance...")
            alpha_pval = regr.BS_Walpha(ndraws=ndraws, n_jobs=n_jobs, backend=backend)
            significance_results['alpha_pval'] = alpha_pval
            print(f"Intercept p-value: {alpha_pval:.4f}")

            print("\nWARNING: Cannot test individual characteristic loadings when intercept is included.")
            print("To test characteristic loadings, set include_intercept=False and run again.")

        else:
            # Test individual characteristic loadings (only when no intercept)
            print("Testing characteristic loading significance...")
            char_pvals = {}

            # Get the characteristic names from the Gamma dataframe index
            char_names = Gamma_df.index.tolist()

            # Determine which characteristics to test
            if test_chars is None:
                chars_to_test = char_names  # Test all
            else:
                # Validate that test_chars are in the model
                invalid_chars = [c for c in test_chars if c not in char_names]
                if invalid_chars:
                    raise ValueError(f"test_chars contains characteristics not in model: {invalid_chars}")
                chars_to_test = test_chars

            print(f"Testing {len(chars_to_test)} characteristics: {chars_to_test}")

            for char_name in chars_to_test:
                i = char_names.index(char_name)  # Get index of characteristic
                print(f"Testing {char_name} (characteristic {i})...")
                pval = regr.BS_Wbeta(l=i, ndraws=ndraws, n_jobs=n_jobs, backend=backend)
                char_pvals[char_name] = pval
                print(f"{char_name} p-value: {pval:.4f}")

            significance_results['char_pvals'] = char_pvals

        if compute_sdf_sharpe:
            return total_R2, predictive_R2, Gamma_df.sort_index(), Factors_df.sort_index(), significance_results, sdf_sharpe_ratio
        else:
            return total_R2, predictive_R2, Gamma_df.sort_index(), Factors_df.sort_index(), significance_results

    elif compute_sdf_sharpe:
        return total_R2, predictive_R2, Gamma_df.sort_index(), Factors_df.sort_index(), sdf_sharpe_ratio

    else:
        return total_R2, predictive_R2, Gamma_df.sort_index(), Factors_df.sort_index()


def print_significance_results(significance_results, alpha_level=0.05):
    """
    Helper function to nicely print significance test results
    """
    print("\n" + "=" * 50)
    print("SIGNIFICANCE TEST RESULTS")
    print("=" * 50)

    if 'alpha_pval' in significance_results:
        alpha_pval = significance_results['alpha_pval']
        alpha_sig = "***" if alpha_pval < 0.01 else "**" if alpha_pval < 0.05 else "*" if alpha_pval < 0.10 else ""
        print(f"\nIntercept (alpha) test:")
        print(f"  H0: alpha = 0")
        print(f"  p-value: {alpha_pval:.4f} {alpha_sig}")
        print(f"  Result: {'Reject H0' if alpha_pval < alpha_level else 'Fail to reject H0'} at {alpha_level} level")

    if 'char_pvals' in significance_results:
        char_pvals = significance_results['char_pvals']
        print(f"\nCharacteristic loading tests:")
        print(f"  H0: characteristic loading = 0")
        print(f"  {'Characteristic':<20} {'p-value':<10} {'Significance':<5} {'Result':<20}")
        print(f"  {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 20}")

        for char_name, pval in char_pvals.items():
            sig_stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            result = 'Significant' if pval < alpha_level else 'Not significant'
            print(f"  {char_name:<20} {pval:<10.4f} {sig_stars:<12} {result:<20}")

    print("\nSignificance levels: *** p<1%, ** p<5%, * p<10%")
    print("=" * 50)

def format_r2_latex(wide):
    """
    Expects `wide` like the example (after `reset_index().T`),
    with rows:
      'Total_R2, Γα = 0', 'Total_R2, Γα ≠ 0',
      'Predictive_R2, Γα = 0', 'Predictive_R2, Γα ≠ 0'
    """
    df = wide.copy()
    # normalize index just in case
    df.index = df.index.astype(str).str.strip()

    total_rows = ['Total_R2, Γα = 0', 'Total_R2, Γα ≠ 0']
    pred_rows  = ['Predictive_R2, Γα = 0', 'Predictive_R2, Γα ≠ 0']

    # scale and round
    df.loc[total_rows] = df.loc[total_rows].astype(float).mul(100).round(1)
    df.loc[pred_rows]  = df.loc[pred_rows].astype(float).mul(100).round(2)

    # helpers to format with fixed decimals
    def fmt_row(row_vals, places):
        fmt = f"{{:.{places}f}}"
        return [fmt.format(float(x)) for x in row_vals.to_numpy()]

    tot0 = fmt_row(df.loc['Total_R2, Γα = 0'], 1)
    tot1 = fmt_row(df.loc['Total_R2, Γα ≠ 0'], 1)
    pr0  = fmt_row(df.loc['Predictive_R2, Γα = 0'], 2)
    pr1  = fmt_row(df.loc['Predictive_R2, Γα ≠ 0'], 2)

    # build LaTeX lines (no % symbols)
    line1 = r"\multirow{2}{*}{Total $R^2$} & $\Gamma_\alpha = 0$   & " + " & ".join(tot0) + r" \\"
    line2 = r"                             & $\Gamma_\alpha \neq 0$ & " + " & ".join(tot1) + r" \\"
    line3 = r"\multirow{2}{*}{Pred.\ $R^2$} & $\Gamma_\alpha = 0$   & " + " & ".join(pr0) + r" \\"
    line4 = r"                             & $\Gamma_\alpha \neq 0$ & " + " & ".join(pr1) + r" \\"

    latex_block = "\n".join([line1, line2, line3, line4])
    print(latex_block)  # ready to copy-paste
    return latex_block

def print_topics_table(df, topics):
    for topic in topics:
        if topic in df.index:
            values = df.loc[topic].round(3).tolist()
            values_str = " & ".join(f"{v:.3f}" for v in values)
            print(f"{topic} & {values_str} \\\\")

def format_pvalue_with_stars(pval):
    """
    Format p-value with star notation
    Returns string like "0.042**"
    """
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    return f"{pval:.3f}{stars}"

def print_compact_significance_results(significance_results):
    """
    Print compact format of significance results with p-values and stars
    """
    if 'char_pvals' in significance_results:
        char_pvals = significance_results['char_pvals']
        print("Characteristic significance results:")
        for char_name, pval in char_pvals.items():
            formatted_pval = format_pvalue_with_stars(pval)
            print(f"{char_name}: {formatted_pval}")
        print("\nSignificance levels: *** p<1%, ** p<5%, * p<10%")

    if 'alpha_pval' in significance_results:
        alpha_pval = significance_results['alpha_pval']
        formatted_alpha = format_pvalue_with_stars(alpha_pval)
        print(f"Intercept (alpha): {formatted_alpha}")


def plot_topics_over_time(df, columns_to_plot):
    """
    Plot daily sums of specified columns over time.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with a 'date' column
    columns_to_plot : list
        List of column names to plot (will be summed by date)
    """

    # Group by date and sum the specified columns
    daily_sums = df.groupby('date')[columns_to_plot].sum().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot each column
    for col in columns_to_plot:
        plt.plot(daily_sums['date'], daily_sums[col],
                 linewidth=3, label=col, alpha=0.8)

    # Styling preferences
    plt.legend(fontsize=14, frameon=False)
    plt.tick_params(labelsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Daily Sum', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Remove top and right spines for cleaner look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

    return daily_sums

if __name__ == "__main__":
    #####################################################################
    # Setup char data
    #####################################################################
    data = pd.read_csv("/Users/junhao/Desktop/UCSD/ucsd_Econ_Lab/ipca replication/shared project/characteristics_data_feb2017.csv")
    data = data[data['lme']>0]