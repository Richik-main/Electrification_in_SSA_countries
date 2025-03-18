import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, pearsonr, spearmanr, f_oneway
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def generate_all_statisticalTestand_plots(df: pd.DataFrame):
    """
    Perform a series of statistical tests, correlation analyses, regression, and
    create multiple plots using the given DataFrame (df).
    This function consolidates the various code sections you provided into
    a single modular flow.

    Parameters
    ----------
    df : pd.DataFrame
        Your data, which must contain (at least) the following columns:
          1. 'Firm Size Category' containing 'Small' or 'Large'.
          2. 'Backup Power Usage (Own/Shared Generator)' containing 1 (Yes) or 2 (No).
          3. 'Sales Revenue'.
          4. 'Electricity Consumption in Typical Month (kWh)'.
          5. 'Power Outage Impact Score'.
          6. 'Electricity Dependency Ratio'.
          7. 'Sales Revenue per Employee'.
        Adjust the code if your column names or data structures differ.

    Returns
    -------
    None.
    Prints and displays all relevant outputs (statistical test results,
    correlation analyses, bar charts, KDE plots, regression metrics, etc.).
    """

    # -------------------------------------------------------------------------
    # 1. Clean up column names (remove extra spaces, if any).
    # -------------------------------------------------------------------------
    df.columns = df.columns.str.strip()

    # -------------------------------------------------------------------------
    # 2. T-tests: Compare Sales Revenue for different groups
    #    (Firm Size: Small vs Large, and Backup Power: Yes vs No).
    # -------------------------------------------------------------------------
    print("========== TWO-SAMPLE T-TESTS ==========")

    # (a) T-test: Small vs Large firms
    small_sales = df.loc[df['Firm Size Category'] == 'Small', 'Sales Revenue'].dropna()
    large_sales = df.loc[df['Firm Size Category'] == 'Large', 'Sales Revenue'].dropna()
    t_stat_firm, p_value_firm = ttest_ind(small_sales, large_sales, equal_var=False)
    print("\n[1] T-test Results comparing Sales Revenue by Firm Size:")
    print("    T-statistic:", t_stat_firm)
    print("    p-value:    ", p_value_firm)

    # (b) T-test: Backup Power (Yes=1) vs (No=2)
    yes_backup_sales = df.loc[df['Backup Power Usage (Own/Shared Generator)'] == 1, 'Sales Revenue'].dropna()
    no_backup_sales  = df.loc[df['Backup Power Usage (Own/Shared Generator)'] == 2, 'Sales Revenue'].dropna()
    t_stat_backup, p_value_backup = ttest_ind(yes_backup_sales, no_backup_sales, equal_var=False)
    print("\n[2] T-test Results comparing Sales Revenue by Backup Power Usage:")
    print("    T-statistic:", t_stat_backup)
    print("    p-value:    ", p_value_backup)

    # -------------------------------------------------------------------------
    # 3. Grouped Bar Chart by Firm Size Category and Backup Power Usage,
    #    and a separate bar showing overall average by Backup Usage.
    # -------------------------------------------------------------------------
    print("\n========== GROUPED BAR CHARTS ==========")

    # (a) Prepare data for grouped bar chart
    grouped_firm_backup = (
        df.groupby(['Firm Size Category', 'Backup Power Usage (Own/Shared Generator)'])['Sales Revenue']
          .mean()
          .reset_index()
    )

    # (b) Pivot to get a matrix: index=Firm Size Category, columns=Backup Usage
    grouped_pivot = grouped_firm_backup.pivot(
        index='Firm Size Category',
        columns='Backup Power Usage (Own/Shared Generator)',
        values='Sales Revenue'
    )
    # Rename columns to Yes/No
    grouped_pivot = grouped_pivot.rename(columns={1: 'Yes', 2: 'No'})

    # (c) Plot: Grouped bar chart
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    grouped_pivot.plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Average Sales Revenue', fontsize=14)
    ax1.set_title('Average Sales Revenue by Firm Size & Backup Power Usage', fontsize=16, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend(title='Backup Power Usage', title_fontsize=12, fontsize=10, loc='upper right')

    # (d) Overall average by Backup Power Usage
    grouped_backup_avg = (
        df.groupby('Backup Power Usage (Own/Shared Generator)')['Sales Revenue']
          .mean()
          .reset_index()
    )
    # Map numeric codes to Yes/No
    grouped_backup_avg['Backup Power Usage (Own/Shared Generator)'] = (
        grouped_backup_avg['Backup Power Usage (Own/Shared Generator)']
        .replace({1: 'Yes', 2: 'No'})
    )

    # Plot overall bar chart (Yes vs No)
    ax2.bar(grouped_backup_avg['Backup Power Usage (Own/Shared Generator)'],
            grouped_backup_avg['Sales Revenue'],
            color=['C0', 'C1'])
    ax2.set_xlabel('Backup Power Usage (Yes/No)', fontsize=14)
    ax2.set_ylabel('Average Sales Revenue', fontsize=14)
    ax2.set_title('Average Sales Revenue for Firms With/Without Backup Power', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 4. KDE Plot of Sales Revenue by Firm Size (Small vs Large).
    # -------------------------------------------------------------------------
    print("\n========== KDE PLOT: SALES REVENUE by FIRM SIZE ==========")

    df_small = df[df['Firm Size Category'] == 'Small']
    df_large = df[df['Firm Size Category'] == 'Large']

    plt.figure(figsize=(12, 8))
    # Using Seaborn's KDE
    if not df_small.empty:
        sns.kdeplot(data=df_small, x='Sales Revenue', fill=True, alpha=0.5, linewidth=2,
                    label='Small', color='C0')
    if not df_large.empty:
        sns.kdeplot(data=df_large, x='Sales Revenue', fill=True, alpha=0.5, linewidth=2,
                    label='Large', color='C1')

    sns.despine(trim=True)
    plt.title('KDE of Sales Revenue by Firm Size Category', fontsize=18, fontweight='bold')
    plt.xlabel('Sales Revenue', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(title='Firm Size Category', title_fontsize=14, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 5. Grouping by Electricity Consumption into 3 bins, then:
    #    - Pairwise T-tests among the 3 groups
    #    - Bar chart of Average Sales Revenue by Group
    #    - KDE of Sales Revenue by Electricity Group
    # -------------------------------------------------------------------------
    print("\n========== ELECTRICITY CONSUMPTION GROUPS ==========")

    # (a) Create consumption groups via qcut (3 quantile-based bins)
    df['Electricity Group'] = pd.qcut(
        df['Electricity Consumption in Typical Month (kWh)'],
        q=3,
        labels=["Low consumption", "Moderate consumption", "High consumption"]
    )

    # (b) Pairwise T-tests
    groups = ["Low consumption", "Moderate consumption", "High consumption"]
    pvalues = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for i in groups:
        for j in groups:
            if i == j:
                pvalues.loc[i, j] = np.nan
            else:
                data_i = df.loc[df['Electricity Group'] == i, 'Sales Revenue'].dropna()
                data_j = df.loc[df['Electricity Group'] == j, 'Sales Revenue'].dropna()
                t_stat, p_val = ttest_ind(data_i, data_j, equal_var=False)
                pvalues.loc[i, j] = p_val

    print("Pairwise t-test p-values for Sales Revenue among Electricity Groups:")
    print(pvalues)

    # (c) KDE of Sales Revenue by Electricity Group
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.3)

    # For color palette, let's pick a few distinct colors
    palette = ['C0', 'C1', 'C2']
    for i, group in enumerate(groups):
        subset = df[df['Electricity Group'] == group]
        # If there's more than one unique Sales Revenue value, plot a KDE
        if subset['Sales Revenue'].nunique() > 1:
            sns.kdeplot(
                data=subset, x='Sales Revenue', fill=True, alpha=0.5, linewidth=2,
                label=group, color=palette[i]
            )
        else:
            # Just a vertical line if constant
            if not subset.empty:
                constant_val = subset['Sales Revenue'].iloc[0]
                plt.axvline(constant_val, color=palette[i], linestyle='--', linewidth=2,
                            label=f"{group} (constant)")

    sns.despine(trim=True)
    plt.title('KDE of Sales Revenue by Electricity Consumption Groups', fontsize=18, fontweight='bold')
    plt.xlabel('Sales Revenue', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(title='Electricity Group', title_fontsize=14, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

    # (d) Bar chart: Average Sales Revenue by Electricity Group
    grouped_sales_elec = (
        df.groupby('Electricity Group')['Sales Revenue']
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=grouped_sales_elec, x='Electricity Group', y='Sales Revenue', palette='viridis')
    plt.title("Average Sales Revenue by Electricity Consumption Group", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Electricity Consumption Group", fontsize=14, labelpad=10)
    plt.ylabel("Average Sales Revenue", fontsize=14, labelpad=10)
    plt.xticks(rotation=20, ha='right')

    # Annotate each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 6. KDE Plot of Sales Revenue by Backup Power Usage (Yes/No).
    # -------------------------------------------------------------------------
    print("\n========== KDE PLOT: SALES REVENUE BY BACKUP USAGE ==========")

    df_yes_backup = df[df['Backup Power Usage (Own/Shared Generator)'] == 1]
    df_no_backup  = df[df['Backup Power Usage (Own/Shared Generator)'] == 2]

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.3)

    # Plot for 'Yes'
    if not df_yes_backup.empty:
        if df_yes_backup['Sales Revenue'].nunique() > 1:
            sns.kdeplot(
                data=df_yes_backup, x='Sales Revenue', fill=True,
                alpha=0.5, linewidth=2, label='Yes', color='C0'
            )
        else:
            val_const = df_yes_backup['Sales Revenue'].iloc[0]
            plt.axvline(val_const, color='C0', linestyle='--', linewidth=2, label='Yes (constant)')
    else:
        print("No data for Backup=Yes")

    # Plot for 'No'
    if not df_no_backup.empty:
        if df_no_backup['Sales Revenue'].nunique() > 1:
            sns.kdeplot(
                data=df_no_backup, x='Sales Revenue', fill=True,
                alpha=0.5, linewidth=2, label='No', color='C1'
            )
        else:
            val_const = df_no_backup['Sales Revenue'].iloc[0]
            plt.axvline(val_const, color='C1', linestyle='--', linewidth=2, label='No (constant)')
    else:
        print("No data for Backup=No")

    sns.despine(trim=True)
    plt.title('KDE of Sales Revenue by Backup Power Usage', fontsize=18, fontweight='bold')
    plt.xlabel('Sales Revenue', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(title='Backup Power (Yes/No)', title_fontsize=14, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 7. Data Imputation (KNN) and Correlation/Regression Analysis
    # -------------------------------------------------------------------------
    print("\n========== CORRELATION & REGRESSION ANALYSIS ==========")

    # (a) Replace infinite values with NaN in selected columns
    inf_cols = ['Electricity Dependency Ratio', 'Sales Revenue per Employee']
    for col in inf_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # (b) Apply KNN Imputation (example with 5 neighbors)
    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    df[inf_cols] = knn_imputer.fit_transform(df[inf_cols])

    # (c) Pearson & Spearman correlations with 'Sales Revenue'
    #     We'll exclude 'Sales Revenue' from numeric columns.
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['Sales Revenue'])
    pearson_results = {}
    spearman_results = {}

    for col in numeric_columns:
        series_clean = df[col].dropna()
        sr_clean = df['Sales Revenue'].dropna()

        # Must align indexes if we do dropna
        valid_idx = series_clean.index.intersection(sr_clean.index)
        series_clean = series_clean.loc[valid_idx]
        sr_clean = sr_clean.loc[valid_idx]

        if len(series_clean) > 1:  # Need at least 2 data points
            corr_p, pval_p = pearsonr(series_clean, sr_clean)
            pearson_results[col] = {'Pearson_correlation': corr_p, 'p_value': pval_p}

            corr_s, pval_s = spearmanr(series_clean, sr_clean)
            spearman_results[col] = {'Spearman_correlation': corr_s, 'p_value': pval_s}
        else:
            pearson_results[col] = {'Pearson_correlation': np.nan, 'p_value': np.nan}
            spearman_results[col] = {'Spearman_correlation': np.nan, 'p_value': np.nan}

    pearson_df = pd.DataFrame(pearson_results).T.sort_values('p_value')
    spearman_df = pd.DataFrame(spearman_results).T.sort_values('p_value')

    print("--- Pearson Correlation Results (sorted by p-value) ---")
    print(pearson_df)
    print("\n--- Spearman Correlation Results (sorted by p-value) ---")
    print(spearman_df)

    # (d) ANOVA on categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    anova_results = {}
    for col in categorical_columns:
        # Group data by the column
        grouped_vals = [grp['Sales Revenue'].values for _, grp in df.groupby(col)]
        if len(grouped_vals) >= 2:
            f_stat, p_val = f_oneway(*grouped_vals)
            anova_results[col] = {'F-statistic': f_stat, 'p_value': p_val}
        else:
            anova_results[col] = {'F-statistic': None, 'p_value': None, 'note': 'Not enough groups'}

    anova_df = pd.DataFrame(anova_results).T
    print("\n--- ANOVA Results ---")
    print(anova_df)

    # (e) Linear Regression
    #     We'll do a simple multi-feature model: X = numeric_columns, y = 'Sales Revenue'
    X = df[numeric_columns].fillna(0)  # For safety, fill NaN with 0 or handle differently
    y = df['Sales Revenue'].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Linear Regression Performance ---")
    print("R-squared:", r2)
    print("RMSE:     ", rmse)

    # -------------------------------------------------------------------------
    # 8. (Optional) Bar Plots for Pearson / Spearman Correlations
    #    (Excluding 'Sales Revenue per Employee' to mimic original snippet).
    # -------------------------------------------------------------------------
    print("\n========== CORRELATION BAR PLOTS ==========")

    # Exclude 'Sales Revenue per Employee' if present
    pearson_no_sre = pearson_df.drop(index='Sales Revenue per Employee', errors='ignore')
    spearman_no_sre = spearman_df.drop(index='Sales Revenue per Employee', errors='ignore')

    # Pearson bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(pearson_no_sre.index, pearson_no_sre['Pearson_correlation'], color='skyblue')
    plt.title('Pearson Correlation Coefficients (Excluding Sales Revenue per Employee)')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()

    # Spearman bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(spearman_no_sre.index, spearman_no_sre['Spearman_correlation'], color='salmon')
    plt.title('Spearman Correlation Coefficients (Excluding Sales Revenue per Employee)')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 9. Power Outage Impact Score Binning, T-test, and Plots
    # -------------------------------------------------------------------------
    print("\n========== POWER OUTAGE IMPACT ANALYSIS ==========")

    # Create binary categories from the 'Power Outage Impact Score'
    # Adjust q=2 and labels if you want exactly two categories (low/high).
    df['Outage Category'] = pd.qcut(
        df['Power Outage Impact Score'],
        q=2,
        labels=["Low Outages", "High Outages"]
    )

    # (a) Welch T-test between 'Low Outages' and 'High Outages'
    sales_low_out = df.loc[df['Outage Category'] == "Low Outages", 'Sales Revenue'].dropna()
    sales_high_out = df.loc[df['Outage Category'] == "High Outages", 'Sales Revenue'].dropna()
    t_stat_out, p_val_out = ttest_ind(sales_low_out, sales_high_out, equal_var=False)
    print("Welch's t-test comparing Sales Revenue between Low Outages vs High Outages:")
    print(f"  t-statistic: {t_stat_out:.4f}")
    print(f"  p-value:     {p_val_out:.4g}")

    # (b) KDE plot of Sales Revenue by Outage Category
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.3)

    outage_groups = ["Low Outages", "High Outages"]
    colors = ['C0', 'C1']
    for i, group in enumerate(outage_groups):
        subset = df[df['Outage Category'] == group]
        if subset['Sales Revenue'].nunique() > 1:
            sns.kdeplot(
                data=subset, x='Sales Revenue',
                fill=True, alpha=0.5, linewidth=2,
                label=group, color=colors[i]
            )
        else:
            if not subset.empty:
                constant_value = subset['Sales Revenue'].iloc[0]
                plt.axvline(constant_value, color=colors[i], linestyle='--',
                            linewidth=2, label=f"{group} (constant)")

    sns.despine(trim=True)
    plt.title('KDE of Sales Revenue by Outage Category', fontsize=18, fontweight='bold')
    plt.xlabel('Sales Revenue', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(title='Outage Category', title_fontsize=14, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

    # (c) Bar chart of average Sales Revenue by Outage Category
    grouped_outage = df.groupby('Outage Category')['Sales Revenue'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=grouped_outage, x='Outage Category', y='Sales Revenue', palette='viridis')

    # Annotate bars
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + grouped_outage['Sales Revenue'].max() * 0.01,
            f'{height:,.0f}',
            ha='center', va='bottom',
            fontsize=12
        )

    plt.title("Average Sales Revenue by Power Outage Impact Category", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Outage Category", fontsize=14, labelpad=10)
    plt.ylabel("Average Sales Revenue", fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # END: The function prints and shows all results and plots.
    # -------------------------------------------------------------------------
    print("\n===== ALL ANALYSES AND PLOTS ARE COMPLETE. =====")

