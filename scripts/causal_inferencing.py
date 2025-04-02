import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.impute import KNNImputer


# Assume these functions are defined in your project:
# from config_module import load_config, preprocess_data, train_and_evaluate_models
# from visualization_module import generate_all_plots

def generate_all_casualTest(df):
    """
    This function performs the following:
      - Checks for infinite values in numeric columns and replaces them with NaN.
      - Applies KNN imputation to numeric columns.
      - Conducts an IV regression (2SLS) analysis along with multicollinearity checks
        and a Breusch-Pagan test for heteroskedasticity.
    """
    # --------------------------
    # 1. Handle 'inf' Values and Imputation
    # --------------------------
    numeric_df = df.select_dtypes(include=[np.number])

    # Count the number of 'inf' values per numeric column.
    inf_counts = numeric_df.apply(lambda col: np.isinf(col).sum())
    inf_counts_filtered = inf_counts[inf_counts > 0].reset_index()
    inf_counts_filtered.columns = ['Column', 'Number of inf values']
    print("=== Count of 'inf' values in numeric columns ===")
    print(inf_counts_filtered)

    # Replace 'inf' values with NaN in specified columns if they exist.
    if 'Electricity Dependency Ratio' in df.columns:
        df['Electricity Dependency Ratio'].replace(np.inf, np.nan, inplace=True)
    if 'Sales Revenue per Employee' in df.columns:
        df['Sales Revenue per Employee'].replace(np.inf, np.nan, inplace=True)

    # Apply KNN Imputation to all numeric columns.
    numeric_df_for_imputation = df.select_dtypes(include=[np.number])
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(numeric_df_for_imputation)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df_for_imputation.columns)
    df.update(imputed_df)

    # --------------------------
    # 2. IV Regression (2SLS) Analysis
    # --------------------------
    # Define the selected columns exactly as they appear in your DataFrame.
    selected_columns = [
        'Average Duration of Power Outages (Minutes)',
        '% Owned by Private Domestic Individuals',
        '% Owned by Private Foreign Individuals',
        '% of Working Capital Financed from Internal Funds',
        'Year',
        'Sales Revenue',
        'Firm Size (Full-Time Employees)',
        'Power Outages Experienced in Last FY',
        'Number of Power Outages per Month',
        # 'Losses Due to Power Outages (Value)',
        'Backup Power Usage (Own/Shared Generator)',
        'Firm Age (Years Since Establishment)',
        # 'Industry Classification (Sampling Sector)',
        # 'Sampling Region',
        # 'Region of the Establishment',
        '% of Working Capital Borrowed from Banks',
        '% of Working Capital Borrowed from Non-Bank Financial Institutions',
        '% of Working Capital Purchased on Credit/Advances',
        '% of Working Capital Financed by Other (Money Lenders, Friends, Relatives)',
        'Average Duration of Power Outages (Hours)',
        'Electricity Consumption in Typical Month (kWh)',
        # 'Industry Classification (ISIC Rev. 4 Code)',
        'Power Outage Impact Score',
        'Electricity Dependency Ratio',
        'Working Capital Dependency',
        # 'Sales Revenue per Employee',
        'Local Ownership'
    ]

    # Create a cleaned DataFrame by selecting the needed columns and dropping missing values.
    df_numeric = df[selected_columns].dropna()

    # Define variables for the IV regression:
    # Outcome variable: Sales Revenue.
    y = df_numeric["Sales Revenue"]
    # Endogenous regressor: Backup Power Usage (Own/Shared Generator).
    endog = df_numeric[["Backup Power Usage (Own/Shared Generator)"]]
    # Instrument: Number of Power Outages per Month.
    instr = df_numeric[["Number of Power Outages per Month"]]
    # Exogenous regressors: all other predictors.
    exog_cols = [col for col in df_numeric.columns
                 if col not in ["Sales Revenue", "Backup Power Usage (Own/Shared Generator)",
                                "Number of Power Outages per Month"]]
    exog = df_numeric[exog_cols]
    exog = sm.add_constant(exog)

    # --------------------------
    # 3. Multicollinearity Check (VIF Calculation)
    # --------------------------
    exog_endog = pd.concat([exog, endog], axis=1)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = exog_endog.columns
    vif_data["VIF"] = [variance_inflation_factor(exog_endog.values, i)
                       for i in range(exog_endog.shape[1])]
    print("\n=== Variance Inflation Factors (VIF) ===")
    print(vif_data)

    # Remove variables with high VIF (e.g., VIF > 10) if needed.
    high_vif_vars = vif_data[vif_data["VIF"] > 10]["Variable"].tolist()
    exog = exog.drop(columns=high_vif_vars, errors='ignore')
    exog_cols = [col for col in exog_cols if col not in high_vif_vars]

    # --------------------------
    # 4. Fit the 2SLS (IV) Model
    # --------------------------
    iv_model = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instr)
    iv_results = iv_model.fit(cov_type='robust')

    # --------------------------
    # 5. Heteroskedasticity Test (Breusch-Pagan Test)
    # --------------------------
    resids_array = iv_results.resids.to_numpy(dtype=float)
    exog_array = exog.to_numpy(dtype=float)
    bp_test = het_breuschpagan(resids_array, exog_array)
    bp_stat, bp_pvalue, bp_fvalue, bp_fpvalue = bp_test
    print("\n=== Breusch-Pagan Test for Heteroskedasticity ===")
    print(f"BP Statistic: {bp_stat:.3f}")
    print(f"Degrees of Freedom: {iv_results.model.exog.shape[1] - 1}")
    print(f"P-value: {bp_pvalue:.5f}")

    # --------------------------
    # 6. Display IV Regression Results
    # --------------------------
    print("\n=== 2SLS Results for Backup Power Usage (Own/Shared Generator) as Endogenous ===")
    print("Instrument: Number of Power Outages per Month")
    print(iv_results.summary)

    print("\n=== First Stage Regression Results ===")
    print(iv_results.first_stage.summary)