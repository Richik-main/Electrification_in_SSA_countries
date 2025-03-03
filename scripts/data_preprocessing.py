# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import yaml

def load_config(config_path="config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def categorize_firm_age(age, cutoffs):
    """
    Categorize firms based on years since establishment.
    cutoffs = [20, 40, 70]
    """
    if age < cutoffs[0]:
        return 'Startup'
    elif cutoffs[0] < age < cutoffs[1]:
        return 'Young'
    elif cutoffs[1] <= age < cutoffs[2]:
        return 'Established'
    else:
        return 'Mature'

def categorize_firm_size(size, threshold):
    """
    Categorize firms into 'Small' if <= threshold employees, otherwise 'Large'.
    """
    if size <= threshold:
        return 'Small'
    else:
        return 'Large'

def preprocess_data(config_path="config.yaml"):
    """
    Main function to load the data, drop columns, handle invalid values,
    perform feature engineering, and return a cleaned dataframe.
    """

    # -----------------------------
    # 1. Load Configuration
    # -----------------------------
    config = load_config(config_path)
    file_path = config['file_path']
    columns_to_drop = config.get('columns_to_drop', [])
    invalid_values = config.get('invalid_values', [])
    invalid_values2 = config.get('invalid_values2', [])
    imputation_neighbors = config.get('imputation_neighbors', 5)
    target_column = config['target_column']
    age_cutoffs = config['age_cutoffs']
    firm_size_threshold = config['firm_size_threshold']

    # -----------------------------
    # 2. Load Data
    # -----------------------------
    df = pd.read_csv(file_path)

    # -----------------------------
    # 3. Initial Cleanup
    # -----------------------------
    print("Missing values per column (before cleaning):")
    print(df.isna().sum())

    # Drop the specified columns if they exist
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Remove rows containing invalid placeholder values
    # (e.g., -9 indicates invalid data)
    # for val in invalid_values:
    #     df = df[(df != val).all(axis=1)]
    df = df[(df != -9).all(axis=1)]

    #df = df[~df['Sales Revenue'].isin(invalid_values2)]
    df = df[~df['Sales Revenue'].isin(invalid_values2)]
    # Replace inf with NaN so we can impute
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -----------------------------
    # 4. KNN Imputation for numeric columns
    # -----------------------------
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    knn_imputer = KNNImputer(n_neighbors=imputation_neighbors)
    df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

    print("\nMissing values per column (after imputation):")
    print(df.isna().sum())
    print("\nShape (after imputation):")
    print(df.shape)


    # -----------------------------
    # 5. Feature Engineering
    # -----------------------------
    # 5.1 Power Outage Impact Score
    if all(col in df.columns for col in ["Number of Power Outages per Month", "Average Duration of Power Outages (Hours)"]):
        df['Power Outage Impact Score'] = df['Number of Power Outages per Month'] * df['Average Duration of Power Outages (Hours)']

    # 5.2 Electricity Dependency Ratio
    if all(col in df.columns for col in ["Electricity Consumption in Typical Month (kWh)", "Firm Size (Full-Time Employees)"]):
        df['Electricity Dependency Ratio'] = df['Electricity Consumption in Typical Month (kWh)'] / df['Firm Size (Full-Time Employees)']

    # 5.3 Working Capital Dependency
    wc_cols = [
        '% of Working Capital Borrowed from Banks',
        '% of Working Capital Borrowed from Non-Bank Financial Institutions',
        '% of Working Capital Purchased on Credit/Advances',
        '% of Working Capital Financed by Other (Money Lenders, Friends, Relatives)'
    ]
    existing_wc_cols = [col for col in wc_cols if col in df.columns]
    if existing_wc_cols:
        df['Working Capital Dependency'] = df[existing_wc_cols].sum(axis=1)

    # 5.4 Firm Age Category
    if 'Firm Age (Years Since Establishment)' in df.columns:
        df['Firm Age Category'] = df['Firm Age (Years Since Establishment)'].apply(lambda x: categorize_firm_age(x, age_cutoffs))

    # 5.5 Sales Revenue per Employee
    if all(col in df.columns for col in [target_column, "Firm Size (Full-Time Employees)"]):
        df['Sales Revenue per Employee'] = df[target_column] / df['Firm Size (Full-Time Employees)']

    # 5.6 Backup Power Dependency
    if 'Backup Power Usage (Own/Shared Generator)' in df.columns:
        df['Backup Power Dependency'] = df['Backup Power Usage (Own/Shared Generator)'].notnull().astype(int)

    # 5.7 Local Ownership
    if '% Owned by Private Foreign Individuals' in df.columns:
        df['Local Ownership'] = 100 - df['% Owned by Private Foreign Individuals']

    # 5.8 Firm Size Category
    if "Firm Size (Full-Time Employees)" in df.columns:
        df['Firm Size Category'] = df['Firm Size (Full-Time Employees)'].apply(lambda x: categorize_firm_size(x, firm_size_threshold))

    print("\nColumns after feature engineering:")
    print(df.columns)

    return df
