#!/usr/bin/env python
"""
Data Preprocessing Module

This module loads configuration settings from a YAML file, performs data loading,
initial cleaning, advanced cleaning, and feature engineering on the dataset.
The processing includes:
  - Dropping unnecessary columns and removing rows with invalid placeholders.
  - KNN imputation for numeric columns.
  - Advanced imputation for the 'Backup Power Usage (Own/Shared Generator)' column
    using a KNN classifier after label encoding categorical features.
  - Conversion of Sales Revenue to USD based on country-specific exchange rates.
  - Outlier detection and removal on Sales Revenue using the Local Outlier Factor.
  - Feature engineering (calculating power outage impact, dependency ratios, age and size categories, etc.)

Configuration parameters (e.g., file path, columns to drop, invalid values, imputation parameters,
and thresholds for firm age and size) are loaded from a YAML file (default: config.yaml).
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn and Matplotlib styles for consistency
sns.set(style='whitegrid', palette='Set2')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14
})


def load_config(config_path="config.yaml"):
    """
    Load YAML configuration file.

    Parameters:
      config_path (str): Path to the YAML configuration file.

    Returns:
      dict: Configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def categorize_firm_age(age, cutoffs):
    """
    Categorize firms based on years since establishment.

    Parameters:
      age (int or float): Years since establishment.
      cutoffs (list): List of threshold values, e.g. [20, 40, 70].

    Returns:
      str: The category label (Startup, Young, Established, or Mature).
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
    Categorize firms by size.

    Parameters:
      size (int): Number of full-time employees.
      threshold (int): Threshold for small vs. large firms.

    Returns:
      str: 'Small' if size is less than or equal to threshold, else 'Large'.
    """
    return 'Small' if size <= threshold else 'Large'


def preprocess_data(config_path="config.yaml"):
    """
    Main function to load, clean, and engineer features in the dataset.

    Steps:
      1. Load configuration and dataset.
      2. Initial cleanup: drop columns and remove rows with invalid placeholders.
      3. KNN imputation for numeric columns.
      4. Advanced imputation for the backup power usage column using a KNN classifier.
      5. Convert Sales Revenue to USD using exchange rates.
      6. Outlier detection and removal using Local Outlier Factor (LOF) on Sales Revenue.
      7. Feature engineering:
         - Power Outage Impact Score.
         - Electricity Dependency Ratio.
         - Working Capital Dependency.
         - Firm Age and Size Categories.
         - Sales Revenue per Employee.
         - Backup Power Dependency.
         - Local Ownership.

    Parameters:
      config_path (str): Path to the YAML configuration file.

    Returns:
      pandas.DataFrame: The fully preprocessed DataFrame.
    """
    # -----------------------------
    # 1. Load Configuration and Data
    # -----------------------------
    config = load_config(config_path)
    file_path = config['file_path']
    columns_to_drop = config.get('columns_to_drop', [])
    invalid_values = config.get('invalid_values', [])
    invalid_values2 = config.get('invalid_values2', [])
    imputation_neighbors = config.get('imputation_neighbors', 5)
    target_column = config['target_column']  # e.g., "Sales Revenue"
    age_cutoffs = config['age_cutoffs']
    firm_size_threshold = config['firm_size_threshold']

    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Print initial missing values per column
    print("Missing values per column (before cleaning):")
    print(df.isna().sum())

    # -----------------------------
    # 2. Initial Cleanup
    # -----------------------------
    # Drop unnecessary columns (if they exist)
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Remove rows with placeholder value -9 (assumed to indicate invalid data)
    df = df[(df != -9).all(axis=1)]

    # Remove rows where 'Sales Revenue' contains invalid values (e.g., -8, -7, 0)
    df = df[~df['Sales Revenue'].isin(invalid_values2)]

    # Replace infinite values with NaN to enable imputation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -----------------------------
    # 3. KNN Imputation for Numeric Columns
    # -----------------------------
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    knn_imputer = KNNImputer(n_neighbors=imputation_neighbors)
    df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

    print("\nMissing values per column (after imputation):")
    print(df.isna().sum())
    print("\nShape (after imputation):")
    print(df.shape)

    # -----------------------------
    # 4. Advanced Imputation for 'Backup Power Usage (Own/Shared Generator)'
    # -----------------------------
    backup_power_target = 'Backup Power Usage (Own/Shared Generator)'
    if backup_power_target in df.columns:
        # Use all other columns as predictors
        features = [col for col in df.columns if col != backup_power_target]

        # For predictor columns, fill missing values:
        #   - For categorical columns, use the mode.
        #   - For numeric columns, reapply KNN imputation.
        for col in features:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

        # Split data into rows with known and missing backup power values
        train_data = df[df[backup_power_target].notna()].copy()
        missing_data = df[df[backup_power_target].isna()].copy()

        # Label-encode categorical predictors in both datasets
        label_encoders = {}
        for col in features:
            if train_data[col].dtype == 'object':
                le = LabelEncoder()
                train_data[col] = le.fit_transform(train_data[col])
                missing_data[col] = le.transform(missing_data[col])
                label_encoders[col] = le

        # Prepare training inputs and target
        X_train = train_data[features]
        y_train = train_data[backup_power_target]

        # If target is categorical, encode it (typically it is numeric)
        if train_data[backup_power_target].dtype == 'object':
            target_le = LabelEncoder()
            y_train = target_le.fit_transform(y_train)
        else:
            target_le = None

        # Train a KNN classifier (using 5 neighbors) to predict missing backup power usage
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train, y_train)

        # Predict missing backup power usage values
        X_missing = missing_data[features]
        predicted = knn_classifier.predict(X_missing)

        # Convert predictions back to original labels if necessary
        if target_le is not None:
            predicted = target_le.inverse_transform(predicted)

        # Fill the missing values in the original DataFrame
        df.loc[df[backup_power_target].isna(), backup_power_target] = predicted

    # -----------------------------
    # 5. Convert Sales Revenue to USD
    # -----------------------------
    # Define country-specific exchange rates
    exchange_rates = {
        'Angola': 0.00110,
        'Benin': 0.00166,
        'Botswana': 0.0732,
        'Burkina-Faso': 0.00166,
        'Burundi': 0.00034,
        "Côte d'Ivoire": 0.00166,
        'CAR': 0.00166,
        'CaboVerde': 0.00988,
        'Cameroon': 0.00166,
        'Cape-Verde': 0.00988,
        'Central-African-Republic': 0.00166,
        'Chad': 0.00166,
        'Congo--Rep.': 0.00166,
        'CongoRep': 0.00166,
        'DRC': 0.00035,
        'Djibouti': 0.00562,
        'Eritrea': 0.0663,
        'Eswatini': 0.0546,
        'Ethiopia': 0.00777,
        'Gabon': 0.00166,
        'Gambia': 0.01429,
        'Ghana': 0.0647,
        'Kenya': 0.00773,
        'Lesotho': 0.05455,
        'Liberia': 0.00505,
        'Madagascar': 0.000214,
        'Malawi': 0.000582,
        'Mali': 0.00166,
        'Mauritania': 0.02515,
        'Mauritius': 0.02237,
        'Mozambique': 0.01581,
        'Namibia': 0.0546,
        'Niger': 0.00166,
        'Nigeria': 0.000653,
        'Rwanda': 0.000716,
        'Senegal': 0.00166,
        'Seychelles': 0.0690,
        'Sierra-Leone': 0.0000441,
        'South-Africa': 0.05455,
        'SouthSudan': 0.00154,
        'Sudan': 0.00167,
        'Tanzania': 0.000377,
        'Togo': 0.00166,
        'Uganda': 0.000273,
        'Zambia': 0.03500,
        'Zimbabwe': 0.0357
    }

    def convert_sales_to_usd(row):
        """
        Convert Sales Revenue to USD using the appropriate exchange rate.

        Parameters:
          row (pandas.Series): A row from the DataFrame.

        Returns:
          float: Sales Revenue converted to USD.
        """
        country = row['Country']
        rate = exchange_rates.get(country, 1)  # Default rate is 1 if country not found
        return row['Sales Revenue'] * rate

    if 'Sales Revenue' in df.columns and 'Country' in df.columns:
        df['Sales Revenue'] = df.apply(convert_sales_to_usd, axis=1)
        print("\nFirst 10 rows after converting Sales Revenue to USD:")
        print(df[['Country', 'Sales Revenue']].head(10))

    # -----------------------------
    # 6. Outlier Detection and Removal using Local Outlier Factor
    # -----------------------------
    if 'Sales Revenue' in df.columns:
        sales_revenue_values = df[['Sales Revenue']]
        lof = LocalOutlierFactor(n_neighbors=154)
        df['LOF_labels'] = lof.fit_predict(sales_revenue_values)
        # Keep only inliers (LOF label == 1)
        df = df[df['LOF_labels'] == 1].copy()
        df.drop(columns=['LOF_labels'], inplace=True)
        print("Dataset shape after removing outliers:", df.shape)

    # -----------------------------
    # 7. Feature Engineering
    # -----------------------------
    # 7.1 Power Outage Impact Score: number of outages multiplied by average duration
    if all(col in df.columns for col in
           ["Number of Power Outages per Month", "Average Duration of Power Outages (Hours)"]):
        df['Power Outage Impact Score'] = (
                df["Number of Power Outages per Month"] * df["Average Duration of Power Outages (Hours)"]
        )

    # 7.2 Electricity Dependency Ratio: monthly consumption divided by firm size
    if all(col in df.columns for col in
           ["Electricity Consumption in Typical Month (kWh)", "Firm Size (Full-Time Employees)"]):
        df['Electricity Dependency Ratio'] = (
                df["Electricity Consumption in Typical Month (kWh)"] / df["Firm Size (Full-Time Employees)"]
        )

    # 7.3 Working Capital Dependency: sum of various working capital financing percentages
    wc_cols = [
        '% of Working Capital Borrowed from Banks',
        '% of Working Capital Borrowed from Non-Bank Financial Institutions',
        '% of Working Capital Purchased on Credit/Advances',
        '% of Working Capital Financed by Other (Money Lenders, Friends, Relatives)'
    ]
    existing_wc_cols = [col for col in wc_cols if col in df.columns]
    if existing_wc_cols:
        df['Working Capital Dependency'] = df[existing_wc_cols].sum(axis=1)

    # 7.4 Firm Age Category based on years since establishment
    if 'Firm Age (Years Since Establishment)' in df.columns:
        df['Firm Age Category'] = df['Firm Age (Years Since Establishment)'].apply(
            lambda x: categorize_firm_age(x, age_cutoffs)
        )

    # 7.5 Sales Revenue per Employee
    if all(col in df.columns for col in [target_column, "Firm Size (Full-Time Employees)"]):
        df['Sales Revenue per Employee'] = df[target_column] / df["Firm Size (Full-Time Employees)"]

    # 7.6 Backup Power Dependency: binary indicator if backup power usage data exists
    if 'Backup Power Usage (Own/Shared Generator)' in df.columns:
        df['Backup Power Dependency'] = df['Backup Power Usage (Own/Shared Generator)'].notnull().astype(int)

    # 7.7 Local Ownership: computed as 100 minus the percentage owned by private foreign individuals
    if '% Owned by Private Foreign Individuals' in df.columns:
        df['Local Ownership'] = 100 - df['% Owned by Private Foreign Individuals']

    # 7.8 Firm Size Category based on a threshold for full-time employees
    if "Firm Size (Full-Time Employees)" in df.columns:
        df['Firm Size Category'] = df["Firm Size (Full-Time Employees)"].apply(
            lambda x: categorize_firm_size(x, firm_size_threshold)
        )

    print("\nColumns after feature engineering:")
    print(df.columns)

    return df


if __name__ == "__main__":
    # Run the preprocessing pipeline and display the first few rows of the resulting DataFrame.
    processed_df = preprocess_data("config.yaml")
    print("\nPreprocessed DataFrame Head:")
    print(processed_df.head())
