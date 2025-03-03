# scripts/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set Seaborn and Matplotlib styles for consistency
sns.set(style='whitegrid', palette='Set2')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14
})


def plot_distribution(df):
    """
    Plot distributions for key numeric variables.
    """
    columns = [
        "Average Duration of Power Outages (Minutes)",
        "Average Duration of Power Outages (Hours)",
        "Electricity Consumption in Typical Month (kWh)",
        "Sales Revenue"
    ]
    for col in columns:
        if col in df.columns:
            plt.figure()
            plt.hist(df[col].dropna(), bins=30, edgecolor='black')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()


def plot_scatter_outage_vs_revenue(df):
    """
    Scatter plot of average outage duration (hours) vs. sales revenue with regression line.
    """
    if "Average Duration of Power Outages (Hours)" in df.columns and "Sales Revenue" in df.columns:
        corr_value = df["Average Duration of Power Outages (Hours)"].corr(df["Sales Revenue"])
        plt.figure()
        sns.scatterplot(
            x="Average Duration of Power Outages (Hours)",
            y="Sales Revenue",
            data=df,
            alpha=0.7
        )
        sns.regplot(
            x="Average Duration of Power Outages (Hours)",
            y="Sales Revenue",
            data=df,
            scatter=False,
            color="red"
        )
        plt.title(f"Avg. Outage Duration vs Sales Revenue\nCorrelation: {corr_value:.2f}")
        plt.xlabel("Average Duration of Power Outages (Hours)")
        plt.ylabel("Sales Revenue")
        plt.tight_layout()
        plt.show()


def plot_country_sales(df):
    """
    Bar plot of country-wise total sales revenue.
    """
    if "Country" in df.columns and "Sales Revenue" in df.columns:
        plt.figure()
        country_sales = df.groupby("Country")["Sales Revenue"].sum().sort_values(ascending=False)
        sns.barplot(x=country_sales.index, y=country_sales.values)
        plt.title("Country-wise Total Sales Revenue")
        plt.xlabel("Country")
        plt.ylabel("Total Sales Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_year_sales(df):
    """
    Bar plot of year-wise total sales revenue.
    """
    if "Year" in df.columns and "Sales Revenue" in df.columns:
        plt.figure()
        year_sales = df.groupby("Year")["Sales Revenue"].sum().sort_index()
        sns.barplot(x=year_sales.index.astype(str), y=year_sales.values)
        plt.title("Year-wise Total Sales Revenue")
        plt.xlabel("Year")
        plt.ylabel("Total Sales Revenue")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


def plot_firm_size_sales(df):
    """
    Bar plot of average sales revenue by firm size category.
    """
    if "Firm Size Category" in df.columns and "Sales Revenue" in df.columns:
        plt.figure()
        firm_size_sales = df.groupby("Firm Size Category")["Sales Revenue"].mean().reindex(["Small", "Large"])
        sns.barplot(x=firm_size_sales.index, y=firm_size_sales.values)
        plt.title("Sales Revenue by Firm Size Category")
        plt.xlabel("Firm Size Category")
        plt.ylabel("Average Sales Revenue")
        plt.tight_layout()
        plt.show()


def plot_power_outages_last_fy(df):
    """
    Bar plot: Sales revenue by power outages experienced in last FY.
    """
    if "Power Outages Experienced in Last FY" in df.columns and "Sales Revenue" in df.columns:
        plt.figure()
        outages_vs_sales = df.groupby("Power Outages Experienced in Last FY")["Sales Revenue"].sum().sort_index()
        sns.barplot(x=outages_vs_sales.index.astype(str), y=outages_vs_sales.values)
        plt.title("Sales Revenue by Outages Experienced in Last FY")
        plt.xlabel("Power Outages Experienced in Last FY")
        plt.ylabel("Total Sales Revenue")
        plt.tight_layout()
        plt.show()


def plot_backup_power_usage(df):
    """
    Bar plot: Sales revenue by backup power usage.
    """
    if "Backup Power Usage (Own/Shared Generator)" in df.columns and "Sales Revenue" in df.columns:
        plt.figure()
        filtered_df = df[df["Backup Power Usage (Own/Shared Generator)"] != -9]
        backup_power_vs_sales = filtered_df.groupby("Backup Power Usage (Own/Shared Generator)")[
            "Sales Revenue"].sum().sort_index()
        sns.barplot(x=backup_power_vs_sales.index.astype(str), y=backup_power_vs_sales.values)
        plt.title("Sales Revenue by Backup Power Usage")
        plt.xlabel("Backup Power Usage (Own/Shared Generator)")
        plt.ylabel("Total Sales Revenue")
        plt.tight_layout()
        plt.show()


def plot_electricity_consumption_vs_revenue(df):
    """
    Scatter plot: Electricity consumption vs. sales revenue.
    """
    if "Electricity Consumption in Typical Month (kWh)" in df.columns and "Sales Revenue" in df.columns:
        plt.figure()
        sns.scatterplot(
            x="Electricity Consumption in Typical Month (kWh)",
            y="Sales Revenue",
            data=df,
            alpha=0.7
        )
        sns.regplot(
            x="Electricity Consumption in Typical Month (kWh)",
            y="Sales Revenue",
            data=df,
            scatter=False,
            color="red"
        )
        plt.title("Electricity Consumption vs Sales Revenue")
        plt.xlabel("Electricity Consumption in Typical Month (kWh)")
        plt.ylabel("Sales Revenue")
        plt.tight_layout()
        plt.show()


def plot_domestic_ownership_vs_sales_revenue_per_employee(df):
    """
    Scatter plot: % Owned by Private Domestic Individuals vs. Sales Revenue per Employee.
    Marker size represents overall Sales Revenue and colors denote Country.
    """
    if ("% Owned by Private Domestic Individuals" in df.columns and
            "Sales Revenue per Employee" in df.columns and
            "Sales Revenue" in df.columns and
            "Country" in df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="% Owned by Private Domestic Individuals",
            y="Sales Revenue per Employee",
            hue="Country",
            size="Sales Revenue",
            sizes=(20, 200),
            alpha=0.7,
            palette="viridis"
        )
        plt.title("Domestic Ownership vs. Sales Revenue per Employee")
        plt.xlabel("% Owned by Private Domestic Individuals")
        plt.ylabel("Sales Revenue per Employee")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Country")
        plt.tight_layout()
        plt.show()


def plot_working_capital_financing_by_firm_age(df):
    """
    Stacked bar plot: Average working capital financing breakdown by firm age category.
    """
    financing_cols = [
        '% of Working Capital Financed from Internal Funds',
        '% of Working Capital Borrowed from Banks',
        '% of Working Capital Borrowed from Non-Bank Financial Institutions',
        '% of Working Capital Purchased on Credit/Advances',
        '% of Working Capital Financed by Other (Money Lenders, Friends, Relatives)'
    ]
    if "Firm Age Category" in df.columns and all(col in df.columns for col in financing_cols):
        financing_data = df.groupby("Firm Age Category")[financing_cols].mean()
        plt.figure(figsize=(10, 6))
        financing_data.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))
        plt.title("Average Working Capital Financing Breakdown by Firm Age Category")
        plt.xlabel("Firm Age Category")
        plt.ylabel("Average Percentage")
        plt.legend(bbox_to_anchor=(1.05, 1), title="Financing Type")
        plt.tight_layout()
        plt.show()


def plot_electricity_dependency_vs_outage_impact(df):
    """
    Scatter plot: Electricity Dependency Ratio vs. Power Outage Impact Score.
    """
    if ("Electricity Dependency Ratio" in df.columns and
            "Power Outage Impact Score" in df.columns and
            "Country" in df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="Electricity Dependency Ratio",
            y="Power Outage Impact Score",
            hue="Country",
            palette="coolwarm",
            alpha=0.7
        )
        plt.title("Electricity Dependency Ratio vs. Power Outage Impact Score")
        plt.xlabel("Electricity Dependency Ratio")
        plt.ylabel("Power Outage Impact Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def plot_backup_power_dependency_by_firm_size(df):
    """
    Box plot: Backup Power Dependency by Firm Size Category.
    """
    if "Firm Size Category" in df.columns and "Backup Power Dependency" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=df,
            x="Firm Size Category",
            y="Backup Power Dependency",
            palette="Set2"
        )
        plt.title("Backup Power Dependency by Firm Size Category")
        plt.xlabel("Firm Size Category")
        plt.ylabel("Backup Power Dependency")
        plt.tight_layout()
        plt.show()


def plot_firm_age_vs_sales_revenue_per_employee(df):
    """
    Scatter plot: Firm Age vs. Sales Revenue per Employee colored by Firm Age Category.
    """
    if ("Firm Age (Years Since Establishment)" in df.columns and
            "Sales Revenue per Employee" in df.columns and
            "Firm Age Category" in df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="Firm Age (Years Since Establishment)",
            y="Sales Revenue per Employee",
            hue="Firm Age Category",
            palette="coolwarm",
            alpha=0.7
        )
        plt.title("Firm Age vs. Sales Revenue per Employee")
        plt.xlabel("Firm Age (Years Since Establishment)")
        plt.ylabel("Sales Revenue per Employee")
        plt.legend(title="Firm Age Category", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df):
    """
    Heatmap: Correlation matrix for selected financial and ownership variables.
    """
    corr_cols = [
        '% Owned by Private Domestic Individuals',
        '% Owned by Private Foreign Individuals',
        '% of Working Capital Financed from Internal Funds',
        '% of Working Capital Borrowed from Banks',
        '% of Working Capital Borrowed from Non-Bank Financial Institutions',
        '% of Working Capital Purchased on Credit/Advances',
        '% of Working Capital Financed by Other (Money Lenders, Friends, Relatives)',
        'Working Capital Dependency',
        'Firm Size (Full-Time Employees)',
        'Sales Revenue per Employee',
        'Sales Revenue',
        'Backup Power Usage (Own/Shared Generator)',
        'Average Duration of Power Outages (Minutes)',
        'Power Outages Experienced in Last FY',
        'Number of Power Outages per Month'
    ]
    # Only use columns that exist in the dataframe
    existing_corr_cols = [col for col in corr_cols if col in df.columns]
    if existing_corr_cols:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[existing_corr_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.75,
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.8},
            square=True
        )
        plt.title("Correlation Matrix for Financial Variables", fontsize=16, weight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()


def plot_kde_continuous(df):
    """
    Generate KDE plots for all continuous (numeric) variables with >10 unique values.
    """
    continuous_vars = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if df[col].nunique() > 10]
    for col in continuous_vars:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(df[col].dropna(), fill=True)
        plt.title(f'KDE Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()


def plot_kde_sales_revenue_by_categorical(df):
    """
    Generate KDE plots of Sales Revenue for each category within each categorical variable.
    """
    # Identify categorical variables: object type and numeric columns with few unique values (excluding Sales Revenue)
    categorical_vars = list(df.select_dtypes(include=['object']).columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].nunique() <= 10 and col != "Sales Revenue":
            categorical_vars.append(col)
    categorical_vars = list(set(categorical_vars))

    if "Sales Revenue" not in df.columns:
        print("Sales Revenue column not found.")
        return
    for cat in categorical_vars:
        plt.figure(figsize=(10, 6))
        groups = sorted(df[cat].dropna().unique())
        for group in groups:
            subset = df[df[cat] == group]
            sns.kdeplot(subset["Sales Revenue"].dropna(), label=str(group), fill=True, alpha=0.3)
        plt.title(f'KDE Plot of Sales Revenue by {cat}')
        plt.xlabel("Sales Revenue")
        plt.ylabel("Density")
        plt.legend(title=cat)
        plt.tight_layout()
        plt.show()


def generate_all_plots(df):
    """
    Master function to generate all visualizations.
    """
    # plot_distribution(df)
    # plot_scatter_outage_vs_revenue(df)
    # plot_country_sales(df)
    # plot_year_sales(df)
    # plot_firm_size_sales(df)
    # plot_power_outages_last_fy(df)
    # plot_backup_power_usage(df)
    # plot_electricity_consumption_vs_revenue(df)
    # plot_domestic_ownership_vs_sales_revenue_per_employee(df)
    # plot_working_capital_financing_by_firm_age(df)
    # plot_electricity_dependency_vs_outage_impact(df)
    # plot_backup_power_dependency_by_firm_size(df)
    # plot_firm_age_vs_sales_revenue_per_employee(df)
    plot_correlation_heatmap(df)
    plot_kde_continuous(df)
    plot_kde_sales_revenue_by_categorical(df)


# Optional: run all plots when executed directly
if __name__ == "__main__":
    # For testing purposes, load a sample dataframe (update file_path as necessary)
    file_path = r"/content/all_merged_data.csv"  # Update to your CSV file location
    df = pd.read_csv(file_path)
    generate_all_plots(df)
