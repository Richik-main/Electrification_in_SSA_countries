# EDA and Model Training Project

This repository contains a modular pipeline for:
1. **Data Preprocessing** (cleaning, feature engineering)
2. **Exploratory Data Analysis** (visualizations)
3. **Model Training** (hyperparameter tuning and performance evaluation)

## Project Structure


your-project/
├── config.yaml
├── data/
│   └── all_merged_data.csv
├── main.py
├── README.md
├── requirements.txt
└── scripts/
    ├── data_preprocessing.py
    ├── visualization.py
    └── model_training.py

- **config.yaml**: Central place for file paths, columns to drop, invalid placeholder values, and model hyperparameters.
- **requirements.txt**: Python dependencies for this project.
- **scripts/**:
  - **data_preprocessing.py**: Contains functions to load data, clean, impute, and engineer features.
  - **visualization.py**: Contains plotting functions for EDA.
  - **model_training.py**: Contains model building, hyperparameter tuning, evaluation.
- **main.py**: Orchestrates the entire workflow by calling scripts in sequence.

## Getting Started

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YourUsername/your-project.git
   cd your-project
