# scripts/model_training.py

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
XGBRegressor.__sklearn_tags__ = lambda self: {"non_deterministic": True}

def train_and_evaluate_models(df, config):
    """
    Train models (RandomForest, XGBoost, etc.) using hyperparameter grids from config.
    Perform cross-validation and evaluate on test data.
    """
    # -----------------------------
    # 1. Extract config items
    # -----------------------------
    target_column = config['target_column']
    test_size = config['test_size']
    random_state = config['random_state']
    model_params = config['model_params']

    # -----------------------------
    # 2. Prepare data
    # -----------------------------
    # Drop Sales Revenue per Employee if it exists
    #if 'Sales Revenue per Employee' in df.columns:
    df = df.drop(columns=['Sales Revenue per Employee'], errors='ignore')
    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle infinite values
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    print('Training model... SHAPE OF THE DATA')
    print(df.shape)
    # Re-impute missing values with KNN
    knn_imputer = KNNImputer(n_neighbors=config['imputation_neighbors'])
    X_train = knn_imputer.fit_transform(X_train)
    X_test = knn_imputer.transform(X_test)

    # -----------------------------
    # 3. Define models to train
    # -----------------------------
    # Changed key from "Random Forest" to "RandomForest" for consistency with YAML
    models = {
        #"RandomForest": RandomForestRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(random_state=random_state, objective='reg:squarederror')
    }
    models["XGBoost"]._estimator_type = "regressor"

    # -----------------------------
    # 4. Hyperparameter Tuning / Default Fit
    # -----------------------------
    results = []
    best_params = {}
    best_models = {}

    for name, model in models.items():
        # Get hyperparameter grid; if not defined, use an empty dict
        param_dist = model_params.get(name, {})

        if param_dist:
            print(f"\nRunning hyperparameter tuning for model: {name}")
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=20,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=random_state,
                return_train_score=True
            )
            random_search.fit(X_train, y_train)
            best_models[name] = random_search.best_estimator_
            best_params[name] = {
                "Best Hyperparameters": random_search.best_params_,
                "Best R^2 Score": random_search.best_score_
            }
            # Save each iteration's performance
            for i, params in enumerate(random_search.cv_results_['params']):
                train_score = random_search.cv_results_['mean_train_score'][i]
                test_score = random_search.cv_results_['mean_test_score'][i]
                results.append((name, params, train_score, test_score))
        else:
            print(f"\nNo hyperparameter grid found for {name}. Fitting with default parameters.")
            model.fit(X_train, y_train)
            best_models[name] = model
            best_params[name] = {
                "Best Hyperparameters": "Default parameters",
                "Best R^2 Score": model.score(X_train, y_train)
            }
            # Record a single performance entry using default model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            results.append((name, "Default parameters", train_score, test_score))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Model', 'Hyperparameters', 'Train R^2', 'Test R^2'])

    print("\nHyperparameter Tuning / Default Fit Results:")
    print(results_df)

    print("\nBest Hyperparameters for Each Model:")
    for model_name, details in best_params.items():
        print(f"\nModel: {model_name}")
        print(f"Best Hyperparameters: {details['Best Hyperparameters']}")
        print(f"Best R^2 Score: {details['Best R^2 Score']:.4f}")

    # -----------------------------
    # 5. Evaluate best models
    # -----------------------------
    print("\nEvaluation Metrics on Train/Test for Best Models:")
    for name, model in best_models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # RMSE
        train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))

        # MAE
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # MAPE
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

        print(f"\nModel: {name}")
        print(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Train MAPE: {train_mape:.2f}%")
        print(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.2f}%")

    # -----------------------------
    # 6. Optional: Plot Cross Validation Performance
    # -----------------------------
    for model_name in results_df['Model'].unique():
        subset = results_df[results_df['Model'] == model_name].reset_index(drop=True)
        plt.figure(figsize=(8, 6))
        plt.plot(subset.index, subset['Train R^2'], marker='o', linestyle='-', label='Train R^2')
        plt.plot(subset.index, subset['Test R^2'], marker='o', linestyle='-', label='Test R^2')
        plt.xlabel('Iteration')
        plt.ylabel('R^2 Score')
        plt.title(f'Cross Validation Performance for {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return best_models, results_df
