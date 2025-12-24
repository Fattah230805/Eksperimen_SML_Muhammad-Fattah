# =========================
# 1. IMPORT LIBRARY
# =========================
import os
import dagshub
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error
)


# =========================
# 2. INIT DAGSHUB (ONLINE MLFLOW)
# =========================
dagshub.init(
    repo_owner="Fattah230805",
    repo_name="Eksperimen_SML_Muhammad-Fattah",
    mlflow=True
)

mlflow.set_experiment("HousePrice-RF-Tuning-DagsHub")


# =========================
# 3. LOAD DATA (PREPROCESSED)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "kriteria_2",
    "data",
    "preprocess.csv"
)

print("Loading dataset from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# 4. HYPERPARAMETER GRID
# =========================
n_estimators_list = [50, 100]
max_depth_list = [5, 10, None]


# =========================
# 5. TUNING + MANUAL LOGGING
# =========================
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        with mlflow.start_run():

            # ----- Model -----
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            # ----- Prediction -----
            y_pred = model.predict(X_test)

            # ----- Metrics (VERSI BARU) -----
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # ----- Log Params -----
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # ----- Log Metrics -----
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # =========================
            # 6. EXTRA ARTEFAK (ADVANCE)
            # =========================

            os.makedirs("artifacts", exist_ok=True)

            # (1) Feature Importance
            plt.figure(figsize=(8, 5))
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]

            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), X.columns[indices])
            plt.title("Top 10 Feature Importances")
            plt.tight_layout()

            fi_path = "artifacts/feature_importance.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)

            # (2) Prediction vs Actual
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            plt.title("Actual vs Predicted")
            plt.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "--r"
            )
            plt.tight_layout()

            pv_path = "artifacts/prediction_vs_actual.png"
            plt.savefig(pv_path)
            plt.close()
            mlflow.log_artifact(pv_path)

            # ----- Log Model -----
            mlflow.sklearn.log_model(
                model,
                name="random_forest_model"
            )

            print(
                f"Run | n_estimators={n_estimators}, "
                f"max_depth={max_depth}, "
                f"RMSE={rmse:.2f}, R2={r2:.4f}"
            )


print("=== Hyperparameter tuning (ADVANCE - DAGSHUB) selesai ===")
