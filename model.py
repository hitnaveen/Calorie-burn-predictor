"""
model.py
Data loading, preprocessing, and model training for the
Calories Burnt Prediction project.

All file paths are resolved relative to this file's own location
(not the process's working directory), which is what actually
fixes the FileNotFoundError seen on Streamlit Community Cloud —
Streamlit Cloud can launch the app from /mount/src/<repo>, and a
plain relative path like "data/calories.csv" only works if the
process happens to be started from the repo root.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
TARGET = "Calories"


def load_data(path: str = "data/calories.csv") -> pd.DataFrame:
    """Load the calories dataset. `path` is resolved relative to this file,
    so it works no matter what directory the app is launched from."""
    full_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Could not find dataset at '{full_path}'. "
            f"Make sure 'data/calories.csv' is committed to the repository "
            f"(check it isn't listed in .gitignore or stored via Git LFS)."
        )
    df = pd.read_csv(full_path)
    df.columns = df.columns.str.strip()
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the raw dataset."""
    df = df.drop_duplicates().copy()
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map({"male": 0, "female": 1})
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def get_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 22):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s, y_train, y_val, scaler


def train_all_models(df: pd.DataFrame):
    """Train the full model comparison suite used in the report
    (Linear Regression, Ridge, Lasso, Random Forest, XGBoost) and
    return the fitted models, the scaler, and a metrics table."""
    X_train_s, X_val_s, y_train, y_val, scaler = get_train_test(df)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=22, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=22,
            n_jobs=-1, verbosity=0,
        ),
    }

    fitted = {}
    rows = []
    for name, m in candidates.items():
        m.fit(X_train_s, y_train)
        train_pred = m.predict(X_train_s)
        val_pred = m.predict(X_val_s)
        rows.append({
            "Model": name,
            "Train MAE": mean_absolute_error(y_train, train_pred),
            "Val MAE": mean_absolute_error(y_val, val_pred),
            "Val RMSE": np.sqrt(mean_squared_error(y_val, val_pred)),
            "Val R2": r2_score(y_val, val_pred),
        })
        fitted[name] = m

    metrics_df = pd.DataFrame(rows).sort_values("Val R2", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_model = fitted[best_name]

    return {
        "models": fitted,
        "metrics": metrics_df,
        "scaler": scaler,
        "best_name": best_name,
        "best_model": best_model,
        "X_val": X_val_s,
        "y_val": y_val,
    }


def predict_calories(model, scaler, gender, age, height, weight, duration, heart_rate, body_temp):
    """Run a single prediction for the given user inputs."""
    gender_code = 0 if str(gender).lower().startswith("m") else 1
    row = pd.DataFrame([{
        "Gender": gender_code,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }])[FEATURES]
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    return max(float(pred), 0.0)


def feature_importance(model, feature_names=FEATURES):
    """Return a sorted feature-importance DataFrame if the model supports it."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None
    fi = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    fi["Importance"] = fi["Importance"] / fi["Importance"].sum()
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)
