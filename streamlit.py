# app.py
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "StudentPerformanceFactors.csv"
TARGET_COL = "Exam_Score"  # falls back to last column if not found

st.set_page_config(page_title="Student Performance", layout="wide")

# -----------------------------
# LEFT NAV (6 tabs)
# -----------------------------
tabs = ["Welcome", "About", "Explore", "Visualizations", "Predictions", "Conclusion"]
selected = st.sidebar.radio("", tabs, index=4)

# Empty tabs (for now)
if selected != "Predictions":
    st.empty()
    st.stop()

# -----------------------------
# PREDICTIONS TAB
# -----------------------------
# Load dataset (hardcoded)
df = pd.read_csv(DATA_PATH)

# Target column fallback
if TARGET_COL not in df.columns:
    TARGET_COL = df.columns[-1]

# UI controls (only what you requested)
model_name = st.selectbox("Model", ["Linear Regression", "Ridge", "Lasso"])
test_size_pct = st.slider("Test size (%)", min_value=15, max_value=25, value=20, step=1)
add_engagement = st.checkbox("Add engagement feature")

train_clicked = st.button("Train")

if train_clicked:
    # Prepare X, y
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # Add engagement feature (only if source columns exist)
    if add_engagement:
        X["Engagement"] = X["Attendance"] * X["Hours_Studied"]

    # Identify feature types
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocess: impute + encode categoricals, impute + scale numerics
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # Model choice (3 models)
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0, random_state=42)
    else:
        model = Lasso(alpha=0.1, random_state=42, max_iter=10000)

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Split
    test_size = test_size_pct / 100.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train + evaluate
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred)) ** 0.5

    st.write(f"R2: {r2:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"RMSE: {rmse:.4f}")