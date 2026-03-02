import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
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
selected = st.sidebar.radio("Navigation", tabs, index=3)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
    # Target column fallback
    if TARGET_COL not in df.columns:
        TARGET_COL = df.columns[-1]
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'StudentPerformanceFactors.csv' is in the directory.")
    st.stop()

# Empty tabs (for now)
if selected not in ["Visualizations", "Predictions"]:
    st.title(selected)
    st.info(f"The {selected} page is currently under construction.")
    st.stop()

# -----------------------------
# VISUALIZATIONS TAB
# -----------------------------
if selected == "Visualizations":
    st.title("Data Visualizations 📊")
    st.markdown("All visualizations derived exactly from the Google Colab Notebook.")

    # 1. Preprocess data specifically for the Correlation Matrix (Matches Colab)
    df_corr = df.copy()
    
    ordinal_cols = [
        "Parental_Involvement", "Access_to_Resources", "Motivation_Level",
        "Family_Income", "Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"
    ]

    # normalize strings (keeps NaN as NaN)
    df_corr[ordinal_cols] = df_corr[ordinal_cols].apply(lambda s: s.astype("string").str.strip().str.lower())

    # impute missing values (most frequent category in each column)
    for col in ordinal_cols:
        df_corr[col] = df_corr[col].fillna(df_corr[col].mode(dropna=True)[0])

    encoder = OrdinalEncoder(categories=[
        ["low", "medium", "high"],          # Parental_Involvement
        ["low", "medium", "high"],          # Access_to_Resources
        ["low", "medium", "high"],          # Motivation_Level
        ["low", "medium", "high"],          # Family_Income
        ["low", "medium", "high"],          # Teacher_Quality
        ["high school", "college", "postgraduate"],
        ["near", "moderate", "far"]
    ])
    df_corr[ordinal_cols] = encoder.fit_transform(df_corr[ordinal_cols])

    binary_maps = {
        "Extracurricular_Activities": {"No": 0, "Yes": 1},
        "Internet_Access": {"No": 0, "Yes": 1},
        "School_Type": {"Public": 0, "Private": 1},
        "Peer_Influence": {"Negative": 0, "Positive": 1},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "Gender": {"Female": 0, "Male": 1},
    }

    for col, mp in binary_maps.items():
        df_corr[col] = df_corr[col].astype("string").str.strip()
        df_corr[col] = df_corr[col].map(mp)

    df_corr["Peer_Influence"] = df_corr["Peer_Influence"].fillna(-1)

    # 2. Render the Correlation Matrix
    st.markdown("### Correlation Matrix")
    corr = df_corr.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    plt.title("Correlation Matrix")
    st.pyplot(fig_corr)
    
    st.divider()

    # 3. Render all other plots in a clean 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribution of Exam Score")
        fig, ax = plt.subplots()
        sns.histplot(df["Exam_Score"], kde=True, ax=ax)
        plt.title("Distribution of Exam Score")
        st.pyplot(fig)

        st.markdown("### Attendance vs Exam Score")
        fig, ax = plt.subplots()
        sns.regplot(x="Attendance", y="Exam_Score", data=df, ax=ax)
        plt.title("Attendance vs Exam Score")
        st.pyplot(fig)

        st.markdown("### Previous Scores vs Exam Score")
        fig, ax = plt.subplots()
        sns.regplot(x="Previous_Scores", y="Exam_Score", data=df, ax=ax)
        plt.title("Previous Scores vs Exam Score")
        st.pyplot(fig)

        st.markdown("### Access to Resources Impact")
        fig, ax = plt.subplots()
        sns.boxplot(x="Access_to_Resources", y="Exam_Score", data=df, ax=ax)
        plt.title("Access to Resources Impact")
        st.pyplot(fig)

        st.markdown("### Study Hours by Attendance")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Hours_Studied", y="Exam_Score", hue="Attendance", data=df, ax=ax)
        plt.title("Study Hours by Attendance")
        st.pyplot(fig)

    with col2:
        st.markdown("### Exam Score Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Exam_Score"], ax=ax)
        plt.title("Exam Score Boxplot")
        st.pyplot(fig)

        st.markdown("### Hours Studied vs Exam Score")
        fig, ax = plt.subplots()
        sns.regplot(x="Hours_Studied", y="Exam_Score", data=df, ax=ax)
        plt.title("Hours Studied vs Exam Score")
        st.pyplot(fig)

        st.markdown("### Parental Involvement Impact")
        fig, ax = plt.subplots()
        sns.boxplot(x="Parental_Involvement", y="Exam_Score", data=df, ax=ax)
        plt.title("Parental Involvement Impact")
        st.pyplot(fig)

        st.markdown("### Tutoring Sessions Impact")
        fig, ax = plt.subplots()
        sns.barplot(x="Tutoring_Sessions", y="Exam_Score", data=df, ax=ax)
        plt.title("Tutoring Sessions Impact")
        st.pyplot(fig)

# -----------------------------
# PREDICTIONS TAB
# -----------------------------
if selected == "Predictions":
    st.title("Predictions 🧠")

    # UI controls (only what you requested)
    model_name = st.selectbox("Model", ["Linear Regression", "Ridge", "Lasso"])
    test_size_pct = st.slider("Test size (%)", min_value=15, max_value=25, value=20, step=1)
    add_engagement = st.checkbox("Add engagement feature")

    train_clicked = st.button("Train")

    if train_clicked:
        with st.spinner("Training model..."):
            # Prepare X, y
            X = df.drop(columns=[TARGET_COL]).copy()
            y = df[TARGET_COL].copy()

            # Add engagement feature (only if source columns exist)
            if add_engagement and "Attendance" in X.columns and "Hours_Studied" in X.columns:
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

            st.success(f"{model_name} trained successfully!")
            st.write(f"**R2:** {r2:.4f}")
            st.write(f"**MAE:** {mae:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")