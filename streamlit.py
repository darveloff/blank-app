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
# CONFIG & STYLING
# -----------------------------
DATA_PATH = "StudentPerformanceFactors.csv"
TARGET_COL = "Exam_Score"  # falls back to last column if not found

st.set_page_config(page_title="Student Performance", layout="wide", page_icon="🎓")

# Set seaborn theme for more pleasing visuals
sns.set_theme(style="whitegrid", palette="muted")

# -----------------------------
# LEFT NAV (6 tabs)
# -----------------------------
with st.sidebar:
    st.title("🎓 Student Dashboard")
    st.markdown("Navigate through the analysis pipeline.")
    tabs = ["Welcome", "About", "Explore", "Visualizations", "Predictions", "Conclusion"]
    selected = st.radio("Navigation", tabs, index=3)

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
    st.info(f"The {selected} page is currently under construction. 🚧")
    st.stop()

# -----------------------------
# VISUALIZATIONS TAB
# -----------------------------
if selected == "Visualizations":
    st.title("Data Visualizations 📊")
    st.markdown("Explore various aspects of the student performance dataset through the tabs below.")

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

    # 2. Create Sub-tabs for better UI
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Correlation 🔗", 
        "Distributions 📊", 
        "Relationships 📈", 
        "Categorical Impacts 📦"
    ])

    # --- SUB-TAB 1: Correlation ---
    with viz_tab1:
        st.subheader("Feature Correlation Matrix")
        st.markdown("This heatmap shows the linear relationships between encoded features.")
        corr = df_corr.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, 
                    cbar_kws={'label': 'Correlation Coefficient'})
        plt.title("Correlation Matrix")
        st.pyplot(fig_corr)

    # --- SUB-TAB 2: Distributions ---
    with viz_tab2:
        st.subheader("Target Variable Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["Exam_Score"], kde=True, ax=ax, color="skyblue")
            plt.title("Distribution of Exam Score")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df["Exam_Score"], ax=ax, color="lightgreen")
            plt.title("Exam Score Boxplot")
            st.pyplot(fig)

    # --- SUB-TAB 3: Relationships ---
    with viz_tab3:
        st.subheader("Numerical Features vs. Target")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Attendance", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Attendance vs Exam Score")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Previous_Scores", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Previous Scores vs Exam Score")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Hours_Studied", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Hours Studied vs Exam Score")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="Hours_Studied", y="Exam_Score", hue="Attendance", data=df, ax=ax, palette="viridis", alpha=0.7)
            plt.title("Study Hours by Attendance")
            st.pyplot(fig)

    # --- SUB-TAB 4: Categorical Impacts ---
    with viz_tab4:
        st.subheader("Impact of Categorical Variables on Exam Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Parental_Involvement", y="Exam_Score", data=df, ax=ax, palette="Set2")
            plt.title("Parental Involvement Impact")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Tutoring_Sessions", y="Exam_Score", data=df, ax=ax, palette="Set2", errorbar=None)
            plt.title("Tutoring Sessions Impact")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Access_to_Resources", y="Exam_Score", data=df, ax=ax, palette="Set2")
            plt.title("Access to Resources Impact")
            st.pyplot(fig)

# -----------------------------
# PREDICTIONS TAB
# -----------------------------
if selected == "Predictions":
    st.title("Model Predictions 🧠")
    st.markdown("Configure hyperparameters, add engineered features, and evaluate regression models.")

    with st.container(border=True):
        st.subheader("Model Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_name = st.selectbox("Select Model", ["Linear Regression", "Ridge", "Lasso"])
        with col2:
            test_size_pct = st.slider("Test Set Size (%)", min_value=15, max_value=25, value=20, step=1)
        with col3:
            st.markdown("<br>", unsafe_allow_html=True) # Spacing alignment
            add_engagement = st.checkbox("Add 'Engagement' Feature", help="Multiplies Attendance by Hours_Studied")

        train_clicked = st.button("🚀 Train Model", type="primary", use_container_width=True)

    if train_clicked:
        with st.spinner(f"Training {model_name}..."):
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

            st.success(f"🎉 **{model_name}** model trained successfully!")
            
            # Display metrics in a visually pleasing way
            st.markdown("### Model Performance Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("Mean Absolute Error", f"{mae:.4f}")
            m3.metric("Root Mean Squared Error", f"{rmse:.4f}")