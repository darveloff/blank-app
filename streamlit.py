import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# Step 00 - Page Config
# ==========================================
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="centered",
    page_icon="🎓",
)

# ==========================================
# Step 01 - Data Loading & Preprocessing
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("StudentPerformanceFactors.csv")
    
    # 1. Ordinal Encoding
    ordinal_cols = [
        "Parental_Involvement", "Access_to_Resources", "Motivation_Level",
        "Family_Income", "Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"
    ]
    
    # Normalize strings (keeps NaN as NaN)
    df[ordinal_cols] = df[ordinal_cols].apply(lambda s: s.astype("string").str.strip().str.lower())
    
    # Impute missing values (most frequent category)
    for col in ordinal_cols:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
        
    encoder = OrdinalEncoder(categories=[
        ["low", "medium", "high"],          # Parental_Involvement
        ["low", "medium", "high"],          # Access_to_Resources
        ["low", "medium", "high"],          # Motivation_Level
        ["low", "medium", "high"],          # Family_Income
        ["low", "medium", "high"],          # Teacher_Quality
        ["high school", "college", "postgraduate"], # Parental_Education_Level
        ["near", "moderate", "far"]         # Distance_from_Home
    ])
    df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])
    
    # 2. Binary Encoding
    binary_maps = {
        "Extracurricular_Activities": {"No": 0, "Yes": 1},
        "Internet_Access": {"No": 0, "Yes": 1},
        "School_Type": {"Public": 0, "Private": 1},
        "Peer_Influence": {"Negative": 0, "Positive": 1},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "Gender": {"Female": 0, "Male": 1},
    }
    
    for col, mp in binary_maps.items():
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].map(mp)
        
    # 3. Handle Remaining Missing Values
    df["Peer_Influence"] = df["Peer_Influence"].fillna(-1)
    
    # Drop rows where target might be null (just in case)
    df = df.dropna(subset=['Exam_Score'])
    
    return df

df = load_and_preprocess_data()

# ==========================================
# Step 02 - Setup Sidebar & Navigation
# ==========================================
st.sidebar.title("Student Performance 🎓")
page = st.sidebar.selectbox("Select Page", ["Introduction 📘", "Visualization 📊", "Prediction 🧠"])

# ==========================================
# Step 03 - Pages Logic
# ==========================================
if page == "Introduction 📘":
    st.subheader("01 Introduction 📘")

    st.markdown("##### Data Preview (Preprocessed)")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "0 missing values")

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":
    st.subheader("02 Data Viz 📊")

    col_x = st.selectbox("Select X-axis variable", df.columns, index=0)
    col_y = st.selectbox("Select Y-axis variable", df.columns, index=df.columns.get_loc("Exam_Score"))

    tab1, tab2, tab3 = st.tabs(["Bar Chart 📊", "Line Chart 📈", "Correlation Heatmap 🔥"])

    with tab1:
        st.subheader("Bar Chart")
        st.bar_chart(df[[col_x, col_y]].sort_values(by=col_x).set_index(col_x), use_container_width=True)

    with tab2:
        st.subheader("Line Chart")
        st.line_chart(df[[col_x, col_y]].sort_values(by=col_x).set_index(col_x), use_container_width=True)

    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr, annot_kws={"size": 8})
        st.pyplot(fig_corr)

elif page == "Prediction 🧠":
    st.subheader("03 Prediction 🧠")
    
    # Create the requested tabs matching the screenshots
    tab1, tab2, tab3 = st.tabs(["Linear Regression 📈", "Ridge ⛰️", "Lasso 🤠"])
    
    # Reusable function to train and display model results inside a specific tab
    def render_model_ui(model_name, tab_context):
        with tab_context:
            st.markdown(f"### {model_name} Model")
            
            # Interactive widgets with unique keys per tab
            test_size_pct = st.slider("Test Set Size (%)", 15, 25, 15, key=f"slider_{model_name}")
            add_engagement = st.checkbox("Add Engagement Feature", key=f"checkbox_{model_name}")
            
            # Prepare data
            model_df = df.copy()
            if add_engagement:
                model_df["Engagement"] = model_df["Attendance"] * model_df["Hours_Studied"]
                
            X = model_df.drop(columns=["Exam_Score"])
            y = model_df["Exam_Score"]
            
            # Train-Test Split
            test_size_ratio = test_size_pct / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)
            
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Initialize Model
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Ridge":
                model = Ridge(alpha=1.0)
            elif model_name == "Lasso":
                model = Lasso(alpha=0.01)
                
            # Train and Predict
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Display Metrics beautifully
            st.markdown("##### Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="R² Score", value=f"{r2:.4f}")
            col2.metric(label="MAE", value=f"{mae:.4f}")
            col3.metric(label="RMSE", value=f"{rmse:.4f}")

    # Render UI for each tab
    render_model_ui("Linear Regression", tab1)
    render_model_ui("Ridge", tab2)
    render_model_ui("Lasso", tab3)