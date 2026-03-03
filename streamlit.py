import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from PIL import Image
from streamlit_option_menu import option_menu

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
st.sidebar.title("🗺️ Navigation")
selected = st.sidebar.selectbox("",["👋 Welcome", "🔎 Business Case", "👩‍💻 Data Presentation", "📊 Data Viz", "🧠 Regressions", "🎬 Conclusion"])

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

# -----------------------------
# WELCOME TAB
# -----------------------------
if selected == "👋 Welcome":
    st.title("🎓 CollegeBoard: Student Performance") 
    st.image("logo.png", use_container_width=True)
    st.markdown(
        """
        ---
        ## 👋 Welcome!
        This interactive dashboard allows you to explore factors affecting student exam scores, visualize relationships, and make predictions using regression models.

        Key objectives of this platform:
        - Explore patterns in student performance and attendance.
        - Visualize relationships between study habits, demographics, and exam scores.
        - Train regression models to predict exam performance.
        - Understand the impact of categorical and numerical features on outcomes.
        """
    )

    # Team / Contributors
    st.subheader("👥 Meet the Team")
    st.write("- Amina Magomedova")
    st.write("- Yerdanat Kurmantayev")
    st.write("- Darmen")
    st.write("- Asset Nukushev")

    st.success("Explore the tabs to analyze, visualize, and predict student performance. 🚀")

# -----------------------------
# BUSINESS CASE TAB
# -----------------------------
if selected == "🔎 Business Case":
    st.title("🎓 CollegeBoard: Student Performance")
    st.image("logo.png", use_container_width=True)
    st.markdown("---")
    st.header("About")
    st.markdown("""
    **OutLook Telecom : Annual Data Review** app is designed to provide insights and predictions based on the data collected over the past year by OutLook Telecom. This platform leverages predictive analytics and advanced data visualization techniques to help stakeholders make informed decisions about business strategies and opportunities for growth.
                Our primary Focus is to analyse Customer retention and churn, which are critical factors in the telecom due to competition in the industry.
    """)

    st.header("Purpose")
    st.markdown("""
    The telecom industry generates a wealth of data, but extracting meaningful insights from this data can be challenging. By building this platform for **Outlook Telecom** using their 2024 dataset from the state of **Califonia**, we aim to provide decision-makers with clear, actionable insights to help drive strategic initiatives, improve customer experiences, and optimize business performance and most imporntantly **Retain Customers**.
    """)

    st.header("Usability")
    st.markdown("""
    - **Data Exploration**: A thorough examination of the dataset used, including available columns, context, statistical summary, and data types.
    - **Data Visualization**: Interactive charts and graphs that provide a clear visual representation of the data, categorised into Customer Demographics and Provided services and Financial trends.  
    - **Predictive Analytics**: Predictive Machine Learning models that forecast trends of Customer status or Customer Retention within Outlook Telecom.  
    - **Actionable Insights**: Data-driven suggestions for improving business operations and customer satisfaction and retention.
    """)

    st.header("Objectives")
    st.markdown("""
    Through this app, we hope to empower OutLook Telecom's management with the tools they need to analyze performance, identify areas for improvement, and make data-driven decisions that foster growth and success in the telecom industry.
    """)

    st.header("Key Features")
    st.markdown("""
    - **Welcome Page**: An introduction to the app, our client OutLook Telecom, and the the team
    - **About Page**: A brief overview of the app's purpose and objectives, Features and Usability.  
    - **Explore Page**: A detailed summary of the dataset, including available columns.
    - **Predictions Page**: Machine learning models predicting customer churn and compare model perfomance.
    - **Conclusion Page**: A summary of the findings and recommendations based on the analysis.
                    
    We hope you find the platform insightful and beneficial in making strategic decisions for OutLook Telecom's growth.
    """)

    st.success("Dive into the data, discover insights, and keep learning .")

# -----------------------------
# DATA PRESENTATION TAB
# -----------------------------
if selected == "👩‍💻 Data Presentation":
    st.title("🔎 Explore Student Performance Data")

    # Load dataset
    dataset = pd.read_csv("StudentPerformanceFactors.csv")

    # Horizontal option menu
    selected = option_menu(
        menu_title=None,
        options=["01: Dictionary", "02: Summary"],
        orientation="horizontal"
    )

    # Complete field dictionary
    field_dict = {
        "Exam_Score": "Final exam score achieved by the student",
        "Attendance": "Percentage of classes attended",
        "Hours_Studied": "Average study hours per week",
        "Previous_Scores": "Average of previous test scores",
        "Parental_Involvement": "Level of parental engagement in the student's education (low/medium/high)",
        "Access_to_Resources": "Availability of study resources at home or school (low/medium/high)",
        "Motivation_Level": "Self-reported motivation to succeed academically (low/medium/high)",
        "Family_Income": "Categorized family income level (low/medium/high)",
        "Teacher_Quality": "Perceived quality of teachers by the student (low/medium/high)",
        "Parental_Education_Level": "Highest education level of parents (high school/college/postgraduate)",
        "Distance_from_Home": "Distance from home to school (near/moderate/far)",
        "Extracurricular_Activities": "Whether the student participates in extracurricular activities (Yes/No)",
        "Internet_Access": "Whether the student has reliable internet access at home (Yes/No)",
        "School_Type": "Type of school attended (Public/Private)",
        "Peer_Influence": "Positive or negative influence from peers (Positive/Negative)",
        "Learning_Disabilities": "Whether the student has learning disabilities (Yes/No)",
        "Gender": "Gender of the student (Male/Female)",
        "Tutoring_Sessions": "Number of additional tutoring sessions attended"
    }

    # --- Dictionary Tab ---
    if selected == "01: Dictionary":
        st.subheader("Dataset Description")
        st.markdown("""
        **This dataset contains records of student performance**,  
        including details about **attendance, study habits, engagement, demographics, and exam scores**.  
        It allows us to:  
        - Explore **factors affecting exam performance**,  
        - Analyze the **impact of categorical and numerical features**,  
        - Build **predictive models** to estimate student scores.
        """)
        
        st.subheader("Columns & Descriptions")
        dataset_columns = pd.DataFrame({'Columns': dataset.columns})
        st.dataframe(dataset_columns, width=700)

        # Dropdown for field descriptions
        field_options = list(field_dict.keys())
        selected_field = st.selectbox("Select a field to view its description:", field_options)
        st.code(f'"{field_dict[selected_field]}"')

    # --- Summary Tab ---
    elif selected == "02: Summary":
        st.subheader("Data Preview")
        view_option = st.radio("View from:", ("Top", "Bottom"))
        if view_option == "Top":
            st.dataframe(dataset.head(10))
        else:
            st.dataframe(dataset.tail(10))

        st.success(f"**Dataset Shape:** {dataset.shape[0]} rows and {dataset.shape[1]} columns")

        st.write("### Statistical Summary")
        st.write(dataset.describe())

        st.write("### Data Types")
        dtypes = dataset.dtypes
        dtype_details = {}
        for dtype in dtypes.unique():
            columns = dtypes[dtypes == dtype].index.tolist()
            dtype_details[str(dtype)] = {
                "Columns": ", ".join(columns),
                "Count": len(columns)
            }

        dtype_df = pd.DataFrame(dtype_details).T.reset_index()
        dtype_df.columns = ['Data Type', 'Columns', 'Count']
        st.write(dtype_df)
    
# -----------------------------
# VISUALIZATIONS TAB
# -----------------------------
if selected == "📊 Data Viz":
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
                    cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 7})
        plt.title("Correlation Matrix")
        st.pyplot(fig_corr)

    # --- SUB-TAB 2: Distributions ---
    with viz_tab2:
         # --- Numeric Feature Distributions in Two Columns ---
        st.write("### Numeric Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="Set2")

        for i in range(0, len(numeric_cols), 2):
            col1, col2 = st.columns(2)
            # First column
            if i < len(numeric_cols):
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(df[numeric_cols[i]], kde=True, ax=ax, color="skyblue")
                    ax.set_title(f"{numeric_cols[i]} Distribution")
                    st.pyplot(fig)
            # Second column
            if i + 1 < len(numeric_cols):
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(df[numeric_cols[i + 1]], kde=True, ax=ax, color="lightgreen")
                    ax.set_title(f"{numeric_cols[i + 1]} Distribution")
                    st.pyplot(fig)

    # --- SUB-TAB 3: Relationships ---
    with viz_tab3:
        st.subheader("Numerical Features vs. Target")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Attendance", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Attendance vs Exam Score")
            plt.xlabel("Attendance, %")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Previous_Scores", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Previous Scores vs Exam Score")
            plt.xlabel("Previous Scores")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Hours_Studied", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Hours Studied vs Exam Score")
            plt.xlabel("Hours Studied")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="Hours_Studied", y="Exam_Score", hue="Attendance", data=df, ax=ax, palette="viridis", alpha=0.7)
            plt.title("Study Hours by Attendance")
            plt.xlabel("Hours Studied")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

    # --- SUB-TAB 4: Categorical Impacts ---
    with viz_tab4:
        st.subheader("Impact of Categorical Variables on Exam Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Parental_Involvement", y="Exam_Score", data=df, ax=ax, palette="Set2")
            plt.title("Parental Involvement Impact")
            plt.xlabel("Parental Involvement")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Tutoring_Sessions", y="Exam_Score", data=df, ax=ax, palette="Set2", errorbar=None)
            plt.title("Tutoring Sessions Impact")
            plt.xlabel("Number of Tutoring Sessions")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Access_to_Resources", y="Exam_Score", data=df, ax=ax, palette="Set2")
            plt.title("Access to Resources Impact")
            plt.xlabel("Access to Resources")
            plt.ylabel("Exam Score")
            st.pyplot(fig)

# -----------------------------
# PREDICTIONS TAB
# -----------------------------
if selected == "🧠 Regressions":
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
                model = Ridge(alpha=1.0)
            else:
                model = Lasso(alpha=0.1, max_iter=10000)

            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

            # Split
            test_size = test_size_pct / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )

            # Train + evaluate
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = (mean_squared_error(y_test, y_pred)) ** 0.5

            st.markdown("### Regression Visualizations")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Predicted vs Actual**")
                fig_pred, ax_pred = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax_pred.set_xlabel("Actual Exam Score")
                ax_pred.set_ylabel("Predicted Exam Score")
                ax_pred.set_title("Predicted vs Actual")
                st.pyplot(fig_pred)

            with col2:
                st.markdown("**Residuals Plot**")
                residuals = y_test - y_pred
                fig_resid, ax_resid = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
                ax_resid.axhline(0, color='r', linestyle='--', lw=2)
                ax_resid.set_xlabel("Predicted Exam Score")
                ax_resid.set_ylabel("Residuals")
                ax_resid.set_title("Residuals vs Predicted")
                st.pyplot(fig_resid)

            st.success(f"🎉 **{model_name}** model trained successfully!")
            
            # Display metrics in a visually pleasing way
            st.markdown("### Model Performance Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("Mean Absolute Error", f"{mae:.4f}")
            m3.metric("Root Mean Squared Error", f"{rmse:.4f}")

# -----------------------------
# CONCLUSION TAB
# -----------------------------
if selected == "🎬 Conclusion":
    st.title("Conclusion & Key Insights 🎬")

    st.markdown("""
    ---
    ## 🔹 Summary of Findings
    This dashboard provides a comprehensive analysis of student performance data. Key takeaways include:

    - **Attendance and Hours Studied** are strongly correlated with exam performance.
    - **Parental Involvement, Motivation, and Access to Resources** significantly impact student outcomes.
    - **Predictive models** (Linear Regression, Ridge, Lasso) provide reasonable estimates of exam scores based on available features.
    - **Engagement** (Attendance × Hours Studied) can improve model performance when included as a feature.

    ## 🔹 Recommendations
    Based on the analysis:

    1. Encourage **consistent class attendance** and **structured study hours**.  
    2. Provide **additional learning resources** and support to students with lower motivation or parental involvement.  
    3. Consider **tutoring programs** targeted at students struggling in key areas.  
    4. Monitor **student engagement metrics** as an early indicator for potential performance issues.

    ## 🔹 Next Steps
    - Expand the dataset with more students or semesters to improve model accuracy.  
    - Explore **advanced predictive models** (e.g., Random Forest, Gradient Boosting).  
    - Integrate **interactive dashboards for teachers and parents** to track student progress in real time.  
    """)

    st.success("This concludes the analysis. Use the other tabs to explore, visualize, and predict student performance. 🎓")