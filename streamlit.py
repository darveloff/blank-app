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
        Our client is College Board, a non-profit College Entrance Examination Board whose goal is to expand access to higher education. In the light of the recent reinstating of mandatory standardized testing (SAT®/ACT®) across top US universities, College Board developed concerns regarding education environment factors that may impact objective evaluation of students' success on the exam.

        Our team was approached with a task to analyse data about 6000+ high school students in order to determine what types of educational environments ensure optimal exam performance. This interactive platform allows you to:
        - Explore key student performance indicators.
        - Develop insights about factors that affect exam scores.
        - Suggest measures that would make standardised testing a more objective metric.
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

    st.header("Project Goal")
    st.markdown("""
    The primary goal of this project is to identify the most significant predictors of standardised test performance and evaluate how different educational environments contribute to student outcomes. Combining interactive visualisations and statistical analysis, we aim to discover patterns and relationships that can inform data-driven educational policies and strategies to support students in their preparation for testing.
    """)

    st.header("Key Features")
    st.markdown("""
    - **Data Exploration**: Comprehensive examination of the dataset is made through an overview of columns, data types, and statistics.
    - **Data Visualisation**: Visual representation of data is facilitated through interactive graphs and charts, facilitating understanding of the discovered patterns and relationships between different performance factors.
    - **Data-Driven Insights**: Analysis of synthetic data allows us to extrapolate conclusions into real-world settings, formulating relevant suggestions for improvement.
    """)

    st.success("Dive into the data, discover insights, and keep learning.")

# -----------------------------
# DATA PRESENTATION TAB
# -----------------------------
if selected == "👩‍💻 Data Presentation":
    st.title("🔎 Explore Student Performance Data")

    # Load dataset
    dataset = pd.read_csv("StudentPerformanceFactors.csv")

    df_viz = dataset.copy()
    
    ordinal_cols = [
        "Parental_Involvement", "Access_to_Resources", "Motivation_Level",
        "Family_Income", "Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"
    ]

    # normalize strings (keeps NaN as NaN)
    df_viz[ordinal_cols] = df_viz[ordinal_cols].apply(lambda s: s.astype("string").str.strip().str.lower())

    # impute missing values (most frequent category in each column)
    for col in ordinal_cols:
        df_viz[col] = df_viz[col].fillna(df_viz[col].mode(dropna=True)[0])

    encoder = OrdinalEncoder(categories=[
        ["low", "medium", "high"],          # Parental_Involvement
        ["low", "medium", "high"],          # Access_to_Resources
        ["low", "medium", "high"],          # Motivation_Level
        ["low", "medium", "high"],          # Family_Income
        ["low", "medium", "high"],          # Teacher_Quality
        ["high school", "college", "postgraduate"],
        ["near", "moderate", "far"]
    ])

    df_viz[ordinal_cols] = encoder.fit_transform(df_viz[ordinal_cols])

    binary_maps = {
        "Extracurricular_Activities": {"No": 0, "Yes": 1},
        "Internet_Access": {"No": 0, "Yes": 1},
        "School_Type": {"Public": 0, "Private": 1},
        "Peer_Influence": {"Negative": 0, "Positive": 1},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "Gender": {"Female": 0, "Male": 1},
    }

    for col, mp in binary_maps.items():
        df_viz[col] = df_viz[col].astype("string").str.strip()
        df_viz[col] = df_viz[col].map(mp)

    df_viz["Peer_Influence"] = df_viz["Peer_Influence"].fillna(-1)

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
        st.header("Dataset Overview")
        st.markdown("""
        The dataset provides a comprehensive overview of various factors affecting student performance in exams. It includes information on study habits, attendance, parental involvement, and other aspects influencing academic success, which is measured by the final exam score.

        The data consists of 6000+ synthetic student records and combines numerical, ordinal, and categorical variables to simulate diverse academic environments.
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

        st.subheader("Cleaned Data")
        view_option = st.radio("View from:", ("Top", "Bottom"), key=1)
        if view_option == "Top":
            st.dataframe(df_viz.head(10))
        else:
            st.dataframe(df_viz.tail(10))

        st.markdown("""*This table shows the cleaned dataset after categorical features have been encoded into numerical values for statistical analysis and modeling, i.e. Low, Medium, High --> 0, 1, 2*""")

        st.success(f"**Dataset Shape:** {df_viz.shape[0]} rows and {df_viz.shape[1]} columns")

        st.write("### Statistical Summary")
        st.write(df_viz.describe())
        st.markdown("""*We can see that on average, attendance is around 80% and that interestingly, the average exam score feel down from 75 to 67*""")

        st.write("### Data Types")
        dtypes = df_viz.dtypes
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

        st.write("### Feature Importance (Correlation with Exam Score)")

        # Compute correlations with target
        corr_with_target = df_viz.corr()["Exam_Score"].drop("Exam_Score")

        # Sort by absolute correlation (strongest relationships first)
        importance = corr_with_target.reindex(
            corr_with_target.abs().sort_values(ascending=False).index
        )

        # Display as dataframe
        importance_df = importance.reset_index()
        importance_df.columns = ["Feature", "Correlation with Exam_Score"]
        st.dataframe(importance_df)
        st.markdown("""*Hence, Attendance, Hours Studied, and Previous Scores are the most important factors for Exam Scores*""")
    
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
        corr = df_corr.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, 
                    cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 7})
        plt.title("Correlation Matrix")
        st.pyplot(fig_corr)
        st.markdown("*This heatmap shows the linear relationships between encoded features.*")

        st.write("### Correlation-Based Feature Ranking")

        # Compute correlations with target
        corr_with_target = df_corr.corr()["Exam_Score"].drop("Exam_Score")

        # Sort by absolute correlation (strongest relationships first)
        importance = corr_with_target.reindex(
            corr_with_target.abs().sort_values(ascending=False).index
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x=importance.values,
            y=importance.index,
            palette="coolwarm",
            ax=ax
        )
        ax.set_title("Feature Importance (Correlation with Exam_Score)")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
        st.markdown("*The correlations of factors with Exam Scores sorted by importance.*")

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
        st.markdown(
            """
            ## Findings 🔍
            - Exam Score distribution shows that there is low variance, i.e. that students perform decently very often and that getting high scores above 80 is rare.
            - Attendance is quite evenly distributed and there are no students who attend less than 60% of classes
            - Previous scores are very widely distributed and there were a lot of students who got a high score unlike current exam scores. Could imply collaboration between students
            """
        )

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
            st.markdown("""**Finding:** Exam scores tend to increase as attendance improves. Students with consistently high attendance generally achieve higher exam results, indicating a positive relationship between classroom participation and performance.""")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Previous_Scores", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Previous Scores vs Exam Score")
            plt.xlabel("Previous Scores")
            plt.ylabel("Exam Score")
            st.pyplot(fig)
            st.markdown("""**Finding:** Previous academic performance is also associated with final exam outcomes. Students who performed well historically are more likely to maintain higher scores, suggesting performance consistency over time.""")


        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x="Hours_Studied", y="Exam_Score", data=df, ax=ax, scatter_kws={'alpha':0.5})
            plt.title("Hours Studied vs Exam Score")
            plt.xlabel("Hours Studied")
            plt.ylabel("Exam Score")
            st.pyplot(fig)
            st.markdown("""**Finding:** There is a positive relationship between study hours and exam scores, but less than Attendance and Hours Studied. Students dedicating more time to studying tend to achieve better results, though the relationship is not perfectly linear.""")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="Hours_Studied", y="Exam_Score", hue="Attendance", data=df, ax=ax, palette="viridis", alpha=0.7)
            plt.title("Study Hours by Attendance")
            plt.xlabel("Hours Studied")
            plt.ylabel("Exam Score")
            st.pyplot(fig)
            st.markdown("""**Finding:** In general, students who combine high study hours with strong attendance achieve the highest exam scores. This suggests that engagement inside and outside the classroom jointly contributes to academic success. However, among high achievers, attendance metrics are quite dispersed""")

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
            st.markdown("""**Finding:** There is a weak but noticeable correlation between parental involvement and exam scores, which suggests that in families where parents are more involved with students, academic performance is higher""")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Tutoring_Sessions", y="Exam_Score", data=df, ax=ax, palette="Set2", errorbar=None)
            plt.title("Tutoring Sessions Impact")
            plt.xlabel("Number of Tutoring Sessions")
            plt.ylabel("Exam Score")
            st.pyplot(fig)
            st.markdown("""**Finding:** The number of tutoring sessions has a weak positive relationship with exam scores, but it declines slightly after 6 sessions, suggesting that 6 is the optimal amount""")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Access_to_Resources", y="Exam_Score", data=df, ax=ax, palette="Set2")
            plt.title("Access to Resources Impact")
            plt.xlabel("Access to Resources")
            plt.ylabel("Exam Score")
            st.pyplot(fig)
            st.markdown("""**Finding:** Having access to educational resources leads to higher exam scores""")

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

        selected_cols = st.multiselect(
            "Select Features to Include",
            options=[c for c in df.columns if c != TARGET_COL],
            default=[c for c in df.columns if c != TARGET_COL],
            help="Choose which columns to use as predictors in the model"
        )

        train_clicked = st.button("🚀 Train Model", type="primary", use_container_width=True)

    if train_clicked:
        if not selected_cols:
            st.error("Select at least one feature before training.")
            st.stop()

        with st.spinner(f"Training {model_name}..."):
            # Prepare X, y
            X = df[selected_cols].copy()
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
                model = Lasso(alpha=0.1, max_iter=10000, random_state=42)

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
    Based on the observed relationships, we propose the following measures to enhance objectivity of standardised testing across diverse academic settings:

    1. **Equity-Focused Preparation Support** — Both tutoring sessions and resource access positively affect students’ scores. As such, College Board is advised to expand free or subsidised exam preparation programs, especially for students from public schools and power-income households. 
    2. **Expanded Digital Access Programs** — Drawing upon the correlation between Internet access and performance, another suggestion is to provide reliable digital resources, practice platforms, and learning materials, targeted at disadvantaged communities. This would improve fairness in preparation opportunities.
    3. **Attendance and Engagement Incentives** — Strong interaction between attendance and study time suggests that sustained student engagement during the learning process significantly improves their performance during the test. Schools should be encouraged to implement engagement tracking and targeted support for students with low attendance before the exam period.
    4. **Holistic Admissions Advocacy** — Our findings reinforce the idea that standardised testing cannot be a universal measure of academic potential on its own, independent from the learning environment. College Board should advocate for admissions policies that combine test scores with school context indicators to maintain fairness. 
    """)

    st.success("This concludes the analysis. Use the other tabs to explore, visualize, and predict student performance. 🎓")