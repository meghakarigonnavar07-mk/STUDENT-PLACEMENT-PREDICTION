import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PlacePro - Student Placement",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# CUSTOM CSS - TEXT AND CHATBOT COLOR FIX
# ==========================================
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #FFFFFF !important;
    color: #172554 !important;
}

/* Main headings */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #172554 !important;
}

/* Normal text */
.stApp p, .stApp li, .stApp label, .stApp span {
    color: #1E293B;
}

/* Markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #1E293B !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #243B55) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #FFFFFF !important;
}

/* Metrics */
[data-testid="stMetricLabel"] { color: #334155 !important; }
[data-testid="stMetricValue"] { color: #1D4ED8 !important; }
[data-testid="stMetricDelta"] { color: #1E293B !important; }

/* Inputs */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
    color: #172554 !important;
}
.stTextInput input, .stNumberInput input {
    color: #172554 !important;
    background-color: #FFFFFF !important;
}
.stSelectbox div, .stMultiSelect div { color: #172554; }

/* Tables */
[data-testid="stDataFrame"] { background-color: #FFFFFF !important; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background-color: #4338CA !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
}
.stButton > button p, .stFormSubmitButton > button p, .stDownloadButton > button p {
    color: #FFFFFF !important;
}

/* Alerts */
[data-testid="stAlert"] p { color: #172554 !important; }

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5F1;
}

/* Radio buttons */
[data-testid="stRadio"] label { color: #172554 !important; }

/* Custom metric cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    min-height: 110px;
}
.metric-label { color: #475569 !important; font-size: 14px; }
.metric-value { color: #1D4ED8 !important; font-size: 27px; font-weight: 700; }

/* CHATBOT: white message boxes with blue text */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 10px !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #1D4ED8 !important;
}
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #1D4ED8 !important;
    border: 1px solid #93C5FD !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748B !important; }
[data-testid="stChatMessageAvatar"] { background-color: #DBEAFE !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOGIN PAGE
# ==========================================
USERNAME = "student"
PASSWORD = "student123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Login-only styling. Keep this HTML inside a Python triple-quoted string.
    st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 25, 55, 0.38), rgba(10, 25, 55, 0.48)),
        url("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?auto=format&fit=crop&w=1920&q=85") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.97) !important;
        padding: 24px !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.22) !important;
    }
    [data-testid="stForm"] label,
    [data-testid="stForm"] p,
    [data-testid="stForm"] div { color: #172554 !important; }
    [data-testid="stForm"] input {
        background: #FFFFFF !important;
        color: #172554 !important;
        border: 1px solid #93C5FD !important;
        border-radius: 8px !important;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #4338CA !important;
        color: #FFFFFF !important;
        border-radius: 9px !important;
        min-height: 44px !important;
    }
    [data-testid="stFormSubmitButton"] button p { color: #FFFFFF !important; }
    [data-testid="stAlert"] { background: rgba(255,255,255,0.96) !important; }
    [data-testid="stAlert"] p { color: #172554 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:22px 10px 16px;">
        <h1 style="color:#FFFFFF !important; font-size:48px; font-weight:800;
                   text-shadow:2px 2px 8px #000000; margin-bottom:8px;">
            🎓 PlacePro
        </h1>
        <h3 style="color:#FFFFFF !important; text-shadow:1px 1px 6px #000000;">
            Student Placement Analytics System
        </h3>
        <p style="color:#FFFFFF !important; font-size:18px;
                  text-shadow:1px 1px 6px #000000;">
            Predict Your Future. Prepare for Success.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, login_col, _ = st.columns([1, 1.5, 1])

    with login_col:
        st.subheader("🔐 Login to Your Account")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login", use_container_width=True)

        if login_button:
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password!")

        st.info("Demo username: student")
        st.info("Demo password: student123")

    st.stop()

# ==========================================
# CREATE DEMO DATASET
# ==========================================
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    n = 600

    cgpa = rng.uniform(4, 10, n).round(2)
    aptitude = rng.integers(25, 101, n)
    communication = rng.integers(2, 11, n)
    internship = rng.integers(0, 2, n)

    departments = rng.choice(["CSE", "ISE", "ECE", "EEE", "Mechanical"], n)

    score = (
        (cgpa - 5) * 0.75
        + (aptitude - 40) * 0.025
        + (communication - 4) * 0.20
        + internship * 0.65
        + rng.normal(0, 0.9, n)
    )
    placement = (score > 2.45).astype(int)

    package = np.where(
        placement == 1,
        rng.uniform(3, 12, n) + np.maximum(cgpa - 8, 0) * 1.5,
        0
    ).round(2)

    return pd.DataFrame({
        "CGPA": cgpa,
        "Aptitude": aptitude,
        "Communication": communication,
        "Internship": internship,
        "Department": departments,
        "Placement": placement,
        "Package_LPA": package
    })


df = create_dataset()
FEATURES = ["CGPA", "Aptitude", "Communication", "Internship"]

# ==========================================
# TRAIN MACHINE LEARNING MODEL
# ==========================================
@st.cache_resource
def train_model():
    data = create_dataset()
    X = data[FEATURES]
    y = data["Placement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, X_test, y_test, predictions


model, accuracy, X_test, y_test, y_pred = train_model()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def get_prediction(student):
    prediction = int(model.predict(student)[0])
    probabilities = model.predict_proba(student)[0]
    placed_index = list(model.classes_).index(1)
    score = probabilities[placed_index] * 100
    return prediction, score


def show_placement_charts(data):
    placed = int(data["Placement"].sum())
    total = len(data)
    not_placed = total - placed
    left, right = st.columns(2)

    with left:
        counts = pd.DataFrame({
            "Status": ["Placed", "Not Placed"],
            "Students": [placed, not_placed]
        })
        fig = px.pie(counts, names="Status", values="Students", hole=0.5,
                     title="Placement Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        dept = (
            data.groupby("Department")["Placement"].mean().mul(100)
            .reset_index(name="Placement Rate")
        )
        fig = px.bar(dept, x="Department", y="Placement Rate", text_auto=".1f",
                     title="Department-wise Placement Rate")
        st.plotly_chart(fig, use_container_width=True)


def prepare_download_data(data):
    result = data.copy()
    result["Placement"] = result["Placement"].map({1: "Placed", 0: "Not Placed"})
    result["Internship"] = result["Internship"].map({1: "Yes", 0: "No"})
    return result


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("🎓 PlacePro")
    st.caption("STUDENT PLACEMENT ANALYTICS")
    st.divider()

    page = st.radio(
        "📌 MAIN NAVIGATION",
        [
            "🧑‍🎓 Student Information",
            "📊 Dashboard",
            "🔮 Predict Placement",
            "📈 Placement Analytics",
            "👨‍🎓 Student Performance",
            "🤖 Model Evaluation",
            "💡 Career Guidance",
            "📁 Student Dataset",
            "📤 Upload Student Data",
            "📑 Generate Report",
            "ℹ️ About Project",
            "🤖 Chatbot"
        ],
        key="main_navigation"
    )

    st.divider()
    st.subheader("🤖 Model Information")
    st.write("Algorithm: Random Forest")
    st.write("Features: 4")
    st.write("Framework: Streamlit")
    st.divider()
    st.metric("Total Students", len(df))
    st.metric("Placed Students", int(df["Placement"].sum()))
    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
# ==========================================
# STUDENT INFORMATION
# ==========================================
if page == "🧑‍🎓 Student Information":

    st.title("🧑‍🎓 Student Information Management")
    st.write("Add, edit, view and delete student information.")

    # Store student information in session
    if "students_data" not in st.session_state:
        st.session_state.students_data = pd.DataFrame(                  
            columns=[
                "Roll Number",
                "Student Name",
                "Department",
                "Year",
                "Email",
                "Phone Number",
                "CGPA"
            ]
        )

    st.subheader("➕ Add Student")

    with st.form("add_student_form"):

        roll_no = st.text_input("Roll Number")
        student_name = st.text_input("Student Name")
        department = st.selectbox(
            "Department",
            ["BCA", "BBA", "BCom", "BSc", "BE"]
        )
        year = st.selectbox(
            "Year",
            ["1st Year", "2nd Year", "3rd Year", "4th Year"]
        )
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")
        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

        save_button = st.form_submit_button("💾 Save Student")

    if save_button:

        if not roll_no.strip() or not student_name.strip():
            st.error("Please enter Roll Number and Student Name.")

        elif roll_no.strip() in st.session_state.students_data["Roll Number"].values:
            st.error("This Roll Number already exists.")

        else:
            new_student = pd.DataFrame([{
                "Roll Number": roll_no.strip(),
                "Student Name": student_name.strip(),
                "Department": department,
                "Year": year,
                "Email": email.strip(),
                "Phone Number": phone.strip(),
                "CGPA": cgpa
            }])

            st.session_state.students_data = pd.concat(
                [st.session_state.students_data, new_student],
                ignore_index=True
            )

            st.success("Student information saved successfully!")
            st.rerun()

    st.divider()

    st.subheader("📋 Student List")

    students = st.session_state.students_data

    if not students.empty:

        st.dataframe(
            students,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("✏️ Edit Student")

        selected_roll = st.selectbox(
            "Select Roll Number to Edit",
            students["Roll Number"].tolist()
        )

        selected_student = students[
            students["Roll Number"] == selected_roll
        ].iloc[0]

        with st.form("edit_student_form"):

            edited_name = st.text_input(
                "Student Name",
                value=str(selected_student["Student Name"])
            )

            edited_department = st.selectbox(
                "Department",
                ["BCA", "BBA", "BCom", "BSc", "BE"],
                index=["BCA", "BBA", "BCom", "BSc", "BE"].index(
                    selected_student["Department"]
                )
            )

            edited_year = st.selectbox(
                "Year",
                ["1st Year", "2nd Year", "3rd Year", "4th Year"],
                index=["1st Year", "2nd Year", "3rd Year", "4th Year"].index(
                    selected_student["Year"]
                )
            )

            edited_email = st.text_input(
                "Email",
                value=str(selected_student["Email"])
            )

            edited_phone = st.text_input(
                "Phone Number",
                value=str(selected_student["Phone Number"])
            )

            edited_cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=float(selected_student["CGPA"]),
                step=0.1
            )

            update_button = st.form_submit_button("✏️ Update Student")

        if update_button:

            if not edited_name.strip():
                st.error("Student Name cannot be empty.")

            else:
                idx = st.session_state.students_data.index[
                    st.session_state.students_data["Roll Number"] == selected_roll
                ][0]

                st.session_state.students_data.loc[idx, "Student Name"] = edited_name.strip()
                st.session_state.students_data.loc[idx, "Department"] = edited_department
                st.session_state.students_data.loc[idx, "Year"] = edited_year
                st.session_state.students_data.loc[idx, "Email"] = edited_email.strip()
                st.session_state.students_data.loc[idx, "Phone Number"] = edited_phone.strip()
                st.session_state.students_data.loc[idx, "CGPA"] = edited_cgpa

                st.success("Student information updated successfully!")
                st.rerun()

        st.subheader("🗑️ Delete Student")

        delete_roll = st.selectbox(
            "Select Roll Number to Delete",
            students["Roll Number"].tolist(),
            key="delete_student_roll"
        )

        if st.button("🗑️ Delete Selected Student"):

            st.session_state.students_data = (
                st.session_state.students_data[
                    st.session_state.students_data["Roll Number"] != delete_roll
                ].reset_index(drop=True)
            )

            st.success("Student deleted successfully!")
            st.rerun()

        csv_data = st.session_state.students_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Student Information",
            data=csv_data,
            file_name="student_information.csv",
            mime="text/csv"
        )

    else:
        st.info("No students added yet. Enter student details above.")

# ==========================================
# DASHBOARD
# ==========================================
if page == "📊 Dashboard":
    st.title("📊 Placement Analytics Dashboard")
    st.write("Welcome to PlacePro. Explore student performance and placement statistics.")

    total = len(df)
    placed = int(df["Placement"].sum())
    not_placed = total - placed
    rate = placed / total * 100
    placed_data = df[df["Placement"] == 1]
    avg_package = placed_data["Package_LPA"].mean() if not placed_data.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Students", total, "Demo records")
    with c2:
        metric_card("Placed Students", placed, "Sample data")
    with c3:
        metric_card("Placement Rate", f"{rate:.1f}%", "Demo rate")
    with c4:
        metric_card("Average Package", f"{avg_package:.2f} LPA", "Placed demo records")

    st.subheader("Placement Overview")
    show_placement_charts(df)
    st.subheader("Academic Performance")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="CGPA", color="Placement", nbins=20,
                           barmode="overlay", title="CGPA Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(df, x="CGPA", y="Aptitude", color="Placement",
                         size="Communication", hover_data=["Department", "Internship"],
                         title="CGPA vs Aptitude")
        st.plotly_chart(fig, use_container_width=True)
    st.info("This dashboard uses synthetic demonstration data.")

# ==========================================
# PREDICT PLACEMENT
# ==========================================
elif page == "🔮 Predict Placement":
    st.title("🔮 Student Placement Predictor")
    st.write("Enter the student's details to predict placement.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            aptitude = st.slider("Aptitude Score", 0, 100, 70)
        with right:
            communication = st.slider("Communication Skills", 1, 10, 7)
            internship = st.selectbox("Internship Experience", ["Yes", "No"])
        submit = st.form_submit_button("Predict Placement", use_container_width=True)

    if submit:
        student = pd.DataFrame([[
            cgpa, aptitude, communication, 1 if internship == "Yes" else 0
        ]], columns=FEATURES)
        prediction, score = get_prediction(student)
        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Model predicts: Likely to be Placed")
        else:
            st.warning("⚠️ Model predicts: Not Likely to be Placed")

        st.metric("Model Placement Score", f"{score:.1f}%")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": "%"},
            title={"text": "Placement Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4338ca"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        importance = pd.DataFrame({
            "Feature": FEATURES, "Importance": model.feature_importances_
        }).sort_values("Importance")
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is an educational prediction, not a guarantee of employment.")

# ==========================================
# PLACEMENT ANALYTICS
# ==========================================
elif page == "📈 Placement Analytics":
    st.title("📈 Placement Analytics")
    st.write("Detailed analysis of student placement outcomes.")
    total = len(df)
    placed = int(df["Placement"].sum())
    rate = placed / total * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", total)
    c2.metric("Placed Students", placed)
    c3.metric("Placement Rate", f"{rate:.1f}%")
    show_placement_charts(df)

    st.subheader("Department Summary")
    summary = df.groupby("Department").agg(
        Students=("Placement", "count"),
        Placed=("Placement", "sum"),
        Average_CGPA=("CGPA", "mean"),
        Average_Aptitude=("Aptitude", "mean"),
        Placement_Rate=("Placement", "mean")
    ).reset_index()
    summary["Placement_Rate"] = (summary["Placement_Rate"] * 100).round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.info("Analytics are based on demo data.")

# ==========================================
# STUDENT PERFORMANCE
# ==========================================
elif page == "👨‍🎓 Student Performance":
    st.title("👨‍🎓 Student Performance")
    st.write("Explore academic performance and student skills.")
    left, right = st.columns(2)

    with left:
        fig = px.histogram(df, x="CGPA", color="Placement", nbins=20,
                           barmode="overlay", title="CGPA Distribution by Placement")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(df, x="CGPA", y="Aptitude", color="Placement",
                         size="Communication", hover_data=["Department", "Internship"],
                         title="CGPA vs Aptitude")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Department-wise Performance")
    summary = df.groupby("Department").agg(
        Students=("CGPA", "count"),
        Average_CGPA=("CGPA", "mean"),
        Average_Aptitude=("Aptitude", "mean"),
        Average_Communication=("Communication", "mean"),
        Placement_Rate=("Placement", "mean")
    ).reset_index()
    summary["Placement_Rate"] = (summary["Placement_Rate"] * 100).round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Filter Students")
    min_cgpa = st.slider("Minimum CGPA", 0.0, 10.0, 5.0, 0.5)
    filtered = df[df["CGPA"] >= min_cgpa]
    st.write(f"Students matching the filter: {len(filtered)}")
    st.dataframe(prepare_download_data(filtered), use_container_width=True, hide_index=True)

# ==========================================
# MODEL EVALUATION
# ==========================================
elif page == "🤖 Model Evaluation":
    st.title("🤖 Model Evaluation")
    st.write("Evaluate the Random Forest classification model.")
    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig = px.imshow(
        cm, text_auto=True, x=["Not Placed", "Placed"], y=["Not Placed", "Placed"],
        labels={"x": "Predicted", "y": "Actual", "color": "Students"},
        title="Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Not Placed", "Placed"],
        output_dict=True, zero_division=0
    )
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    importance = pd.DataFrame({
        "Feature": FEATURES, "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("The model is evaluated on synthetic data. Its accuracy does not represent real-world performance.")

# ==========================================
# CAREER GUIDANCE
# ==========================================
elif page == "💡 Career Guidance":
    st.title("💡 Career Guidance")
    st.write("Get general suggestions based on your skills.")

    with st.form("career_form"):
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
        aptitude = st.slider("Aptitude Score", 0, 100, 60)
        communication = st.slider("Communication Skills", 1, 10, 6)
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        submit = st.form_submit_button("Get Career Guidance")

    if submit:
        student = pd.DataFrame([[
            cgpa, aptitude, communication, 1 if internship == "Yes" else 0
        ]], columns=FEATURES)
        prediction, score = get_prediction(student)
        st.subheader("Career Readiness")
        st.metric("Model Placement Score", f"{score:.1f}%")
        if prediction == 1:
            st.success("The model predicts a likely placement outcome.")
        else:
            st.warning("Further preparation may be helpful.")

        st.subheader("Recommended Improvement Areas")
        suggestions = []
        if cgpa < 7:
            suggestions.append("Strengthen academic knowledge and core subjects.")
        if aptitude < 70:
            suggestions.append("Practice quantitative aptitude and logical reasoning.")
        if communication < 7:
            suggestions.append("Practice interviews, presentations and group discussions.")
        if internship == "No":
            suggestions.append("Explore internships and practical project opportunities.")
        if not suggestions:
            suggestions.append("Continue developing technical skills and interview confidence.")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
        st.info("These are general educational suggestions. The model score does not guarantee employment.")


# ==========================================
# STUDENT DATASET
# ==========================================
elif page == "📁 Student Dataset":
    st.title("📁 Student Dataset Explorer")
    st.write("View, filter and download student records.")
    status = st.selectbox("Filter Placement Status", ["All", "Placed", "Not Placed"])
    filtered = df.copy()

    if status == "Placed":
        filtered = filtered[filtered["Placement"] == 1]
    elif status == "Not Placed":
        filtered = filtered[filtered["Placement"] == 0]

    display_df = prepare_download_data(filtered)
    st.write(f"Records displayed: {len(display_df)}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Dataset as CSV", data=csv,
                       file_name="student_placement_demo.csv", mime="text/csv")

# ==========================================
# UPLOAD STUDENT DATA
# ==========================================
elif page == "📤 Upload Student Data":
    st.title("📤 Upload Student Data")
    st.write("Upload a CSV file containing student records.")
    st.info("Required columns: CGPA, Aptitude, Communication, Internship. Optional columns: Department, Placement, Package_LPA.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(20), use_container_width=True, hide_index=True)
            required_columns = ["CGPA", "Aptitude", "Communication", "Internship"]
            missing = [col for col in required_columns if col not in uploaded_df.columns]

            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                for col in required_columns:
                    uploaded_df[col] = pd.to_numeric(uploaded_df[col], errors="coerce")

                if uploaded_df[required_columns].isnull().any().any():
                    st.error("Required columns contain missing or invalid values.")
                elif not uploaded_df["CGPA"].between(0, 10).all():
                    st.error("CGPA values must be between 0 and 10.")
                elif not uploaded_df["Aptitude"].between(0, 100).all():
                    st.error("Aptitude values must be between 0 and 100.")
                elif not uploaded_df["Communication"].between(1, 10).all():
                    st.error("Communication values must be between 1 and 10.")
                elif not uploaded_df["Internship"].isin([0, 1]).all():
                    st.error("Internship must use 1 for Yes and 0 for No.")
                else:
                    st.success("CSV uploaded and validated successfully!")
                    st.subheader("Predict Placement for Uploaded Students")
                    uploaded_df["Predicted_Placement"] = model.predict(uploaded_df[FEATURES])
                    probabilities = model.predict_proba(uploaded_df[FEATURES])
                    placed_index = list(model.classes_).index(1)
                    uploaded_df["Placement_Score_Percent"] = (probabilities[:, placed_index] * 100).round(2)

                    result_df = uploaded_df.copy()
                    result_df["Predicted_Placement"] = result_df["Predicted_Placement"].map({1: "Placed", 0: "Not Placed"})
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    download_csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Prediction Results", data=download_csv,
                                       file_name="uploaded_student_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Unable to read CSV file: {e}")

# ==========================================
# GENERATE REPORT
# ==========================================
elif page == "📑 Generate Report":
    st.title("📑 Generate Placement Report")
    st.write("View student placement statistics and download reports.")

    total = len(df)
    placed = int(df["Placement"].sum())
    not_placed = total - placed
    placement_rate = (placed / total * 100) if total > 0 else 0
    placed_df = df[df["Placement"] == 1]
    average_package = placed_df["Package_LPA"].mean() if not placed_df.empty else 0

    st.header("📊 Report Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Students", total)
    with c2:
        st.metric("Placed Students", placed)
    with c3:
        st.metric("Not Placed Students", not_placed)
    with c4:
        st.metric("Placement Rate", f"{placement_rate:.2f}%")
    st.metric("Average Package", f"{average_package:.2f} LPA")
    st.divider()

    st.header("🏢 Department-wise Report")
    department_report = df.groupby("Department").agg(
        Total_Students=("Placement", "count"),
        Placed_Students=("Placement", "sum"),
        Average_CGPA=("CGPA", "mean")
    ).reset_index()
    department_report["Not_Placed_Students"] = department_report["Total_Students"] - department_report["Placed_Students"]
    department_report["Placement_Rate"] = (
        department_report["Placed_Students"] / department_report["Total_Students"] * 100
    ).round(2)
    st.dataframe(department_report.round(2), use_container_width=True, hide_index=True)

    st.header("📈 Placement Overview")
    counts = pd.DataFrame({
        "Status": ["Placed", "Not Placed"], "Students": [placed, not_placed]
    })
    fig = px.pie(counts, names="Status", values="Students",
                 title="Student Placement Distribution", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.header("📥 Download Report")
    report_text = f"""
PLACEPRO - STUDENT PLACEMENT REPORT

Project: Student Placement Prediction

REPORT SUMMARY
------------------------------
Total Students: {total}
Placed Students: {placed}
Not Placed Students: {not_placed}
Placement Rate: {placement_rate:.2f}%
Average Package: {average_package:.2f} LPA
Model Test Accuracy: {accuracy * 100:.2f}%

NOTE:
This report uses synthetic demonstration data.
It does not represent actual college placement results.
"""
    st.download_button(label="⬇️ Download Placement Report", data=report_text,
                       file_name="PlacePro_Report.txt", mime="text/plain", use_container_width=True)
    csv_data = department_report.to_csv(index=False).encode("utf-8")
    st.download_button(label="⬇️ Download Department Report CSV", data=csv_data,
                       file_name="Department_Report.csv", mime="text/csv", use_container_width=True)

# ==========================================
# ABOUT PROJECT
# ==========================================
elif page == "ℹ️ About Project":
    st.title("🎓 About PlacePro")
    st.subheader("Student Placement Prediction Using Machine Learning")
    st.divider()
    st.header("📌 Project Overview")
    st.write("""
    PlacePro is a student placement analytics web application developed using Python,
    Streamlit and Machine Learning.

    It demonstrates how academic performance and skill-related information can be used
    to explore student placement patterns and generate educational predictions.
    """)
    st.header("🎯 Project Objectives")
    st.markdown("""
    - Predict student placement outcomes.
    - Analyze academic performance.
    - Explore aptitude and communication skills.
    - Compare department-wise placement rates.
    - Display interactive charts and statistics.
    - Upload student data through CSV files.
    - Generate downloadable placement reports.
    """)
    st.header("🛠️ Technologies Used")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🐍 Python\n\nProgramming Language")
    with c2:
        st.info("🤖 Scikit-learn\n\nMachine Learning")
    with c3:
        st.info("📊 Streamlit + Plotly\n\nDashboard and Charts")

    st.header("📥 Input Features")
    st.markdown("""
    - **CGPA:** Academic performance.
    - **Aptitude:** Problem-solving and reasoning score.
    - **Communication:** Communication skill rating.
    - **Internship:** Previous internship experience.
    """)
    st.header("⚙️ Project Modules")
    with st.expander("1. Dashboard"):
        st.write("Displays student counts, placement rate, department charts and performance charts.")
    with st.expander("2. Placement Prediction"):
        st.write("Uses a Random Forest classifier to predict placement outcomes from student input.")
    with st.expander("3. Placement Analytics"):
        st.write("Compares placement rates across departments and summarizes placement statistics.")
    with st.expander("4. Student Performance"):
        st.write("Explores CGPA, aptitude, communication and department-wise performance.")
    with st.expander("5. Model Evaluation"):
        st.write("Displays test accuracy, confusion matrix, classification report and feature importance.")
    with st.expander("6. Career Guidance"):
        st.write("Provides general suggestions for improving academic, aptitude, communication and internship skills.")
    with st.expander("7. Dataset and Upload"):
        st.write("Allows users to explore sample records, upload CSV data and download prediction results.")
    with st.expander("8. Report Generation"):
        st.write("Creates a placement summary and downloadable department report.")

    st.header("🌟 Advantages")
    st.markdown("""
    - Simple user interface.
    - Interactive data visualization.
    - Fast demonstration predictions.
    - CSV upload and download support.
    - Report generation.
    - Useful for learning machine learning concepts.
    """)
    st.header("🚀 Future Enhancements")
    st.markdown("""
    - Connect a real college placement database.
    - Add administrator and student accounts.
    - Add secure user authentication.
    - Improve the model using real placement records.
    - Add salary package prediction.
    - Generate PDF reports.
    """)
    st.header("👩‍💻 Project Information")
    st.write("**Project Title:** PlacePro - Student Placement Prediction")
    st.write("**Domain:** Machine Learning")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Frontend:** Streamlit")
    st.write("**Programming Language:** Python")
    st.write("**Data Visualization:** Plotly")
    st.success("About Project page loaded successfully!")
    st.warning("This application uses synthetic demo data. Its predictions are not actual placement results.")

# ==========================================
# CHATBOT PAGE - WHITE BOX + BLUE TEXT
# ==========================================
elif page == "🤖 Chatbot":
    st.title("🤖 PlacePro Chatbot")
    st.write("Hello! Welcome to PlacePro. 🎓 Ask me questions about student placement, career guidance, aptitude, interviews and this project.")

    knowledge = {
        "hello": "Hello! 👋 Welcome to PlacePro. How can I help you today?",
        "hi": "Hi! 😊 I am the PlacePro chatbot. Ask me anything about student placement.",
        "what is placepro": "PlacePro is a Student Placement Analytics and Prediction project developed using Python, Streamlit and Machine Learning.",
        "what is placement prediction": "Placement prediction uses student information, such as CGPA, aptitude, communication skills and internship experience, to estimate placement outcomes.",
        "which algorithm is used": "This project uses the Random Forest Classifier from the Scikit-learn library.",
        "what is random forest": "Random Forest is a machine learning algorithm that combines multiple decision trees to make predictions.",
        "what is cgpa": "CGPA stands for Cumulative Grade Point Average. It represents a student's overall academic performance.",
        "what is aptitude": "Aptitude measures problem-solving, numerical reasoning and logical thinking skills. Practising questions regularly can help improve your aptitude.",
        "what is communication skill": "Communication skills help you express ideas clearly. You can improve them by practising English, reading aloud and participating in presentations.",
        "how to improve cgpa": "Create a study timetable, revise topics regularly, practise previous question papers and ask teachers for help when you have doubts.",
        "how to prepare for interview": "Prepare a self-introduction, revise your technical subjects, practise common interview questions and learn about the company.",
        "what is internship": "An internship gives students practical experience in a workplace and helps them understand real-world tasks.",
        "what is machine learning": "Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data and make predictions.",
        "what is python": "Python is a beginner-friendly programming language used for web development, data analysis and machine learning.",
        "what is streamlit": "Streamlit is a Python framework used to create interactive web applications and data dashboards.",
        "what is the objective of this project": "The objective is to explore student placement data, analyse academic performance and demonstrate placement prediction using machine learning.",
        "thank you": "You're welcome! 😊 Keep learning and improving your skills.",
        "bye": "Goodbye! 👋 All the best for your studies and placement preparation."
    }

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Hello! 👋 I am PlacePro Chatbot. Ask me a question about your project or placement preparation."
        }]

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask your question here...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        user_question = question.lower().strip()
        answer = None

        for key, value in knowledge.items():
            if key in user_question:
                answer = value
                break

        if answer is None:
            if "placement" in user_question:
                answer = "Placement means getting a job through campus recruitment or another hiring process. Prepare your technical skills, aptitude and interview skills."
            elif "career" in user_question:
                answer = "Choose a career based on your interests and strengths. As a BCA student, you can explore software development, data analytics, web development and testing."
            elif "project" in user_question:
                answer = "PlacePro is a student placement prediction project. It uses Python, Streamlit, Plotly and a Random Forest Classifier. You can explore the project using the sidebar."
            elif "help" in user_question:
                answer = "I can answer questions about PlacePro, machine learning, CGPA, aptitude, internships, interviews and career guidance."
            else:
                answer = "Sorry, I don't know the answer to that question yet. Please try asking about placement, CGPA, aptitude, internships, interviews, Python or machine learning."

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()