import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.inspection import permutation_importance

# Set page layout
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    /* Card-like containers */
    .stCard {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #1F77B4;
        font-weight: 700;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.subheader("Employee Attrition Overview")
# add charts, KPIs, etc.
st.markdown('</div>', unsafe_allow_html=True)

# --------------------
# Utility Functions
# --------------------

def split_features(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    cat_cols, num_cols = [], []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    return pre


def metrics_table(y_true, y_pred, y_proba=None) -> pd.DataFrame:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    return pd.DataFrame({
        "Accuracy": [acc],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
        "ROC-AUC": [auc],
    })


def plot_confusion(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["No", "Yes"]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label", title=title)
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_roc(model, X_test, y_test, title: str):
    fig, ax = plt.subplots(figsize=(4, 3))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def add_risk_band(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


# --------------------
# Sidebar – Navigation
# --------------------
st.sidebar.title("📊 HR Attrition Dashboard")
section = st.sidebar.radio(
    "Navigate",
    ("Upload & EDA", "Modeling", "Feature Importance", "Predict (User Input)", "Export / Import"),
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use the same target label values as 'Yes'/'No' for Attrition.")

# --------------------
# Data Ingestion
# --------------------
@st.cache_data(show_spinner=False)
def load_data(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    if upload.name.endswith(".csv"):
        return pd.read_csv(upload)
    elif upload.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel.")
        return pd.DataFrame()

upload = st.sidebar.file_uploader("Upload HR dataset (CSV/Excel)", type=["csv", "xlsx", "xls"])
df = load_data(upload)

if df.empty:
    st.info("Upload a dataset to begin. Expected columns include features like Age, MonthlyIncome, JobRole, Department, OverTime, etc., and a target column e.g. 'Attrition'.")
else:
    st.success(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

# Allow user to choose the target column (default to 'Attrition' if present)
all_cols = list(df.columns) if not df.empty else []
default_target = "Attrition" if "Attrition" in all_cols else (all_cols[-1] if all_cols else None)
target = st.sidebar.selectbox("Target column (binary)", options=all_cols, index=(all_cols.index(default_target) if default_target in all_cols else 0)) if all_cols else None

# --------------------
# Section: Upload & EDA
# --------------------
if section == "Upload & EDA":
    st.title("📥 Upload & Exploratory Data Analysis")

    if df.empty:
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    if target and target in df.columns:
        pos = df[target].astype(str).str.lower().isin(["yes", "1", "true", "attrition", "left"]).sum()
        rate = 100.0 * pos / len(df)
        c4.metric("Positive (Yes) %", f"{rate:.1f}%")

    with st.expander("Missing Values by Column"):
        st.write(df.isna().sum().to_frame("MissingCount").sort_values("MissingCount", ascending=False))

    st.subheader("Numeric Distributions")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        sel_num = st.multiselect("Choose numeric columns", num_cols, default=num_cols[: min(6, len(num_cols))])
        for col in sel_num:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df[col].dropna(), bins=30)
            ax.set_title(f"Distribution – {col}")
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected.")

    st.subheader("Categorical Breakdown vs Target")
    if target in df.columns:
        cat_cols = [c for c in df.columns if c != target and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))]
        if not cat_cols:
            st.info("No categorical columns detected.")
        else:
            sel_cat = st.multiselect("Choose categorical columns", cat_cols, default=cat_cols[: min(4, len(cat_cols))])
            for col in sel_cat:
                tmp = df[[col, target]].copy()
                tmp[target] = tmp[target].astype(str)
                counts = tmp.groupby([col, target]).size().unstack(fill_value=0)
                st.write(f"**{col}**")
                st.bar_chart(counts)

    st.subheader("Correlation (Numeric Only)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=90)
        ax.set_yticks(range(len(num_cols)))
        ax.set_yticklabels(num_cols)
        fig.colorbar(cax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for correlation heatmap.")

# --------------------
# Section: Modeling
# --------------------
elif section == "Modeling":
    st.title("🤖 Modeling: Train & Evaluate")
    if df.empty or target is None:
        st.stop()

    # Map target to 1/0
    y = df[target].astype(str).str.lower().isin(["yes", "1", "true", "attrition", "left"]).astype(int)
    X = df.drop(columns=[target])

    num_cols, cat_cols = split_features(df, target)
    pre = build_preprocessor(num_cols, cat_cols)

    # Controls
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 1, 9999, 42, 1)
    threshold = st.slider("Classification Threshold (on positive prob)", 0.1, 0.9, 0.5, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Logistic Regression")
        logreg = Pipeline(steps=[("pre", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        grid_lr = GridSearchCV(logreg, {"model__C": [0.1, 1.0, 10.0]}, scoring="roc_auc", cv=5, n_jobs=-1)
        grid_lr.fit(X_train, y_train)
        lr_best = grid_lr.best_estimator_
        y_proba_lr = lr_best.predict_proba(X_test)[:, 1]
        y_pred_lr = (y_proba_lr >= threshold).astype(int)
        st.write("Best C:", grid_lr.best_params_["model__C"])
        st.dataframe(metrics_table(y_test, y_pred_lr, y_proba_lr))
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        st.pyplot(plot_confusion(cm_lr, "LogReg – Confusion"), use_container_width=False)
        st.pyplot(plot_roc(lr_best, X_test, y_test, "LogReg – ROC"), use_container_width=False)

    with colB:
        st.markdown("### Decision Tree")
        tree = Pipeline(steps=[("pre", pre), ("model", DecisionTreeClassifier(class_weight="balanced", random_state=random_state))])
        grid_tr = GridSearchCV(tree, {"model__max_depth": [3, 5, 7, 9, None], "model__min_samples_split": [2, 10, 20]}, scoring="roc_auc", cv=5, n_jobs=-1)
        grid_tr.fit(X_train, y_train)
        tr_best = grid_tr.best_estimator_
        y_proba_tr = tr_best.predict_proba(X_test)[:, 1]
        y_pred_tr = (y_proba_tr >= threshold).astype(int)
        st.write("Best Params:", grid_tr.best_params_)
        st.dataframe(metrics_table(y_test, y_pred_tr, y_proba_tr))
        cm_tr = confusion_matrix(y_test, y_pred_tr)
        st.pyplot(plot_confusion(cm_tr, "Tree – Confusion"), use_container_width=False)
        st.pyplot(plot_roc(tr_best, X_test, y_test, "Tree – ROC"), use_container_width=False)

    # Select best
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    auc_tr = roc_auc_score(y_test, y_proba_tr)
    best_model = lr_best if auc_lr >= auc_tr else tr_best
    best_name = "LogisticRegression" if auc_lr >= auc_tr else "DecisionTree"
    st.success(f"Best by ROC-AUC: {best_name}  (LR={auc_lr:.3f} | Tree={auc_tr:.3f})")

    # Store for other pages
    st.session_state["best_model"] = best_model
    st.session_state["best_name"] = best_name
    st.session_state["X_columns"] = X.columns.tolist()
    st.session_state["num_cols"] = num_cols
    st.session_state["cat_cols"] = cat_cols

# --------------------
# Section: Feature Importance
# --------------------
elif section == "Feature Importance":
    st.title("🧠 Feature Importance & Drivers")

    if "best_model" not in st.session_state or df.empty or target is None:
        st.info("Train a model first in the 'Modeling' section.")
        st.stop()

    model = st.session_state["best_model"]
    X = df.drop(columns=[target])
    y = df[target].astype(str).str.lower().isin(["yes", "1", "true", "attrition", "left"]).astype(int)

    # Permutation importance on a holdout split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with st.spinner("Computing permutation importances..."):
        r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # Extract names after preprocessing
    pre: ColumnTransformer = model.named_steps["pre"]
    num_cols = st.session_state["num_cols"]
    cat_cols = st.session_state["cat_cols"]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_names)

    importances = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
    top_k = st.slider("Top K features", 5, min(30, len(importances)), min(10, len(importances)))
    st.bar_chart(importances.head(top_k))

    st.subheader("Business Takeaways (Auto-Generated)")
    bullets = []
    for name, val in importances.head(8).items():
        if "JobSatisfaction" in name:
            bullets.append("Lower JobSatisfaction correlates with higher attrition risk.")
        if "MonthlyIncome" in name:
            bullets.append("Lower MonthlyIncome bands are linked to higher risk.")
        if "DistanceFromHome" in name:
            bullets.append("Employees living farther tend to churn more; consider hybrid policy.")
        if "OverTime" in name:
            bullets.append("Frequent overtime may increase attrition likelihood.")
        if "YearsAtCompany" in name:
            bullets.append("Early-tenure employees (1–3 years) may be more at risk.")
    if not bullets:
        bullets = ["Focus interventions on the top features highlighted in the chart above."]
    for b in bullets:
        st.write("• ", b)

# --------------------
# Section: Predict (User Input)
# --------------------
elif section == "Predict (User Input)":
    st.title("🧾 Predict Attrition – Employee Form")

    if "best_model" not in st.session_state:
        st.info("Please train a model in the 'Modeling' section first.")
        st.stop()

    model = st.session_state["best_model"]

    if df.empty or target is None:
        st.stop()

    X = df.drop(columns=[target])
    # Build dynamic form from training data metadata
    num_cols = st.session_state.get("num_cols", [])
    cat_cols = st.session_state.get("cat_cols", [])

    with st.form("emp_form"):
        st.write("Enter employee details:")
        inputs = {}
        cnum, ccat = st.columns(2)

        with cnum:
            for col in num_cols:
                col_min = float(np.nanmin(df[col])) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                col_max = float(np.nanmax(df[col])) if pd.api.types.is_numeric_dtype(df[col]) else 100.0
                default = float(np.nanmedian(df[col])) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                val = st.number_input(f"{col}", value=default, min_value=col_min, max_value=col_max, step=1.0)
                inputs[col] = val

        with ccat:
            for col in cat_cols:
                choices = sorted([str(x) for x in df[col].dropna().unique().tolist()])
                if not choices:
                    choices = ["Unknown"]
                val = st.selectbox(f"{col}", choices)
                inputs[col] = val

        pred_threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        # Build single-row X with same columns
        x_df = pd.DataFrame([inputs])
        # Predict proba via pipeline (preprocessor handles encode/scale)
        proba = float(model.predict_proba(x_df)[:, 1][0])
        pred = int(proba >= pred_threshold)
        band = add_risk_band(proba)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "Yes" if pred == 1 else "No")
        c2.metric("Probability of Leaving", f"{proba:.2f}")
        c3.metric("Risk Band", band)

        st.subheader("Recommendations")
        recs = []
        if band == "High":
            recs.append("Immediate retention conversation; review compensation/role fit.")
            recs.append("Assign mentor; craft growth plan within 30 days.")
        if "OverTime" in inputs and str(inputs["OverTime"]).lower() == "yes":
            recs.append("Monitor workload; reduce overtime where possible.")
        if "JobSatisfaction" in inputs and float(inputs["JobSatisfaction"]) <= 2:
            recs.append("Engagement initiatives: manager 1:1s, recognition program, L&D budget.")
        if "DistanceFromHome" in inputs and float(inputs["DistanceFromHome"]) >= np.nanmedian(df["DistanceFromHome"]) if "DistanceFromHome" in df.columns else False:
            recs.append("Consider remote/hybrid arrangement to ease commute burden.")
        if not recs:
            recs = ["Focus on top drivers from the Feature Importance page; tailor actions to employee context."]
        for r in recs:
            st.write("• ", r)

# --------------------
# Section: Export / Import
# --------------------
elif section == "Export / Import":
    st.title("📦 Export / Import Model")

    if "best_model" in st.session_state:
        model = st.session_state["best_model"]
        best_name = st.session_state.get("best_name", "Model")
        buf = io.BytesIO()
        joblib.dump(model, buf)
        st.download_button(
            label=f"Download trained pipeline ({best_name})",
            data=buf.getvalue(),
            file_name=f"best_model_{best_name}.joblib",
            mime="application/octet-stream",
        )
    else:
        st.info("Train a model in the 'Modeling' section to enable export.")

    st.subheader("Load a Saved Model")
    model_file = st.file_uploader("Upload joblib model", type=["joblib", "pkl"])
    if model_file is not None:
        try:
            model = joblib.load(model_file)
            st.session_state["best_model"] = model
            st.success("Model loaded and ready for predictions.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.caption("HR Analytics — Attrition Prediction • Built with Streamlit, scikit-learn, and love ✨")