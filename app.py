import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🚀",
    layout="wide"
)

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/model_columns.pkl")

# -----------------------
# CUSTOM STYLE
# -----------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}
.big-title {
    font-size:42px;
    font-weight:800;
}
.subtitle {
    color:#9ca3af;
    font-size:18px;
}
.block {
    padding:18px;
    border-radius:16px;
    background:#111827;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown('<p class="big-title">🚀 Customer Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enterprise churn analytics, customer scoring, retention actions.</p>', unsafe_allow_html=True)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("⚙️ Inputs")

tenure = st.sidebar.slider("Tenure Months", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 80)
total = st.sidebar.number_input("Total Charges", 0.0, 100000.0, 2500.0)

predict_btn = st.sidebar.button("🚀 Predict")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("📂 Upload CSV for Bulk Prediction", type=["csv"])

# -----------------------
# KPI ROW
# -----------------------
k1, k2, k3 = st.columns(3)
k1.metric("Tenure", f"{tenure} Months")
k2.metric("Monthly Charge", f"${monthly}")
k3.metric("Lifetime Value", f"${total:,.0f}")

# -----------------------
# TABS
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Prediction", "📈 Analytics", "📌 Actions", "📂 Bulk Scoring"]
)

# ===================================================
# SINGLE PREDICTION
# ===================================================
if predict_btn:

    df = pd.DataFrame([{
        "Tenure Months": tenure,
        "Monthly Charges": monthly,
        "Total Charges": total
    }])

    df["AvgMonthlyValue"] = df["Total Charges"] / (df["Tenure Months"] + 1)
    df["IsLongTerm"] = (df["Tenure Months"] > 24).astype(int)
    df["HighCharges"] = (df["Monthly Charges"] > 70).astype(int)

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])

    risk = int(proba * 100)

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Live Churn Risk")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 35], "color": "green"},
                    {"range": [35, 65], "color": "orange"},
                    {"range": [65, 100], "color": "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if pred == 1:
            st.error(f"⚠️ High Churn Risk ({risk}%)")
        else:
            st.success(f"✅ Likely to Stay ({100-risk}%)")

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("Customer Analytics")

        chart_df = pd.DataFrame({
            "Metric": ["Tenure", "Monthly Charge", "Avg Monthly Value"],
            "Value": [
                tenure,
                monthly,
                total / (tenure + 1)
            ]
        })

        fig2 = px.bar(chart_df, x="Metric", y="Value", title="Customer Metrics")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("Business Actions")

        if pred == 1:
            st.markdown("""
### Retention Plan
- Offer discount immediately  
- Send support callback  
- Push annual contract migration  
- Priority retention outreach  
""")
        else:
            st.markdown("""
### Growth Plan
- Upsell premium plan  
- Offer bundled products  
- Loyalty reward  
- Referral incentives  
""")

# ===================================================
# BULK CSV
# ===================================================
with tab4:
    st.subheader("Bulk Customer Scoring")

    if uploaded is not None:
        bulk = pd.read_csv(uploaded)

        if {"Tenure Months","Monthly Charges","Total Charges"}.issubset(bulk.columns):

            bulk["AvgMonthlyValue"] = bulk["Total Charges"] / (bulk["Tenure Months"] + 1)
            bulk["IsLongTerm"] = (bulk["Tenure Months"] > 24).astype(int)
            bulk["HighCharges"] = (bulk["Monthly Charges"] > 70).astype(int)

            bulk_model = pd.get_dummies(bulk)
            bulk_model = bulk_model.reindex(columns=columns, fill_value=0)

            bulk["Prediction"] = model.predict(bulk_model)
            bulk["Churn Probability"] = model.predict_proba(bulk_model)[:,1]

            st.dataframe(bulk.head(20), use_container_width=True)

            csv = bulk.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Predictions CSV",
                csv,
                "bulk_predictions.csv",
                "text/csv"
            )
        else:
            st.warning("CSV must contain: Tenure Months, Monthly Charges, Total Charges")
    else:
        st.info("Upload CSV file to score multiple customers.")