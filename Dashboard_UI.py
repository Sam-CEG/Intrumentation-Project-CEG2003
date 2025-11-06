import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# ===============================================================
# 0. PAGE SETUP
# ===============================================================
st.set_page_config(
    page_title="Bus Captain Safety Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# 1. LOAD DATA
# ===============================================================
USE_SAMPLE = False
if USE_SAMPLE:
    merged_file = "Driver_Summary_Combined_Sample.parquet"
else:
    merged_file = "Driver_Summary_Combined.parquet"

st.title("🚌 Bus Captain Safety Dashboard")

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

df = load_data(merged_file)
st.write("Data loaded:", df.shape, "rows")

# ===============================================================
# 2. SIDEBAR FILTERS
# ===============================================================
st.sidebar.header("Filter Options")

accident_filter = st.sidebar.radio(
    "Filter by Accident Involvement",
    options=["All Drivers", "With Accident", "Without Accident"],
    index=0
)

# Alarm range filter
min_alarm, max_alarm = int(df["total_alarms"].min()), int(df["total_alarms"].max())
alarm_range = st.sidebar.slider(
    "Select Alarm Count Range",
    min_value=min_alarm,
    max_value=max_alarm,
    value=(min_alarm, max_alarm)
)

# Route filter
if "routes" in df.columns:
    all_routes = sorted(
        {r.strip() for sublist in df["routes"].dropna().str.split(",") for r in sublist}
    )
    selected_routes = st.sidebar.multiselect(
        "Filter by Operation Line(s)", options=all_routes, default=[]
    )
else:
    selected_routes = []

# Sort and Top N filter
sort_column = st.sidebar.selectbox(
    "Sort Drivers By",
    options=["total_alarms", "avg_speed"]
)
top_n = st.sidebar.slider("Number of Top Drivers to Display", 5, 30, 10)

# ===============================================================
# 3. APPLY FILTERS
# ===============================================================
filtered_df = df[(df["total_alarms"] >= alarm_range[0]) & (df["total_alarms"] <= alarm_range[1])]

if accident_filter == "With Accident":
    filtered_df = filtered_df[filtered_df["has_accident"] == 1]
elif accident_filter == "Without Accident":
    filtered_df = filtered_df[filtered_df["has_accident"] == 0]

if selected_routes and "routes" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["routes"].apply(
        lambda x: any(r in str(x).split(",") for r in selected_routes)
    )]

# ===============================================================
# 4. SUMMARY METRICS
# ===============================================================
col1, col2, col3 = st.columns(3)
col1.metric("Total Drivers", len(filtered_df))
col2.metric("Drivers with Accident", int(filtered_df["has_accident"].sum()))
col3.metric("Average Alarms per Driver", round(filtered_df["total_alarms"].mean(), 2))
st.markdown("---")

# ===============================================================
# 5–11. VISUALIZATIONS (original)
# ===============================================================
st.subheader("Distribution of Total Alarms per Driver")
sns.set(style="whitegrid")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df["total_alarms"], bins=30, kde=True, color="steelblue", ax=ax1)
st.pyplot(fig1)

st.subheader(f"Top {top_n} Drivers by {sort_column}")
top_drivers = filtered_df.sort_values(sort_column, ascending=False).head(top_n)
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=sort_column, y="driver", data=top_drivers, palette="viridis", ax=ax2)
st.pyplot(fig2)

st.subheader("Total Alarms by Accident Involvement")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(x="has_accident", y="total_alarms", data=filtered_df, palette="pastel", ax=ax3)
st.pyplot(fig3)

if "avg_speed" in filtered_df.columns:
    st.subheader("Average Speed vs Total Alarms")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x="avg_speed", y="total_alarms", hue="has_accident",
        data=filtered_df, palette="Set1", ax=ax4, alpha=0.7
    )
    st.pyplot(fig4)

# ===============================================================
# 12. MACHINE LEARNING RESULTS (All Models Integrated)
# ===============================================================
st.markdown("---")
st.header("🤖 Machine Learning: At-Risk Bus Captain Identification")

def safe_load_parquet(file):
    try:
        return pd.read_parquet(file)
    except:
        return pd.DataFrame()

risk_df = safe_load_parquet("driver_risk_assessment.parquet")
fatigue_df = safe_load_parquet("driver_fatigue_risk.parquet")
speed_df = safe_load_parquet("driver_speeding_risk.parquet")
combined_df = safe_load_parquet("driver_combined_risk.parquet")

# ===============================================================
# 12.1 Filter by Risk Level (High / Medium / Low)
# ===============================================================
if not risk_df.empty and "risk_category" in risk_df.columns:
    st.sidebar.subheader("Risk Category Filter")
    selected_risk_levels = st.sidebar.multiselect(
        "Select Risk Category",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    risk_df = risk_df[risk_df["risk_category"].isin(selected_risk_levels)]

# ===============================================================
# 12.2 Accident Risk Model
# ===============================================================
st.subheader("🚧 Accident Risk Model")
if not risk_df.empty:
    st.dataframe(
        risk_df[["driver", "risk_score", "risk_category", "has_accident"]]
        .sort_values("risk_score", ascending=False)
        .head(20)
        .style.background_gradient(subset=["risk_score"], cmap="Reds"),
        use_container_width=True
    )
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    sns.histplot(risk_df["risk_score"], bins=30, kde=True, color="red", ax=ax_acc)
    ax_acc.set_title("Accident Risk Score Distribution")
    st.pyplot(fig_acc)
else:
    st.info("Accident risk results not found.")

# ===============================================================
# 12.3 Fatigue Risk Model
# ===============================================================
st.subheader("😴 Fatigue Risk Model")
if not fatigue_df.empty:
    st.dataframe(
        fatigue_df.sort_values("fatigue_probability", ascending=False)
        .head(20)
        .style.background_gradient(subset=["fatigue_probability"], cmap="Oranges"),
        use_container_width=True
    )
    fig_fat, ax_fat = plt.subplots(figsize=(8, 5))
    sns.histplot(fatigue_df["fatigue_probability"], bins=30, kde=True, color="orange", ax=ax_fat)
    ax_fat.set_title("Fatigue Probability Distribution")
    st.pyplot(fig_fat)
else:
    st.info("Fatigue model data not found.")

# ===============================================================
# 12.4 Speeding Risk Model
# ===============================================================
st.subheader("💨 Speeding Risk Model")
if not speed_df.empty:
    st.dataframe(
        speed_df.sort_values("speeding_probability", ascending=False)
        .head(20)
        .style.background_gradient(subset=["speeding_probability"], cmap="Blues"),
        use_container_width=True
    )
    fig_spd, ax_spd = plt.subplots(figsize=(8, 5))
    sns.histplot(speed_df["speeding_probability"], bins=30, kde=True, color="skyblue", ax=ax_spd)
    ax_spd.set_title("Speeding Probability Distribution")
    st.pyplot(fig_spd)
else:
    st.info("Speeding model data not found.")

# ===============================================================
# 12.5 Combined Overall Risk
# ===============================================================
st.subheader("🧠 Combined Overall Risk Index")
if not combined_df.empty:
    st.dataframe(
        combined_df[["driver", "risk_score", "fatigue_probability", "speeding_probability", "overall_risk_index"]]
        .sort_values("overall_risk_index", ascending=False)
        .head(20)
        .style.background_gradient(subset=["overall_risk_index"], cmap="Purples"),
        use_container_width=True
    )
    fig_comb, ax_comb = plt.subplots(figsize=(8, 5))
    sns.histplot(combined_df["overall_risk_index"], bins=30, kde=True, color="purple", ax=ax_comb)
    ax_comb.set_title("Overall Risk Index Distribution")
    st.pyplot(fig_comb)
else:
    st.info("Combined risk file not found.")

# ===============================================================
# 13. MODEL PERFORMANCE COMPARISON (Highlight Best)
# ===============================================================
st.markdown("---")
st.header("📊 Model Performance Comparison (All Algorithms)")

if os.path.exists("model_comparison_summary.csv"):
    df_compare = pd.read_csv("model_comparison_summary.csv")

    st.write("### AUC Performance for Each Model")

    if "Is_Best" in df_compare.columns:
        df_compare["Is_Best"] = df_compare["Is_Best"].astype(bool)

        def highlight_best(row):
            return [
                "background-color: gold; font-weight: bold" if row["Is_Best"] else "" for _ in row
            ]

        styled = df_compare.style.apply(highlight_best, axis=1).background_gradient(
            subset=["AUC"], cmap="YlGn"
        )
        st.dataframe(styled, use_container_width=True)

        st.subheader("AUC Performance by Task and Model")
        fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df_compare, x="Task", y="AUC", hue="Model", ax=ax_comp)
        ax_comp.set_title("Model AUC Comparison Across Tasks")
        st.pyplot(fig_comp)
    else:
        st.warning("'Is_Best' column not found in model_comparison_summary.csv.")
else:
    st.info("Model comparison summary not found. Run Machine_Learning.py first.")

st.caption("Bus Captain Safety Dashboard — Multi-Model Analytics © 2025")
