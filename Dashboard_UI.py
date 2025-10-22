import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

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

st.title("Bus Captain Safety Dashboard")

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

# Route filter (multiselect)
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

# Apply route filter if any selected
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
# 5. VISUAL 1: Distribution of Total Alarms per Driver
# ===============================================================
st.subheader("Distribution of Total Alarms per Driver")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["total_alarms"], bins=30, kde=True, color="steelblue", ax=ax1)
ax1.set_xlabel("Total Alarms")
ax1.set_ylabel("Number of Drivers")
st.pyplot(fig1)

# ===============================================================
# 6. VISUAL 2: Top Drivers by Selected Metric
# ===============================================================
st.subheader(f"Top {top_n} Drivers by {sort_column}")
top_drivers = filtered_df.sort_values(sort_column, ascending=False).head(top_n)
fig2, ax2 = plt.subplots()
sns.barplot(x=sort_column, y="driver", data=top_drivers, palette="viridis", ax=ax2)
ax2.set_xlabel(sort_column.replace("_", " ").title())
ax2.set_ylabel("Driver ID")
st.pyplot(fig2)

# ===============================================================
# 7. VISUAL 3: Total Alarms by Accident Involvement
# ===============================================================
st.subheader("Total Alarms by Accident Involvement")
fig3, ax3 = plt.subplots()
sns.boxplot(x="has_accident", y="total_alarms", data=filtered_df, palette="pastel", ax=ax3)
ax3.set_xlabel("Has Accident (1 = Yes, 0 = No)")
ax3.set_ylabel("Total Alarms")
st.pyplot(fig3)

# ===============================================================
# 8. VISUAL 4: Average Speed vs Total Alarms
# ===============================================================
if "avg_speed" in filtered_df.columns:
    st.subheader("Average Speed vs Total Alarms")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(
        x="avg_speed", y="total_alarms", hue="has_accident",
        data=filtered_df, palette="Set1", ax=ax4, alpha=0.7
    )
    ax4.set_xlabel("Average Speed (km/h)")
    ax4.set_ylabel("Total Alarms")
    st.pyplot(fig4)

# ===============================================================
# 9. VISUAL 5: Alarm Type Breakdown for Top Drivers (fixed legend)
# ===============================================================
st.subheader(f"Alarm Type Breakdown for Top {top_n} Drivers")

try:
    df_alarm = pd.read_parquet("Cleaned_Mar24_alarms.parquet")

    # Apply route filter to alarm dataset if selected
    if selected_routes and "operation line" in df_alarm.columns:
        df_alarm = df_alarm[df_alarm["operation line"].apply(
            lambda x: any(r in str(x) for r in selected_routes)
        )]

    # Only keep alarms for top drivers
    top_driver_list = top_drivers["driver"].tolist()
    df_top = df_alarm[df_alarm["driver"].isin(top_driver_list)]

    if "alarm type" in df_top.columns:
        alarm_breakdown = (
            df_top.groupby(["driver", "alarm type"])
            .size()
            .reset_index(name="count")
        )

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x="driver", y="count", hue="alarm type",
            data=alarm_breakdown, ax=ax5
        )
        ax5.set_title("Alarm Type Breakdown per Top Driver")
        ax5.set_xlabel("Driver ID")
        ax5.set_ylabel("Alarm Count")
        plt.xticks(rotation=45)
        # Move legend outside to prevent overlap
        ax5.legend(title="Alarm Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig5)

        # Summary table
        st.write("Alarm breakdown summary table:")
        st.dataframe(alarm_breakdown.pivot(index="driver", columns="alarm type", values="count").fillna(0))
    else:
        st.write("No 'alarm type' column found in alarm dataset.")
except FileNotFoundError:
    st.warning("Detailed alarm dataset not found (Cleaned_Mar24_alarms.parquet).")

# ===============================================================
# 10. VISUAL 6: Accident Rate by Route (Fixed and Simplified)
# ===============================================================
if "routes" in df.columns:
    st.subheader("Accident Rate by Individual Route")

    # Expand multi-route entries into individual rows
    expanded_rows = []
    for _, row in df.iterrows():
        if pd.notna(row["routes"]):
            for route in str(row["routes"]).split(","):
                route = route.strip()
                if route:
                    expanded_rows.append({"route": route, "has_accident": row["has_accident"]})
    route_df = pd.DataFrame(expanded_rows)

    # Compute accident rate per route
    route_stats = (
        route_df.groupby("route")["has_accident"]
        .mean()
        .reset_index()
        .sort_values("has_accident", ascending=False)
        .head(15)  # show only top 15 routes
    )

    # Plot
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.barplot(x="has_accident", y="route", data=route_stats, palette="Reds_r", ax=ax6)
    ax6.set_xlabel("Accident Rate (Proportion of Drivers)")
    ax6.set_ylabel("Route")
    ax6.set_title("Top 15 Routes by Accident Rate")
    st.pyplot(fig6)

# ===============================================================
# 11. VISUAL 7: Alarm Type Frequency vs Accident Status (Improved)
# ===============================================================
st.subheader("Alarm Type Frequency: Accident vs Non-Accident Drivers")

try:
    df_alarm = pd.read_parquet("Cleaned_Mar24_alarms.parquet")
    merged_labels = df[["driver", "has_accident"]]
    df_alarm = df_alarm.merge(merged_labels, on="driver", how="left")

    if "alarm type" in df_alarm.columns:
        alarm_compare = (
            df_alarm.groupby(["alarm type", "has_accident"])
            .size()
            .reset_index(name="count")
        )

        # Sort by total alarm count descending for consistent order
        order = (
            alarm_compare.groupby("alarm type")["count"]
            .sum()
            .sort_values(ascending=False)
            .index
        )

        # Horizontal bar chart for better label alignment
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            y="alarm type", x="count", hue="has_accident",
            data=alarm_compare, order=order, ax=ax7, palette="Set2"
        )
        ax7.set_title("Alarm Type Frequency: Accident vs Non-Accident Drivers")
        ax7.set_xlabel("Number of Alarms")
        ax7.set_ylabel("Alarm Type")
        ax7.legend(title="Accident Involvement", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig7)

        # Optional: summary table below
        st.write("Summary of Alarm Frequencies by Accident Status:")
        st.dataframe(
            alarm_compare.pivot(index="alarm type", columns="has_accident", values="count").fillna(0)
        )
except FileNotFoundError:
    st.warning("Cannot load alarm comparison data. Check Cleaned_Mar24_alarms.parquet.")