import pandas as pd

print("Loading Parquet files...")

# ===============================================================
# CONFIGURATION
# ===============================================================
USE_SAMPLE = False    # Change to False for full dataset
ROW_LIMIT = 1000      # Number of rows for sample mode

# ===============================================================
# 1. LOAD DATA
# ===============================================================
if USE_SAMPLE:
    df_alarm = pd.read_parquet("Mar24_alarms.parquet")[:ROW_LIMIT]
    print(f"Loaded sample of {ROW_LIMIT} rows from alarm data.")
else:
    df_alarm = pd.read_parquet("Mar24_alarms.parquet")
    print("Loaded full alarm dataset.")

df_accident = pd.read_parquet("Accidents_2024.parquet")

print("Alarm data shape:", df_alarm.shape)
print("Accident data shape:", df_accident.shape)

# ===============================================================
# 2. BASIC CLEANING
# ===============================================================
df_alarm.columns = df_alarm.columns.str.strip().str.lower()
df_accident.columns = df_accident.columns.str.strip().str.lower()

df_alarm = df_alarm.dropna(how="all")
df_accident = df_accident.dropna(how="all")

# Clean driver column
if "driver" in df_alarm.columns:
    df_alarm = df_alarm.dropna(subset=["driver"])
    df_alarm["driver"] = (
        df_alarm["driver"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    df_alarm = df_alarm[df_alarm["driver"].str.lower() != "nan"]
    df_alarm = df_alarm[df_alarm["driver"].str.strip() != ""]
else:
    print("No 'driver' column found in alarm data.")

# Clean employee number column in accident data
if "employee no" in df_accident.columns:
    df_accident["employee no"] = (
        df_accident["employee no"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
else:
    print("No 'employee no' column found in accident data.")

# ===============================================================
# 3. SUMMARY STATISTICS (ALARMS)
# ===============================================================
if "driver" in df_alarm.columns:
    alarm_counts = df_alarm.groupby("driver").size().reset_index(name="total_alarms")
    print("\nTop 10 drivers with the most alarms:")
    print(alarm_counts.sort_values("total_alarms", ascending=False).head(10))
else:
    alarm_counts = pd.DataFrame()

if "alarm type" in df_alarm.columns:
    alarm_type_counts = df_alarm["alarm type"].value_counts().reset_index()
    alarm_type_counts.columns = ["alarm type", "count"]
    print("\nAlarm type frequency:")
    print(alarm_type_counts)
else:
    alarm_type_counts = pd.DataFrame()

if "speed(km/h)" in df_alarm.columns:
    df_alarm["speed(km/h)"] = pd.to_numeric(df_alarm["speed(km/h)"], errors="coerce")
    avg_speed = df_alarm.groupby("driver")["speed(km/h)"].mean().reset_index(name="avg_speed")
    print("\nAverage speed per driver (first 10):")
    print(avg_speed.head(10))
else:
    avg_speed = pd.DataFrame()

# ===============================================================
# 4. COMBINE WITH ACCIDENT DATA
# ===============================================================
if "employee no" in df_accident.columns:
    df_accident.rename(columns={"employee no": "driver"}, inplace=True)

if not alarm_counts.empty and "driver" in df_accident.columns:
    df_merged = pd.merge(
        alarm_counts,
        df_accident[["driver"]],
        on="driver",
        how="left",
        indicator=True
    )

    df_merged["has_accident"] = df_merged["_merge"].apply(lambda x: 1 if x == "both" else 0)
    df_merged.drop(columns=["_merge"], inplace=True)

    if not avg_speed.empty:
        df_merged = df_merged.merge(avg_speed, on="driver", how="left")

    print("\nSample merged data (first 10 rows):")
    print(df_merged.head(10))
else:
    df_merged = pd.DataFrame()
    print("\nUnable to merge data. Please verify columns.")

# ===============================================================
# 5. ADD ROUTE INFORMATION
# ===============================================================
if "operation line" in df_alarm.columns:
    route_info = (
        df_alarm.groupby("driver")["operation line"]
        .agg(lambda x: ", ".join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={"operation line": "routes"})
    )
    df_merged = df_merged.merge(route_info, on="driver", how="left")
    print("\nAdded route information per driver.")
else:
    print("No 'operation line' column found; skipping route summary.")

# ===============================================================
# 6. SAVE FINAL CLEANED AND MERGED DATA
# ===============================================================
df_alarm.to_parquet("Cleaned_Mar24_alarms.parquet")
df_accident.to_parquet("Cleaned_Accidents_2024.parquet")

if not df_merged.empty:
    if USE_SAMPLE:
        df_merged.to_parquet("Driver_Summary_Combined_Sample.parquet")
    else:
        df_merged.to_parquet("Driver_Summary_Combined.parquet")

print("\nAll cleaning, merging, and route summary steps complete.")
