import pandas as pd

print("Loading Excel files...")

# Load both parts of the March 2024 alarm data
df1 = pd.read_excel("Mar24_part1.xlsx", engine="openpyxl")
df2 = pd.read_excel("Mar24_part2.xlsx", engine="openpyxl")

# Combine both parts into a single DataFrame
df_alarm = pd.concat([df1, df2], ignore_index=True)
print("Alarm data loaded and merged:", df_alarm.shape)

# Convert all columns to string type to prevent type errors during Parquet save
df_alarm = df_alarm.astype(str)

# Save to Parquet format for faster loading later
df_alarm.to_parquet("Mar24_alarms.parquet")
print("Saved Mar24_alarms.parquet")

# Load the traffic accident data
df_accident = pd.read_excel("Traffic Accident Database 2024.xlsx", engine="openpyxl")
print("Accident data loaded:", df_accident.shape)

# Convert all columns to string before saving
df_accident = df_accident.astype(str)

# Save to Parquet format
df_accident.to_parquet("Accidents_2024.parquet")
print("Saved Accidents_2024.parquet")

print("All done.")
