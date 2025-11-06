import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
# 3. ENHANCED FEATURE ENGINEERING
# ===============================================================
def enhanced_feature_engineering(df_alarm, df_merged):
    """Enhanced feature engineering for machine learning"""
    print("Performing enhanced feature engineering...")
    
    # Create detailed alarm type features
    if "alarm type" in df_alarm.columns:
        # Alarm type frequency per driver
        alarm_type_dummies = pd.get_dummies(df_alarm[['driver', 'alarm type']], 
                                          columns=['alarm type'])
        alarm_type_summary = alarm_type_dummies.groupby('driver').sum().add_prefix('alarm_')
        
        # Merge with existing data
        df_enhanced = df_merged.merge(alarm_type_summary, on='driver', how='left')
        
        # Fill missing alarm types with 0
        alarm_columns = [col for col in df_enhanced.columns if col.startswith('alarm_')]
        df_enhanced[alarm_columns] = df_enhanced[alarm_columns].fillna(0)
        
        print(f"Added {len(alarm_columns)} alarm type features")
        
        # Calculate alarm severity score (weighted sum)
        severity_weights = {
            'harsh acceleration': 1.0,
            'harsh braking': 1.2,
            'speeding': 1.5,
            'swaying': 0.8,
            'sharp cornering': 1.1,
            'fatigue alert': 2.0,
            'yawn': 1.5,
            'headway monitoring alert': 1.3,
            'forward collision warning': 2.5,
            'pedestrian collision alert': 3.0
        }
        
        # Calculate severity score for each driver
        severity_scores = []
        for driver in df_enhanced['driver']:
            score = 0
            driver_alarms = df_alarm[df_alarm['driver'] == driver]
            for alarm_type in driver_alarms['alarm type'].unique():
                count = len(driver_alarms[driver_alarms['alarm type'] == alarm_type])
                weight = severity_weights.get(alarm_type, 1.0)
                score += count * weight
            severity_scores.append(score)
        
        df_enhanced['severity_score'] = severity_scores
        
    else:
        print("No alarm type data available for feature engineering")
        df_enhanced = df_merged
        df_enhanced['severity_score'] = 0
    
    # Add time-based features if timestamp is available
    if 'timestamp' in df_alarm.columns:
        try:
            df_alarm['timestamp'] = pd.to_datetime(df_alarm['timestamp'])
            # Alarms per hour of day
            df_alarm['hour'] = df_alarm['timestamp'].dt.hour
            hour_summary = df_alarm.groupby(['driver', 'hour']).size().unstack(fill_value=0)
            hour_summary.columns = [f'alarms_hour_{h}' for h in hour_summary.columns]
            df_enhanced = df_enhanced.merge(hour_summary, on='driver', how='left')
            print("Added time-based features")
        except:
            print("Could not process timestamp data")
    
    # Add driving pattern features
    if 'speed(km/h)' in df_alarm.columns:
        df_alarm['speed(km/h)'] = pd.to_numeric(df_alarm['speed(km/h)'], errors='coerce')
        speed_stats = df_alarm.groupby('driver')['speed(km/h)'].agg([
            'mean', 'std', 'max', 'min'
        ]).fillna(0)
        speed_stats.columns = ['speed_mean', 'speed_std', 'speed_max', 'speed_min']
        df_enhanced = df_enhanced.merge(speed_stats, on='driver', how='left')
        print("Added speed statistics features")
    
    return df_enhanced

# ===============================================================
# 4. SUMMARY STATISTICS (ALARMS)
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
# 5. COMBINE WITH ACCIDENT DATA
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
# 6. ADD ROUTE INFORMATION
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
# 7. APPLY ENHANCED FEATURE ENGINEERING
# ===============================================================
if not df_merged.empty:
    df_merged = enhanced_feature_engineering(df_alarm, df_merged)
    print("Enhanced feature engineering completed.")
    
    # Display new features
    print("\nNew features added:")
    new_features = [col for col in df_merged.columns if col.startswith('alarm_') or col in ['severity_score', 'speed_mean', 'speed_std']]
    for feature in new_features:
        if feature in df_merged.columns:
            print(f"  - {feature}: range [{df_merged[feature].min():.2f}, {df_merged[feature].max():.2f}]")

# ===============================================================
# 8. MACHINE LEARNING PREPARATION
# ===============================================================
def prepare_ml_data(df):
    """Prepare data for machine learning"""
    print("\nPreparing data for machine learning...")
    
    # Select features for ML
    feature_columns = ['total_alarms', 'severity_score']
    
    # Add speed features if available
    if 'speed_mean' in df.columns:
        feature_columns.extend(['speed_mean', 'speed_std'])
    
    # Add alarm type features
    alarm_features = [col for col in df.columns if col.startswith('alarm_')]
    feature_columns.extend(alarm_features[:10])  # Use top 10 alarm types
    
    # Ensure we only use columns that exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Prepare feature matrix and target
    X = df[feature_columns].copy()
    y = df['has_accident']
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Using {len(feature_columns)} features for ML:")
    for feature in feature_columns:
        print(f"  - {feature}")
    
    return X, y, feature_columns

def train_risk_model(X, y):
    """Train a Random Forest model for risk prediction"""
    print("\nTraining Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - AUC Score: {auc_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, X_test, y_test, feature_importance

def predict_risk_scores(df, model, feature_columns):
    """Predict risk scores for all drivers"""
    print("\nPredicting risk scores...")
    
    # Prepare features
    X = df[feature_columns].copy().fillna(0)
    
    # Predict probabilities
    risk_scores = model.predict_proba(X)[:, 1]
    
    # Add to dataframe
    df_result = df.copy()
    df_result['risk_score'] = risk_scores
    
    # Categorize risk levels
    df_result['risk_category'] = pd.cut(
        risk_scores,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )
    
    # Display risk distribution
    risk_distribution = df_result['risk_category'].value_counts()
    print("\nRisk Category Distribution:")
    for category, count in risk_distribution.items():
        percentage = (count / len(df_result)) * 100
        print(f"  - {category}: {count} drivers ({percentage:.1f}%)")
    
    # Show top risky drivers
    top_risky = df_result.nlargest(10, 'risk_score')[['driver', 'total_alarms', 'has_accident', 'risk_score', 'risk_category']]
    print("\nTop 10 Most At-Risk Drivers:")
    print(top_risky.to_string(index=False))
    
    return df_result

# Run ML if we have sufficient data
if not df_merged.empty and len(df_merged) > 100:
    try:
        X, y, feature_columns = prepare_ml_data(df_merged)
        
        if len(feature_columns) >= 2:  # Need at least 2 features
            model, X_test, y_test, feature_importance = train_risk_model(X, y)
            df_merged = predict_risk_scores(df_merged, model, feature_columns)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title('Top 10 Feature Importance for Risk Prediction')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance plot saved as 'feature_importance.png'")
            
        else:
            print("Insufficient features for machine learning")
    except Exception as e:
        print(f"Machine learning failed: {e}")
else:
    print("Insufficient data for machine learning")

# ===============================================================
# 9. SAVE FINAL CLEANED AND MERGED DATA
# ===============================================================
df_alarm.to_parquet("Cleaned_Mar24_alarms.parquet")
df_accident.to_parquet("Cleaned_Accidents_2024.parquet")

if not df_merged.empty:
    if USE_SAMPLE:
        df_merged.to_parquet("Driver_Summary_Combined_Sample.parquet")
    else:
        df_merged.to_parquet("Driver_Summary_Combined.parquet")
    
    # Also save a version with risk scores if ML was run
    if 'risk_score' in df_merged.columns:
        df_merged.to_parquet("Driver_Summary_With_Risk_Scores.parquet")
        print("Saved risk assessment data to 'Driver_Summary_With_Risk_Scores.parquet'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*60)
print(f"Total drivers analyzed: {len(df_merged) if not df_merged.empty else 0}")
print(f"Drivers with accidents: {df_merged['has_accident'].sum() if not df_merged.empty else 0}")
print(f"Total alarms processed: {len(df_alarm)}")
print(f"Accident records: {len(df_accident)}")

if not df_merged.empty:
    if 'risk_category' in df_merged.columns:
        risk_summary = df_merged['risk_category'].value_counts()
        print("\nFINAL RISK ASSESSMENT:")
        for category in ['High Risk', 'Medium Risk', 'Low Risk']:
            if category in risk_summary:
                count = risk_summary[category]
                percentage = (count / len(df_merged)) * 100
                print(f"  - {category}: {count} drivers ({percentage:.1f}%)")

print("\nAll cleaning, merging, feature engineering, and analysis steps complete.")
print("Files saved:")
print("  - Cleaned_Mar24_alarms.parquet")
print("  - Cleaned_Accidents_2024.parquet")
print("  - Driver_Summary_Combined.parquet")
if 'risk_score' in df_merged.columns:
    print("  - Driver_Summary_With_Risk_Scores.parquet")
    print("  - feature_importance.png")