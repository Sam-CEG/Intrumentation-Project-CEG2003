import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ===============================================================
# 0️⃣ Utility: Compare multiple ML models on a single dataset
# ===============================================================
def compare_models(X, y, model_name="Model"):
    """
    Trains 4 different models on the same dataset and compares their AUC performance.
    Returns:
        df_results: DataFrame of all results (with Best flag)
        best_model: trained sklearn model
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        results.append({
            "Task": model_name,
            "Model": name,
            "AUC": round(auc, 3)
        })
        print(f"{name} ({model_name}) → AUC: {auc:.3f}")

    # Identify best model
    df_results = pd.DataFrame(results).sort_values("AUC", ascending=False)
    df_results["Is_Best"] = df_results["AUC"] == df_results["AUC"].max()

    best_row = df_results.iloc[0]
    best_model_name = best_row["Model"]
    print(f"🏆 Best {model_name} model: {best_model_name} (AUC={best_row['AUC']:.3f})")

    # Retrain best model on all data
    best_model = models[best_model_name]
    best_model.fit(X, y)

    return df_results, best_model



# ===============================================================
# 1️⃣ Accident Risk Prediction
# ===============================================================
def train_accident_model(df):
    print("\n=== Training Accident Risk Model ===")
    features = ["total_alarms", "avg_speed"]
    if "routes" in df.columns:
        df["num_routes"] = df["routes"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)
        features.append("num_routes")

    X = df[features].fillna(0)
    y = df["has_accident"]

    results, best_model = compare_models(X, y, "Accident Risk")
    df["risk_score"] = best_model.predict_proba(X)[:, 1]
    df["risk_category"] = pd.cut(df["risk_score"], bins=[0, 0.3, 0.7, 1], labels=["Low", "Medium", "High"])
    df.to_parquet("driver_risk_assessment.parquet", index=False)
    print("✅ Saved → driver_risk_assessment.parquet")
    return df, results


# ===============================================================
# 2️⃣ Fatigue Risk Prediction
# ===============================================================
def train_fatigue_model(df_alarm):
    print("\n=== Training Fatigue Risk Model ===")
    df_alarm.columns = [c.lower().strip() for c in df_alarm.columns]
    for c in ["speed(km/h)", "alarm duration"]:
        df_alarm[c] = pd.to_numeric(df_alarm[c], errors="coerce")

    df_alarm["is_fatigue_alarm"] = df_alarm["alarm type"].str.lower().str.contains("fatigue|yawn", na=False)
    df_alarm["hour"] = pd.to_datetime(df_alarm["alarm start time"], errors="coerce").dt.hour
    df_alarm["is_night"] = df_alarm["hour"].between(22, 23) | df_alarm["hour"].between(0, 5)

    df_driver = (
        df_alarm.groupby("driver").agg({
            "speed(km/h)": "mean",
            "alarm duration": "mean",
            "is_fatigue_alarm": "sum",
            "is_night": "mean"
        }).fillna(0).reset_index()
    )
    df_driver["fatigue_risk"] = (df_driver["is_fatigue_alarm"] > 5).astype(int)

    features = ["speed(km/h)", "alarm duration", "is_night"]
    X = df_driver[features]
    y = df_driver["fatigue_risk"]

    results, best_model = compare_models(X, y, "Fatigue Risk")
    df_driver["fatigue_probability"] = best_model.predict_proba(X)[:, 1]
    df_driver[["driver", "fatigue_probability"]].to_parquet("driver_fatigue_risk.parquet", index=False)
    print("✅ Saved → driver_fatigue_risk.parquet")
    return df_driver, results


# ===============================================================
# 3️⃣ Speeding Risk Prediction
# ===============================================================
def train_speeding_model(df_alarm):
    print("\n=== Training Speeding Risk Model ===")
    df_alarm.columns = [c.lower().strip() for c in df_alarm.columns]
    df_alarm["is_speeding_alarm"] = df_alarm["alarm type"].str.lower().str.contains("speed", na=False)
    df_alarm["speed(km/h)"] = pd.to_numeric(df_alarm["speed(km/h)"], errors="coerce")

    df_driver = (
        df_alarm.groupby("driver").agg({
            "speed(km/h)": ["mean", "std", "max"],
            "is_speeding_alarm": "sum"
        }).fillna(0)
    )
    df_driver.columns = ["avg_speed", "speed_std", "max_speed", "speeding_count"]
    df_driver.reset_index(inplace=True)
    df_driver["speeding_risk"] = (df_driver["speeding_count"] > 3).astype(int)

    features = ["avg_speed", "speed_std", "max_speed"]
    X = df_driver[features]
    y = df_driver["speeding_risk"]

    results, best_model = compare_models(X, y, "Speeding Risk")
    df_driver["speeding_probability"] = best_model.predict_proba(X)[:, 1]
    df_driver[["driver", "speeding_probability"]].to_parquet("driver_speeding_risk.parquet", index=False)
    print("✅ Saved → driver_speeding_risk.parquet")
    return df_driver, results


# ===============================================================
# 4️⃣ Main Pipeline
# ===============================================================
def run_machine_learning_analysis():
    print("🚦 Starting ML Analysis (Learning Project Mode)")
    print("=" * 60)

    # 1️⃣ Driver Summary Data
    df_driver = pd.read_parquet("Driver_Summary_Combined.parquet")
    accident_df, accident_results = train_accident_model(df_driver)

    # 2️⃣ Alarm Data
    df_alarm = pd.read_parquet("Cleaned_Mar24_alarms.parquet")
    fatigue_df, fatigue_results = train_fatigue_model(df_alarm)
    speeding_df, speed_results = train_speeding_model(df_alarm)

    # 3️⃣ Merge All Outputs
    combined = (
        accident_df[["driver", "risk_score"]]
        .merge(fatigue_df, on="driver", how="left")
        .merge(speeding_df, on="driver", how="left")
        .fillna(0)
    )
    combined["overall_risk_index"] = combined[["risk_score", "fatigue_probability", "speeding_probability"]].mean(axis=1)
    combined.to_parquet("driver_combined_risk.parquet", index=False)
    print("✅ Saved → driver_combined_risk.parquet")

    # 4️⃣ Combine all comparison results
    all_results = pd.concat([accident_results, fatigue_results, speed_results], ignore_index=True)
    all_results.to_csv("model_comparison_summary.csv", index=False)
    print("📊 Saved → model_comparison_summary.csv")

    print("\n🎉 All models trained successfully (latest results only).")
    print("=" * 60)



# ===============================================================
if __name__ == "__main__":
    run_machine_learning_analysis()
