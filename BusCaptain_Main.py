"""
BusCaptain_Main.py
--------------------------------------------------------
Main orchestrator script that runs the full Bus Captain Safety pipeline:
1️⃣ Data Cleaning + Merging + Feature Engineering
2️⃣ Machine Learning Risk Prediction
3️⃣ Launch Streamlit Dashboard UI
--------------------------------------------------------
Run this file with:
    python BusCaptain_Main.py
"""

import os
import subprocess
import time
import sys

# ===============================================================
# 0. ENVIRONMENT CHECKS
# ===============================================================
print("\n=== BUS CAPTAIN SAFETY DASHBOARD PIPELINE ===")
print("Checking environment...")

required_files = [
    "Analyze_data.py",
    "Machine_Learning.py",
    "Dashboard_UI.py"
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print(f"❌ Missing required files: {missing}")
    sys.exit(1)

# ===============================================================
# 1️⃣ STEP 1: DATA ANALYSIS & CLEANING
# ===============================================================
print("\n[1/3] Running data analysis and cleaning...")
try:
    start = time.time()
    subprocess.run([sys.executable, "Analyze_data.py"], check=True)
    print(f"✅ Data analysis completed in {time.time() - start:.1f} seconds.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error while running Analyze_data.py: {e}")
    sys.exit(1)

# ===============================================================
# 2️⃣ STEP 2: MACHINE LEARNING RISK PREDICTION
# ===============================================================
print("\n[2/3] Running machine learning analysis...")
try:
    start = time.time()
    subprocess.run([sys.executable, "Machine_Learning.py"], check=True)
    print(f"✅ ML analysis completed in {time.time() - start:.1f} seconds.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error while running Machine_Learning.py: {e}")
    sys.exit(1)

# ===============================================================
# 3️⃣ STEP 3: LAUNCH STREAMLIT DASHBOARD (robust Windows fix)
# ===============================================================
print("\n[3/3] Launching Streamlit dashboard...")
print("💡 Tip: Press CTRL+C in the terminal to stop the dashboard.\n")

try:
    # Build full Streamlit command
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "Dashboard_UI.py"]

    # Print exact command being run (for debug)
    print("Running command:", " ".join(streamlit_cmd))

    # Launch Streamlit in the same environment
    subprocess.run(streamlit_cmd, check=True)
except FileNotFoundError:
    print("❌ Streamlit executable not found.")
    print("Please install Streamlit with the following command:")
    print("   python -m ")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"❌ Error while launching Streamlit dashboard: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n🛑 Dashboard closed manually. Goodbye 👋")
except Exception as e:
    print(f"❌ Unexpected error while launching Streamlit dashboard: {e}")
    sys.exit(1)
