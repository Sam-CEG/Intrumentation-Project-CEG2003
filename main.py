import subprocess
import sys

def run_pipeline():
    """Run the complete data pipeline"""
    print("Starting Complete Bus Captain Safety Analysis Pipeline")
    print("="*60)
    
    # Step 1: Merge and clean data
    print("\nStep 1: Merging and cleaning data...")
    import merge
    print("? Data merging completed")
    
    # Step 2: Analyze and create features
    print("\nStep 2: Analyzing data and creating features...")
    import Analyze_data as analyze
    print("? Data analysis completed")
    
    # Step 3: Machine learning
    print("\nStep 3: Running machine learning analysis...")
    from machine_learning import run_machine_learning_analysis
    ml_results = run_machine_learning_analysis()
    print("? Machine learning completed")
    
    # Step 4: Launch dashboard
    print("\nStep 4: Starting dashboard...")
    print("Dashboard will open in your browser. Use Ctrl+C to stop.")
    try:
        import Dashboard_UI
        # For Streamlit, you would typically run: streamlit run Dashboard_UI.py
        print("To view dashboard, run: streamlit run Dashboard_UI.py")
    except Exception as e:
        print(f"Dashboard error: {e}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
