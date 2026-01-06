"""
Complete Application Runner
Executes the entire pipeline from data creation to app launch
"""

import subprocess
import sys
from pathlib import Path
import time

def print_step(step_num, title):
    """Print formatted step header"""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in: {description}")
        return False
    print(f"✅ {description} completed!\n")
    return True

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("BOOK RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Create Sample Data
    print_step(1, "Creating Sample Dataset")
    if not run_command(
        f"{sys.executable} scripts/create_sample_data.py",
        "Sample data creation"
    ):
        return
    
    time.sleep(2)
    
    # Step 2: Data Processing
    print_step(2, "Data Processing (Merge & Clean)")
    if not run_command(
        f"{sys.executable} src/data_processing.py",
        "Data processing"
    ):
        return
    
    time.sleep(2)
    
    # Step 3: Feature Extraction
    print_step(3, "NLP Feature Extraction")
    if not run_command(
        f"{sys.executable} src/nlp_features.py",
        "Feature extraction"
    ):
        return
    
    time.sleep(2)
    
    # Step 4: Clustering
    print_step(4, "Clustering Analysis")
    if not run_command(
        f"{sys.executable} src/clustering.py",
        "Clustering"
    ):
        return
    
    time.sleep(2)
    
    # Step 5: Train Models
    print_step(5, "Training Recommendation Models")
    if not run_command(
        f"{sys.executable} scripts/train_models.py",
        "Model training"
    ):
        return
    
    time.sleep(2)
    
    # Step 6: Run Tests
    print_step(6, "Running Tests")
    run_command(
        "pytest tests/ -v --cov=src --cov-report=term-missing",
        "Unit tests"
    )
    
    time.sleep(2)
    
    # Step 7: Launch Streamlit App
    print_step(7, "Launching Streamlit Application")
    print("\n🚀 Starting Streamlit app...")
    print("📱 The app will open in your browser")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    time.sleep(3)
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app/streamlit_app.py",
        "--server.port=8501"
    ])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
