"""
Streamlit App Launcher
Quick script to launch the Streamlit application
"""

import subprocess
import sys
from pathlib import Path

def launch_app():
    """Launch the Streamlit application"""
    
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    
    print("\n" + "="*60)
    print("LAUNCHING BOOK RECOMMENDATION SYSTEM")
    print("="*60)
    print(f"\nApp location: {app_path}")
    print("\nStarting Streamlit server...")
    print("\n🌐 The app will open in your browser automatically")
    print("📝 Press Ctrl+C to stop the server\n")
    print("="*60 + "\n")
    
    # Launch Streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=false"
    ])

if __name__ == "__main__":
    launch_app()
