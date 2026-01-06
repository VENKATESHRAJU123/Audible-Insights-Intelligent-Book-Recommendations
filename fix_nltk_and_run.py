"""
Complete Fix: Download NLTK Data and Run NLP Features
"""

import subprocess
import sys

# Download NLTK data
print("Step 1: Downloading NLTK data...")
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print("✅ NLTK data downloaded\n")

# Run NLP features
print("Step 2: Running NLP feature extraction...")
subprocess.run([sys.executable, "src/nlp_features.py"])
