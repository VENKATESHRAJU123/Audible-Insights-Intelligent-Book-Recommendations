"""
Download Required NLTK Data
Run this once to download all necessary NLTK resources
"""

import nltk
import ssl

# Handle SSL certificate issues (common on Windows)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK data packages...")
print("=" * 60)

# Download required packages
packages = [
    'punkt',           # Tokenizer
    'punkt_tab',       # New tokenizer data (Python 3.13)
    'stopwords',       # Stop words
    'wordnet',         # WordNet lexical database
    'averaged_perceptron_tagger',  # POS tagger
    'omw-1.4'         # Open Multilingual Wordnet
]

for package in packages:
    print(f"\nDownloading '{package}'...")
    try:
        nltk.download(package, quiet=False)
        print(f"✅ {package} downloaded successfully")
    except Exception as e:
        print(f"⚠️  Error downloading {package}: {e}")

print("\n" + "=" * 60)
print("✅ NLTK data download complete!")
print("=" * 60)
print("\nYou can now run: python src/nlp_features.py")
