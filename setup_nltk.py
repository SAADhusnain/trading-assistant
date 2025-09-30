"""
One-time setup for NLTK data
Run this after installing requirements
"""

import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
print("✅ NLTK setup complete!")