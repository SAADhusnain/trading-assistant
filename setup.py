"""
Setup script for AI Trading Assistant
Install: pip install -e .

"""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path("README.md")
long_description = readme.read_text() if readme.exists() else ""

setup(
    name="ai-trading-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered trading assistant with ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trading-assistant",
    packages=find_packages(exclude=["tests", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "yfinance>=0.2.28",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.7.0", "jupyter>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "trading-assistant=run:main",
        ],
    },
)