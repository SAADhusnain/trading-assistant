"""
AI Trading Assistant
A comprehensive ML-powered trading system
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.data.collector import DataCollector, StockDataCollector, CryptoDataCollector
from src.data.processor import DataProcessor
from src.models.baseline import BaselineModelTrainer
from src.strategies.rules import TradingStrategy

__all__ = [
    'DataCollector',
    'StockDataCollector', 
    'CryptoDataCollector',
    'DataProcessor',
    'BaselineModelTrainer',
    'TradingStrategy',
]