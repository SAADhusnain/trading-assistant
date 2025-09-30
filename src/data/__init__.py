# Add this import to your existing imports
from .news_scraper import NewsScraperPro

# Update the __all__ list to include NewsScraperPro
__all__ = [
    'DataCollector',
    'StockDataCollector',
    'CryptoDataCollector',
    'DataProcessor',
    'NewsScraperPro',  # ADD THIS
]