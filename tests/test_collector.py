"""
Unit tests for data collector
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import DataCollector, StockDataCollector, CryptoDataCollector


class TestDataCollector:
    """Test cases for DataCollector class"""
    
    def test_collector_initialization(self):
        """Test collector initialization"""
        collector = DataCollector()
        assert collector.data_dir.exists()
    
    def test_fetch_stock_data(self):
        """Test fetching stock data"""
        collector = StockDataCollector()
        df = collector.fetch_historical_data('AAPL', interval='1d', period='5d', save=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Close' in df.columns
        assert 'Open' in df.columns
    
    def test_fetch_crypto_data(self):
        """Test fetching crypto data"""
        collector = CryptoDataCollector()
        df = collector.fetch_historical_data('BTC-USD', interval='1d', period='5d', save=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_invalid_symbol(self):
        """Test handling invalid symbol"""
        collector = DataCollector()
        with pytest.raises(Exception):
            collector.fetch_historical_data('INVALID123XYZ', interval='1d', period='5d', save=False)
    
    def test_data_columns(self):
        """Test data has required columns"""
        collector = DataCollector()
        df = collector.fetch_historical_data('AAPL', interval='1d', period='5d', save=False)
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            assert col in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])