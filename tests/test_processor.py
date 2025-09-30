"""
Unit tests for data processor
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.data.processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        collector = DataCollector()
        df = collector.fetch_historical_data('AAPL', interval='1d', period='30d', save=False)
        return df
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = DataProcessor()
        assert processor.data_dir.exists()
    
    def test_clean_data(self, sample_data):
        """Test data cleaning"""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert len(df_clean) > 0
        assert df_clean.isnull().sum().sum() == 0  # No missing values
    
    def test_add_technical_indicators(self, sample_data):
        """Test adding technical indicators"""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data)
        df_indicators = processor.add_technical_indicators(df_clean)
        
        # Check for key indicators
        assert 'RSI_14' in df_indicators.columns
        assert 'MACD' in df_indicators.columns
        assert 'BB_Upper' in df_indicators.columns
        assert len(df_indicators.columns) > len(sample_data.columns)
    
    def test_create_labels(self, sample_data):
        """Test label creation"""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data)
        df_indicators = processor.add_technical_indicators(df_clean)
        df_labeled = processor.create_labels(df_indicators)
        
        assert 'Target' in df_labeled.columns
        assert df_labeled['Target'].isin([0, 1]).all()
    
    def test_process_pipeline(self, sample_data):
        """Test complete processing pipeline"""
        processor = DataProcessor()
        df_processed, features = processor.process_pipeline(sample_data, save=False)
        
        assert isinstance(df_processed, pd.DataFrame)
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'Target' in df_processed.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])