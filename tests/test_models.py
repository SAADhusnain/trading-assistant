"""
Unit tests for ML models
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.data.processor import DataProcessor
from src.models.baseline import BaselineModelTrainer


class TestMLModels:
    """Test cases for ML models"""
    
    @pytest.fixture
    def prepared_data(self):
        """Prepare data for model testing"""
        collector = DataCollector()
        df = collector.fetch_historical_data('AAPL', interval='1d', period='180d', save=False)
        
        processor = DataProcessor()
        df_processed, features = processor.process_pipeline(df, save=False)
        
        return df_processed, features
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = BaselineModelTrainer()
        assert trainer.model_dir.exists()
    
    def test_prepare_data(self, prepared_data):
        """Test data preparation"""
        df_processed, features = prepared_data
        trainer = BaselineModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            df_processed, features, test_size=0.2
        )
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) > len(X_test)
    
    def test_train_logistic_regression(self, prepared_data):
        """Test training logistic regression"""
        df_processed, features = prepared_data
        trainer = BaselineModelTrainer()
        trainer.prepare_data(df_processed, features, test_size=0.2)