"""
Data Processing Module
Handles data cleaning and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes raw market data: cleaning, feature engineering, labeling.
    Saves processed data to data/processed/ directory.
    """
    
    def __init__(self, data_dir: str = 'data/processed'):
        """
        Initialize data processor.
        
        Args:
            data_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cols = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw market data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        # Remove timezone
        if df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_localize(None)
        
        # Forward fill then backward fill missing values
        df_clean.fillna(method='ffill', inplace=True)
        df_clean.fillna(method='bfill', inplace=True)
        
        # Remove any remaining NaN
        initial_len = len(df_clean)
        df_clean.dropna(inplace=True)
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by index
        df_clean.sort_index(inplace=True)
        
        # Remove invalid rows (zero or negative prices)
        df_clean = df_clean[
            (df_clean['Open'] > 0) &
            (df_clean['High'] > 0) &
            (df_clean['Low'] > 0) &
            (df_clean['Close'] > 0)
        ]
        
        removed = initial_len - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows")
        
        logger.info(f"✅ Cleaned data: {len(df_clean)} records")
        return df_clean
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        logger.info("Adding technical indicators...")
        
        df_indicators = df.copy()
        
        # Price-based features
        df_indicators['Returns'] = df_indicators['Close'].pct_change()
        df_indicators['Log_Returns'] = np.log(df_indicators['Close'] / df_indicators['Close'].shift(1))
        df_indicators['High_Low_Pct'] = (df_indicators['High'] - df_indicators['Low']) / df_indicators['Close']
        df_indicators['Close_Open_Pct'] = (df_indicators['Close'] - df_indicators['Open']) / df_indicators['Open']
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df_indicators[f'SMA_{period}'] = df_indicators['Close'].rolling(window=period).mean()
            df_indicators[f'EMA_{period}'] = df_indicators['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        df_indicators['RSI_14'] = self._calculate_rsi(df_indicators['Close'], 14)
        df_indicators['RSI_7'] = self._calculate_rsi(df_indicators['Close'], 7)
        df_indicators['RSI_21'] = self._calculate_rsi(df_indicators['Close'], 21)
        
        # MACD
        exp1 = df_indicators['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_indicators['Close'].ewm(span=26, adjust=False).mean()
        df_indicators['MACD'] = exp1 - exp2
        df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
        df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
        
        # Bollinger Bands
        df_indicators['BB_Middle'] = df_indicators['Close'].rolling(window=20).mean()
        bb_std = df_indicators['Close'].rolling(window=20).std()
        df_indicators['BB_Upper'] = df_indicators['BB_Middle'] + (bb_std * 2)
        df_indicators['BB_Lower'] = df_indicators['BB_Middle'] - (bb_std * 2)
        df_indicators['BB_Width'] = (df_indicators['BB_Upper'] - df_indicators['BB_Lower']) / df_indicators['BB_Middle']
        df_indicators['BB_Position'] = (df_indicators['Close'] - df_indicators['BB_Lower']) / (df_indicators['BB_Upper'] - df_indicators['BB_Lower'])
        
        # Stochastic Oscillator
        low_14 = df_indicators['Low'].rolling(window=14).min()
        high_14 = df_indicators['High'].rolling(window=14).max()
        df_indicators['Stoch_K'] = 100 * ((df_indicators['Close'] - low_14) / (high_14 - low_14))
        df_indicators['Stoch_D'] = df_indicators['Stoch_K'].rolling(window=3).mean()
        
        # ATR (Average True Range)
        high_low = df_indicators['High'] - df_indicators['Low']
        high_close = np.abs(df_indicators['High'] - df_indicators['Close'].shift())
        low_close = np.abs(df_indicators['Low'] - df_indicators['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df_indicators['ATR_14'] = true_range.rolling(14).mean()
        df_indicators['ATR_7'] = true_range.rolling(7).mean()
        
        # Volume indicators
        df_indicators['Volume_SMA_20'] = df_indicators['Volume'].rolling(window=20).mean()
        df_indicators['Volume_Ratio'] = df_indicators['Volume'] / df_indicators['Volume_SMA_20']
        df_indicators['Volume_Change'] = df_indicators['Volume'].pct_change()
        
        # Momentum
        df_indicators['Momentum_10'] = df_indicators['Close'] - df_indicators['Close'].shift(10)
        df_indicators['Momentum_20'] = df_indicators['Close'] - df_indicators['Close'].shift(20)
        df_indicators['ROC_10'] = ((df_indicators['Close'] - df_indicators['Close'].shift(10)) / df_indicators['Close'].shift(10)) * 100
        
        # Price position relative to moving averages
        df_indicators['Price_to_SMA20'] = df_indicators['Close'] / df_indicators['SMA_20']
        df_indicators['Price_to_SMA50'] = df_indicators['Close'] / df_indicators['SMA_50']
        df_indicators['Price_to_EMA20'] = df_indicators['Close'] / df_indicators['EMA_20']
        
        # Crossovers
        df_indicators['SMA_5_20_Cross'] = df_indicators['SMA_5'] - df_indicators['SMA_20']
        df_indicators['EMA_10_50_Cross'] = df_indicators['EMA_10'] - df_indicators['EMA_50']
        
        # Volatility
        df_indicators['Volatility_20'] = df_indicators['Close'].rolling(window=20).std()
        df_indicators['Volatility_50'] = df_indicators['Close'].rolling(window=50).std()
        
        # On-Balance Volume (OBV)
        df_indicators['OBV'] = (np.sign(df_indicators['Close'].diff()) * df_indicators['Volume']).fillna(0).cumsum()
        
        # Williams %R
        highest_high = df_indicators['High'].rolling(window=14).max()
        lowest_low = df_indicators['Low'].rolling(window=14).min()
        df_indicators['Williams_R'] = -100 * ((highest_high - df_indicators['Close']) / (highest_high - lowest_low))
        
        # Commodity Channel Index (CCI)
        tp = (df_indicators['High'] + df_indicators['Low'] + df_indicators['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df_indicators['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        logger.info(f"✅ Added technical indicators: {len(df_indicators.columns)} total columns")
        return df_indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_labels(
        self,
        df: pd.DataFrame,
        lookahead: int = 1,
        threshold: float = 0.0,
        label_type: str = 'binary'
    ) -> pd.DataFrame:
        """
        Create target labels for ML.
        
        Args:
            df: DataFrame with features
            lookahead: Number of candles to look ahead
            threshold: Minimum price change % to consider as movement
            label_type: 'binary' or 'multiclass'
            
        Returns:
            DataFrame with labels
        """
        logger.info(f"Creating {label_type} labels (lookahead={lookahead}, threshold={threshold}%)...")
        
        df_labeled = df.copy()
        
        # Future price and return
        df_labeled['Future_Close'] = df_labeled['Close'].shift(-lookahead)
        df_labeled['Future_Return'] = ((df_labeled['Future_Close'] - df_labeled['Close']) / df_labeled['Close']) * 100
        
        if label_type == 'binary':
            # Binary: 1 if price goes up, 0 if down
            df_labeled['Target'] = (df_labeled['Future_Return'] > threshold).astype(int)
            
        elif label_type == 'multiclass':
            # Multi-class: 0=Down, 1=Neutral, 2=Up
            df_labeled['Target'] = pd.cut(
                df_labeled['Future_Return'],
                bins=[-np.inf, -threshold, threshold, np.inf],
                labels=[0, 1, 2]
            ).astype(float)
        
        # Remove rows with NaN labels
        df_labeled = df_labeled[:-lookahead]
        
        # Print label distribution
        label_dist = df_labeled['Target'].value_counts(normalize=True)
        logger.info(f"Label distribution:\n{label_dist}")
        
        return df_labeled
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare final feature set for ML.
        
        Args:
            df: DataFrame with all features and labels
            
        Returns:
            Tuple of (processed DataFrame, list of feature column names)
        """
        logger.info("Preparing features for ML...")
        
        df_features = df.copy()
        
        # Drop non-feature columns
        drop_cols = [
            'Future_Close', 'Future_Return',
            'Dividends', 'Stock Splits'
        ]
        df_features.drop(
            [col for col in drop_cols if col in df_features.columns],
            axis=1,
            inplace=True
        )
        
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove rows with NaN
        initial_len = len(df_features)
        df_features.dropna(inplace=True)
        removed = initial_len - len(df_features)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with NaN values")
        
        # Identify feature columns (everything except Target)
        self.feature_cols = [col for col in df_features.columns if col != 'Target']
        
        logger.info(f"✅ Prepared {len(self.feature_cols)} features, {len(df_features)} samples")
        
        return df_features, self.feature_cols
    
    def process_pipeline(
        self,
        df: pd.DataFrame,
        lookahead: int = 1,
        threshold: float = 0.0,
        label_type: str = 'binary',
        save: bool = True,
        filename: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run complete processing pipeline.
        
        Args:
            df: Raw DataFrame
            lookahead: Lookahead periods for labels
            threshold: Label threshold
            label_type: Type of labels
            save: Whether to save processed data
            filename: Custom filename for saving
            
        Returns:
            Tuple of (processed DataFrame, feature column names)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*60)
        
        # Clean
        df_clean = self.clean_data(df)
        
        # Add indicators
        df_indicators = self.add_technical_indicators(df_clean)
        
        # Create labels
        df_labeled = self.create_labels(df_indicators, lookahead, threshold, label_type)
        
        # Prepare features
        df_processed, feature_cols = self.prepare_features(df_labeled)
        
        # Save if requested
        if save:
            if filename is None:
                symbol = df.attrs.get('symbol', 'unknown')
                interval = df.attrs.get('interval', 'unknown')
                filename = f"{symbol}_{interval}_processed.csv"
            
            filepath = self.data_dir / filename
            df_processed.to_csv(filepath)
            logger.info(f"💾 Saved processed data to {filepath}")
        
        logger.info("\n✅ PROCESSING PIPELINE COMPLETE!")
        logger.info("="*60 + "\n")
        
        return df_processed, feature_cols
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with statistics
        """
        stats = df.describe().T
        stats['missing'] = df.isnull().sum()
        stats['missing_pct'] = (df.isnull().sum() / len(df)) * 100
        
        return stats
    
    def detect_outliers(self, df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using z-score method.
        
        Args:
            df: DataFrame with features
            n_std: Number of standard deviations for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        outliers = pd.DataFrame(index=df.index)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'Target':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[f'{col}_outlier'] = z_scores > n_std
        
        return outliers


# Example usage
if __name__ == "__main__":
    from src.data.collector import DataCollector
    
    # Collect data
    collector = DataCollector()
    df = collector.fetch_historical_data('AAPL', interval='5m', period='7d')
    
    # Process data
    processor = DataProcessor()
    df_processed, features = processor.process_pipeline(df, lookahead=1, threshold=0.0)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Number of features: {len(features)}")
    print(f"\nFirst few feature names: {features[:10]}")
    print(f"\nTarget distribution:\n{df_processed['Target'].value_counts()}")