"""
News-Based Machine Learning Training System
Trains models on news sentiment + price data for prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import logging
from typing import Dict, List, Tuple
import yfinance as yf
from pathlib import Path
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsMLTrainer:
    """
    Trains ML models on combined news sentiment and price data
    Continuously learns from new data to improve predictions
    """
    
    def __init__(self, model_dir: str = 'models/news_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_history = []
        
    def collect_training_data(
        self,
        symbol: str,
        news_scraper,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Collect and combine news sentiment with price data for training
        """
        logger.info(f"Collecting training data for {symbol}...")
        
        # Get historical price data
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        price_data = ticker.history(start=start_date, end=end_date, interval='1h')
        
        if price_data.empty:
            logger.error(f"No price data for {symbol}")
            return pd.DataFrame()
        
        # Calculate technical features
        price_data['Returns'] = price_data['Close'].pct_change()
        price_data['Log_Returns'] = np.log(price_data['Close'] / price_data['Close'].shift(1))
        
        # Price features
        price_data['Price_Change_1h'] = price_data['Close'].pct_change(1)
        price_data['Price_Change_4h'] = price_data['Close'].pct_change(4)
        price_data['Price_Change_24h'] = price_data['Close'].pct_change(24)
        
        # Volume features
        price_data['Volume_Ratio'] = price_data['Volume'] / price_data['Volume'].rolling(24).mean()
        price_data['Volume_Change'] = price_data['Volume'].pct_change()
        
        # Volatility
        price_data['Volatility_24h'] = price_data['Returns'].rolling(24).std()
        price_data['High_Low_Ratio'] = (price_data['High'] - price_data['Low']) / price_data['Close']
        
        # Technical indicators
        # RSI
        delta = price_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        price_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
        price_data['MACD'] = exp1 - exp2
        price_data['MACD_Signal'] = price_data['MACD'].ewm(span=9, adjust=False).mean()
        price_data['MACD_Diff'] = price_data['MACD'] - price_data['MACD_Signal']
        
        # Moving averages
        price_data['SMA_10'] = price_data['Close'].rolling(10).mean()
        price_data['SMA_20'] = price_data['Close'].rolling(20).mean()
        price_data['SMA_50'] = price_data['Close'].rolling(50).mean()
        price_data['Price_to_SMA20'] = price_data['Close'] / price_data['SMA_20']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = price_data['Close'].rolling(bb_period).std()
        price_data['BB_Middle'] = price_data['Close'].rolling(bb_period).mean()
        price_data['BB_Upper'] = price_data['BB_Middle'] + (bb_std * 2)
        price_data['BB_Lower'] = price_data['BB_Middle'] - (bb_std * 2)
        price_data['BB_Position'] = (price_data['Close'] - price_data['BB_Lower']) / (price_data['BB_Upper'] - price_data['BB_Lower'])
        
        # Get news sentiment for each time period
        logger.info("Fetching news sentiment data...")
        sentiment_data = []
        
        for timestamp in price_data.index:
            # Get sentiment for this time period
            sentiment = news_scraper.get_market_sentiment(symbol)
            
            sentiment_features = {
                'news_sentiment_score': sentiment.get('average_score', 0),
                'news_bullish_count': sentiment.get('sentiment_distribution', {}).get('bullish', 0),
                'news_bearish_count': sentiment.get('sentiment_distribution', {}).get('bearish', 0),
                'news_neutral_count': sentiment.get('sentiment_distribution', {}).get('neutral', 0),
                'news_confidence': sentiment.get('confidence', 0),
                'news_article_count': sentiment.get('article_count', 0),
                'news_high_impact': sentiment.get('high_impact_news', 0),
                'timestamp': timestamp
            }
            sentiment_data.append(sentiment_features)
            
            # Consider implementing rate limiting in news_scraper if needed
        
        # Combine sentiment with price data
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df = sentiment_df.set_index('timestamp')
        
        # Merge dataframes
        combined_data = price_data.join(sentiment_df, how='left')
        combined_data.ffill(inplace=True)
        combined_data.fillna(0, inplace=True)
        
        # Create target variable (1 if price goes up in next hour, 0 if down)
        combined_data['Target'] = (combined_data['Close'].shift(-1) > combined_data['Close']).astype(int)
        
        # Add multi-class targets for more detailed predictions
        future_returns = combined_data['Close'].shift(-1) / combined_data['Close'] - 1
        conditions = [
            (future_returns < -0.02),  # Strong sell (< -2%)
            (future_returns < -0.005),  # Sell (-0.5% to -2%)
            (future_returns < 0.005),   # Hold (-0.5% to 0.5%)
            (future_returns < 0.02),    # Buy (0.5% to 2%)
            (future_returns >= 0.02)    # Strong buy (> 2%)
        ]
        choices = [0, 1, 2, 3, 4]  # 0=Strong Sell, 1=Sell, 2=Hold, 3=Buy, 4=Strong Buy
        combined_data['Target_Multiclass'] = np.select(conditions, choices, default=2)
        
        # Remove last row (no target)
        combined_data = combined_data[:-1]
        
        # Drop any remaining NaN values
        combined_data = combined_data.dropna()
        
        logger.info(f"Collected {len(combined_data)} training samples with {len(combined_data.columns)} features")
        
        return combined_data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for training
        """
        # Select features (exclude target and raw price data)
        exclude_cols = ['Target', 'Target_Multiclass', 'Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y_binary = df['Target'].values
        y_multiclass = df['Target_Multiclass'].values
        
        return X, y_binary, y_multiclass, feature_cols
    
    def train_models(
        self,
        symbol: str,
        training_data: pd.DataFrame,
        model_types: List[str] = None
    ) -> Dict:
        """
        Train multiple models on the combined data
        """
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'gradient_boosting']
        
        logger.info(f"Training models for {symbol}...")
        
        # Prepare features
        X, y_binary, y_multiclass, feature_cols = self.prepare_features(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train Random Forest
        if 'random_forest' in model_types:
            logger.info("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.models[f'{symbol}_rf'] = rf_model
            results['random_forest'] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Random Forest - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Train XGBoost
        if 'xgboost' in model_types:
            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            train_score = xgb_model.score(X_train_scaled, y_train)
            test_score = xgb_model.score(X_test_scaled, y_test)
            
            self.models[f'{symbol}_xgb'] = xgb_model
            results['xgboost'] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score
            }
            
            logger.info(f"XGBoost - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Train Gradient Boosting
        if 'gradient_boosting' in model_types:
            logger.info("Training Gradient Boosting...")
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            
            train_score = gb_model.score(X_train_scaled, y_train)
            test_score = gb_model.score(X_test_scaled, y_test)
            
            self.models[f'{symbol}_gb'] = gb_model
            results['gradient_boosting'] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score
            }
            
            logger.info(f"Gradient Boosting - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save models
        self.save_models(symbol)
        
        # Store training history
        self.training_history.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'samples': len(training_data),
            'features': len(feature_cols),
            'results': results
        })
        
        return results
    
    def predict_realtime(
        self,
        symbol: str,
        news_scraper,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Make real-time prediction using trained model and current data
        """
        model_key = f'{symbol}_{model_type.replace("_", "")[:2]}'
        
        if model_key not in self.models:
            logger.error(f"No trained model found for {symbol}")
            return {'error': 'No trained model'}
        
        model = self.models[model_key]
        
        # Get current data
        ticker = yf.Ticker(symbol)
        current_data = ticker.history(period='1d', interval='1h')
        
        if current_data.empty:
            return {'error': 'No current data'}
        
        # Prepare features (same as training)
        latest = current_data.iloc[-1]
        
        # Get current sentiment
        sentiment = news_scraper.get_market_sentiment(symbol)
        
        # Create feature vector
        features = {
            'Returns': 0,
            'Log_Returns': 0,
            'Price_Change_1h': (latest['Close'] - current_data['Close'].iloc[-2]) / current_data['Close'].iloc[-2] if len(current_data) > 1 else 0,
            'Price_Change_4h': (latest['Close'] - current_data['Close'].iloc[-4]) / current_data['Close'].iloc[-4] if len(current_data) > 4 else 0,
            'Price_Change_24h': 0,
            'Volume_Ratio': latest['Volume'] / current_data['Volume'].mean() if current_data['Volume'].mean() > 0 else 1,
            'Volume_Change': 0,
            'Volatility_24h': current_data['Close'].pct_change().std() if len(current_data) > 1 else 0,
            'High_Low_Ratio': (latest['High'] - latest['Low']) / latest['Close'] if latest['Close'] > 0 else 0,
            'RSI': 50,  # Default
            'MACD': 0,
            'MACD_Signal': 0,
            'MACD_Diff': 0,
            'SMA_10': latest['Close'],
            'SMA_20': latest['Close'],
            'SMA_50': latest['Close'],
            'Price_to_SMA20': 1,
            'BB_Middle': latest['Close'],
            'BB_Upper': latest['Close'] * 1.02,
            'BB_Lower': latest['Close'] * 0.98,
            'BB_Position': 0.5,
            'news_sentiment_score': sentiment.get('average_score', 0),
            'news_bullish_count': sentiment.get('sentiment_distribution', {}).get('bullish', 0),
            'news_bearish_count': sentiment.get('sentiment_distribution', {}).get('bearish', 0),
            'news_neutral_count': sentiment.get('sentiment_distribution', {}).get('neutral', 0),
            'news_confidence': sentiment.get('confidence', 0),
            'news_article_count': sentiment.get('article_count', 0),
            'news_high_impact': sentiment.get('high_impact_news', 0)
        }
        
        # Convert to array and scale
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0]
        
        # Determine signal
        confidence = max(probability) * 100
        
        if prediction == 1 and confidence > 70:
            signal = "STRONG BUY"
            action = "BUY"
        elif prediction == 1 and confidence > 55:
            signal = "BUY"
            action = "BUY"
        elif prediction == 0 and confidence > 70:
            signal = "STRONG SELL"
            action = "SELL"
        elif prediction == 0 and confidence > 55:
            signal = "SELL"
            action = "SELL"
        else:
            signal = "HOLD"
            action = "HOLD"
        
        return {
            'symbol': symbol,
            'signal': signal,
            'action': action,
            'confidence': confidence,
            'probability_up': probability[1] * 100 if len(probability) > 1 else 50,
            'probability_down': probability[0] * 100,
            'current_price': latest['Close'],
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'features_used': features
        }
    
    def continuous_learning(
        self,
        symbol: str,
        news_scraper,
        update_interval: int = 3600  # 1 hour
    ):
        """
        Continuously update model with new data
        """
        logger.info(f"Starting continuous learning for {symbol}...")
        
        while True:
            try:
                # Collect new training data
                new_data = self.collect_training_data(symbol, news_scraper, lookback_days=7)
                
                if not new_data.empty:
                    # Retrain models
                    results = self.train_models(symbol, new_data)
                    logger.info(f"Model updated for {symbol}: {results}")
                
                # Wait before next update
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous learning: {e}")
                time.sleep(60)  # Wait a minute before retry
    
    def save_models(self, symbol: str):
        """
        Save trained models to disk
        """
        for key, model in self.models.items():
            if symbol in key:
                model_path = self.model_dir / f"{key}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save training history
        history_path = self.model_dir / f"{symbol}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, default=str, indent=2)
    
    def load_models(self, symbol: str) -> bool:
        """
        Load saved models from disk
        """
        model_loaded = False
        
        # Load all models for this symbol
        for model_file in self.model_dir.glob(f"{symbol}_*.pkl"):
            if 'scaler' not in str(model_file):
                model_key = model_file.stem
                self.models[model_key] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_key}")
                model_loaded = True
        
        # Load scaler
        scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler for {symbol}")
        
        return model_loaded
    
    def get_feature_importance(self, symbol: str, model_type: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance for a trained model
        """
        model_key = f'{symbol}_{model_type.replace("_", "")[:2]}'
        
        if model_key not in self.models:
            return pd.DataFrame()
        
        model = self.models[model_key]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature names from last training
            if self.training_history:
                last_training = self.training_history[-1]
                if model_type in last_training['results']:
                    return last_training['results'][model_type].get('feature_importance', pd.DataFrame())
        
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    from src.data.news_scraper import NewsScraperPro
    
    # Initialize
    news_scraper = NewsScraperPro()
    trainer = NewsMLTrainer()
    
    # Train model for Bitcoin
    symbol = 'BTC-USD'
    
    # Collect training data
    training_data = trainer.collect_training_data(symbol, news_scraper, lookback_days=7)
    
    if not training_data.empty:
        # Train models
        results = trainer.train_models(symbol, training_data)
        print(f"Training results: {results}")
        
        # Make prediction
        prediction = trainer.predict_realtime(symbol, news_scraper)
        print(f"\nPrediction for {symbol}:")
        print(f"Signal: {prediction['signal']}")
        print(f"Confidence: {prediction['confidence']:.1f}%")
        print(f"Probability Up: {prediction['probability_up']:.1f}%")