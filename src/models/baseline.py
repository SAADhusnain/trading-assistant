"""
Baseline ML Models Module
Implements traditional machine learning models for trading
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """
    Trains and evaluates baseline ML models for trading predictions.
    Supports multiple algorithms and time-series aware splitting.
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize model trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'Target',
        test_size: float = 0.2,
        use_time_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test split with proper scaling.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            test_size: Proportion of data for testing
            use_time_split: Whether to use time-based split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data (test_size={test_size}, time_split={use_time_split})...")
        
        X = df[feature_cols]
        y = df[target_col]
        
        if use_time_split:
            # Time-based split to avoid lookahead bias
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            logger.info("Using time-based split (preserves temporal order)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logger.info("Using random split")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        self.X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"✅ Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        logger.info(f"Train label distribution:\n{self.y_train.value_counts(normalize=True)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self, **kwargs) -> LogisticRegression:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs'
        }
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        self._evaluate_model('Logistic Regression', model)
        
        return model
    
    def train_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1,
            'min_samples_split': 10
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        self._evaluate_model('Random Forest', model)
        
        return model
    
    def train_gradient_boosting(self, **kwargs) -> GradientBoostingClassifier:
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting...")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        model = GradientBoostingClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['Gradient Boosting'] = model
        self._evaluate_model('Gradient Boosting', model)
        
        return model
    
    def train_xgboost(self, **kwargs) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = model
        self._evaluate_model('XGBoost', model)
        
        return model
    
    def train_svm(self, **kwargs) -> SVC:
        """Train Support Vector Machine model."""
        logger.info("Training SVM...")
        
        default_params = {
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        model = SVC(**default_params)
        model.fit(self.X_train, self.y_train)
        
        self.models['SVM'] = model
        self._evaluate_model('SVM', model)
        
        return model
    
    def train_all_models(self) -> Dict:
        """Train all baseline models."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING ALL BASELINE MODELS")
        logger.info("="*70 + "\n")
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        
        logger.info("\n✅ All models trained!")
        self.print_comparison()
        
        return self.models
    
    def _evaluate_model(self, name: str, model) -> Dict:
        """Evaluate a trained model."""
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'precision': precision_score(self.y_test, y_pred_test, zero_division=0),
            'recall': recall_score(self.y_test, y_pred_test, zero_division=0),
            'f1': f1_score(self.y_test, y_pred_test, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test),
            'predictions': y_pred_test,
            'probabilities': y_pred_proba
        }
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        self.results[name] = metrics
        
        # Print results
        logger.info(f"\n✅ {name} Results:")
        logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:      {metrics['precision']:.4f}")
        logger.info(f"   Recall:         {metrics['recall']:.4f}")
        logger.info(f"   F1 Score:       {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            logger.info(f"   ROC AUC:        {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def print_comparison(self):
        """Print comparison table of all models."""
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON")
        logger.info("="*70)
        
        comparison = pd.DataFrame({
            name: {
                'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC AUC': f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') else 'N/A'
            }
            for name, metrics in self.results.items()
        }).T
        
        print(comparison)
        logger.info("="*70 + "\n")
    
    def get_feature_importance(
        self,
        model_name: str = 'Random Forest',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        model = self.models.get(model_name)
        
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        if not hasattr(model, 'feature_importances_'):
            logger.error(f"Model {model_name} doesn't have feature_importances_")
            return None
        
        feature_imp = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n🔝 Top {top_n} Features ({model_name}):")
        print(feature_imp.head(top_n))
        
        return feature_imp
    
    def cross_validate(
        self,
        model_name: str,
        cv_folds: int = 5
    ) -> Dict:
        """
        Perform time-series cross-validation.
        
        Args:
            model_name: Name of the model to cross-validate
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Cross-validating {model_name} with {cv_folds} folds...")
        
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Combine train and test for CV
        X_full = pd.concat([self.X_train, self.X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        
        scores = cross_val_score(model, X_full, y_full, cv=tscv, scoring='accuracy')
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        logger.info(f"CV Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def save_models(self):
        """Save all trained models and scaler."""
        logger.info("Saving models...")
        
        # Save each model
        for name, model in self.models.items():
            filename = f"{name.replace(' ', '_').lower()}.pkl"
            filepath = self.model_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"✅ Saved: {filename}")
        
        # Save scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"✅ Saved: scaler.pkl")
        logger.info(f"💾 All models saved to {self.model_dir}/")
    
    def load_model(self, model_name: str):
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        filename = f"{model_name.replace(' ', '_').lower()}.pkl"
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"✅ Loaded model: {model_name}")
        return model
    
    def load_scaler(self):
        """Load saved scaler."""
        filepath = self.model_dir / 'scaler.pkl'
        
        if not filepath.exists():
            logger.error(f"Scaler file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info("✅ Loaded scaler")
        return scaler
    
    def get_best_model(self) -> Tuple[str, object]:
        """
        Get the best performing model based on test accuracy.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.results:
            logger.error("No models trained yet")
            return None, None
        
        best_name = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])[0]
        best_model = self.models[best_name]
        
        logger.info(f"🏆 Best model: {best_name} (Accuracy: {self.results[best_name]['test_accuracy']:.4f})")
        
        return best_name, best_model


# Example usage
if __name__ == "__main__":
    from src.data.collector import DataCollector
    from src.data.processor import DataProcessor
    
    # Collect and process data
    collector = DataCollector()
    df = collector.fetch_historical_data('AAPL', interval='5m', period='60d')
    
    processor = DataProcessor()
    df_processed, features = processor.process_pipeline(df)
    
    # Train models
    trainer = BaselineModelTrainer()
    trainer.prepare_data(df_processed, features, test_size=0.2, use_time_split=True)
    trainer.train_all_models()
    
    # Get feature importance
    trainer.get_feature_importance('Random Forest', top_n=15)
    
    # Save models
    trainer.save_models()
    
    # Get best model
    best_name, best_model = trainer.get_best_model()