"""
Main Execution Script
Run complete pipeline or individual components
Usage: python run.py [command]
Commands: collect, process, train, backtest, paper-trade, all
"""

import sys
import argparse
import yaml
from pathlib import Path
import logging

from src.data.collector import DataCollector, StockDataCollector, CryptoDataCollector
from src.data.processor import DataProcessor
from src.models.baseline import BaselineModelTrainer
from src.strategies.rules import TradingStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return {}


def run_data_collection(config: dict):
    """Run data collection phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("="*70)
    
    symbol = config['symbol']['default']
    interval = config['data']['interval']
    period = config['data']['period']
    
    # Select appropriate collector
    if config['symbol']['type'] == 'crypto':
        collector = CryptoDataCollector()
    else:
        collector = StockDataCollector()
    
    # Fetch data
    df = collector.fetch_historical_data(
        symbol=symbol,
        interval=interval,
        period=period,
        save=True
    )
    
    logger.info(f"✅ Data collection complete: {len(df)} records")
    return df


def run_data_processing(df, config: dict):
    """Run data processing phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: DATA PROCESSING")
    logger.info("="*70)
    
    processor = DataProcessor()
    
    # Process data
    df_processed, features = processor.process_pipeline(
        df=df,
        lookahead=config['labels']['lookahead'],
        threshold=config['labels']['threshold'],
        label_type=config['labels']['type'],
        save=True
    )
    
    logger.info(f"✅ Data processing complete")
    logger.info(f"   Features: {len(features)}")
    logger.info(f"   Samples: {len(df_processed)}")
    
    return df_processed, features


def run_model_training(df_processed, features, config: dict):
    """Run model training phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("="*70)
    
    trainer = BaselineModelTrainer()
    
    # Prepare data
    trainer.prepare_data(
        df=df_processed,
        feature_cols=features,
        test_size=config['models']['test_size'],
        use_time_split=config['models']['use_time_split']
    )
    
    # Train all models
    trainer.train_all_models()
    
    # Get feature importance
    trainer.get_feature_importance('Random Forest', top_n=15)
    
    # Save models
    trainer.save_models()
    
    logger.info("✅ Model training complete")
    
    return trainer


def run_backtesting(trainer, df_processed, config: dict):
    """Run backtesting phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: BACKTESTING")
    logger.info("="*70)
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    
    # Get predictions
    predictions = trainer.results[best_name]['predictions']
    probabilities = trainer.results[best_name]['probabilities']
    
    # Get test data
    test_df = df_processed.loc[trainer.X_test.index]
    
    # Create strategy and backtest
    strategy = TradingStrategy(
        initial_capital=config['strategy']['initial_capital']
    )
    
    metrics = strategy.execute_backtest(
        df=test_df,
        predictions=predictions,
        probabilities=probabilities,
        prob_threshold=config['strategy']['prob_threshold'],
        stop_loss=config['strategy']['stop_loss'],
        take_profit=config['strategy']['take_profit'],
        max_position_size=config['strategy']['max_position_size']
    )
    
    # Save trades
    strategy.save_trades('logs/backtest_trades.csv')
    
    logger.info("✅ Backtesting complete")
    
    return metrics


def run_paper_trading(config: dict):
    """Run paper trading phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 5: PAPER TRADING")
    logger.info("="*70)
    
    logger.info("⚠️  Paper trading requires a separate continuous process")
    logger.info("Run: python -m src.paper_trading")
    logger.info("Or use the Streamlit dashboard: streamlit run app/main.py")


def run_full_pipeline(config: dict):
    """Run complete pipeline."""
    logger.info("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           AI TRADING ASSISTANT - FULL PIPELINE                  ║
    ║                  Running All Phases                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Phase 1: Data Collection
    df = run_data_collection(config)
    
    # Phase 2: Data Processing
    df_processed, features = run_data_processing(df, config)
    
    # Phase 3: Model Training
    trainer = run_model_training(df_processed, features, config)
    
    # Phase 4: Backtesting
    metrics = run_backtesting(trainer, df_processed, config)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info("\n📊 Summary:")
    logger.info(f"   Symbol: {config['symbol']['default']}")
    logger.info(f"   Data Points: {len(df_processed)}")
    logger.info(f"   Features: {len(features)}")
    logger.info(f"   Best Model: {trainer.get_best_model()[0]}")
    logger.info(f"   Backtest Return: {metrics['total_return_pct']:+.2f}%")
    logger.info(f"   Win Rate: {metrics['win_rate']:.1%}")
    logger.info("\n🚀 Next Steps:")
    logger.info("   1. Review results in logs/")
    logger.info("   2. Launch dashboard: streamlit run app/main.py")
    logger.info("   3. Start paper trading if satisfied")
    logger.info("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Trading Assistant')
    parser.add_argument(
        'command',
        choices=['collect', 'process', 'train', 'backtest', 'paper-trade', 'all'],
        help='Command to execute'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--symbol',
        help='Override symbol from config'
    )
    parser.add_argument(
        '--interval',
        help='Override interval from config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.symbol:
        config['symbol']['default'] = args.symbol
    if args.interval:
        config['data']['interval'] = args.interval
    
    # Execute command
    if args.command == 'all':
        run_full_pipeline(config)
    
    elif args.command == 'collect':
        df = run_data_collection(config)
        logger.info(f"Data saved to data/raw/")
    
    elif args.command == 'process':
        # Load latest raw data
        collector = DataCollector()
        df = collector.load_latest_data(
            config['symbol']['default'],
            config['data']['interval']
        )
        if df is None:
            logger.error("No raw data found. Run 'collect' first.")
            return
        
        df_processed, features = run_data_processing(df, config)
        logger.info(f"Processed data saved to data/processed/")
    
    elif args.command == 'train':
        # Load processed data
        import pandas as pd
        processed_files = list(Path('data/processed').glob('*.csv'))
        if not processed_files:
            logger.error("No processed data found. Run 'process' first.")
            return
        
        df_processed = pd.read_csv(processed_files[-1], index_col=0, parse_dates=True)
        features = [col for col in df_processed.columns if col != 'Target']
        
        trainer = run_model_training(df_processed, features, config)
        logger.info(f"Models saved to models/")
    
    elif args.command == 'backtest':
        # Load models and data
        try:
            import pandas as pd
            processed_files = list(Path('data/processed').glob('*.csv'))
            df_processed = pd.read_csv(processed_files[-1], index_col=0, parse_dates=True)
            features = [col for col in df_processed.columns if col != 'Target']
            
            trainer = BaselineModelTrainer()
            trainer.prepare_data(df_processed, features, test_size=config['models']['test_size'])
            
            # Load saved models
            for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Logistic Regression']:
                model = trainer.load_model(model_name)
                if model:
                    trainer.models[model_name] = model
                    # Evaluate
                    trainer._evaluate_model(model_name, model)
            
            metrics = run_backtesting(trainer, df_processed, config)
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            logger.info("Make sure you've run 'train' first")
    
    elif args.command == 'paper-trade':
        run_paper_trading(config)


if __name__ == "__main__":
    main()