"""
Trading Strategies Module
Implements trading rules, signals, and backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingStrategy:
    """
    Base class for trading strategies.
    Implements ML-based and rule-based trading logic.
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize trading strategy.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [initial_capital]
        
    def generate_signals(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        prob_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Generate trading signals from ML predictions.
        
        Args:
            df: DataFrame with price data
            predictions: Model predictions (0 or 1)
            probabilities: Prediction probabilities
            prob_threshold: Minimum probability for signal
            
        Returns:
            DataFrame with signals
        """
        df_signals = df.copy()
        df_signals['Prediction'] = predictions
        df_signals['Probability'] = probabilities
        
        # Generate signals
        df_signals['Signal'] = 0  # 0 = Hold
        df_signals.loc[
            (df_signals['Prediction'] == 1) & 
            (df_signals['Probability'] >= prob_threshold),
            'Signal'
        ] = 1  # Buy
        
        df_signals.loc[
            (df_signals['Prediction'] == 0) | 
            (df_signals['Probability'] < prob_threshold),
            'Signal'
        ] = -1  # Sell/Hold
        
        return df_signals
    
    def apply_risk_management(
        self,
        entry_price: float,
        current_price: float,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03
    ) -> Tuple[bool, str]:
        """
        Apply risk management rules.
        
        Args:
            entry_price: Entry price of position
            current_price: Current market price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            
        Returns:
            Tuple of (should_exit, reason)
        """
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct <= -stop_loss_pct:
            return True, 'STOP_LOSS'
        elif pnl_pct >= take_profit_pct:
            return True, 'TAKE_PROFIT'
        
        return False, None
    
    def execute_backtest(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        prob_threshold: float = 0.6,
        stop_loss: float = 0.02,
        take_profit: float = 0.03,
        max_position_size: float = 0.95
    ) -> Dict:
        """
        Execute backtest with ML predictions.
        
        Args:
            df: DataFrame with price data
            predictions: Model predictions
            probabilities: Prediction probabilities
            prob_threshold: Minimum probability for entry
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            max_position_size: Maximum capital to use per trade
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("\n" + "="*70)
        logger.info("EXECUTING BACKTEST")
        logger.info("="*70)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Probability Threshold: {prob_threshold:.0%}")
        logger.info(f"Stop Loss: {stop_loss:.1%}")
        logger.info(f"Take Profit: {take_profit:.1%}\n")
        
        # Generate signals
        df_signals = self.generate_signals(df, predictions, probabilities, prob_threshold)
        
        # Reset state
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.capital]
        
        # Execute trades
        for i, (idx, row) in enumerate(df_signals.iterrows()):
            price = row['Close']
            signal = row['Signal']
            prob = row['Probability']
            
            # Entry logic
            if self.position is None and signal == 1:
                # Buy
                shares = (self.capital * max_position_size) / price
                self.position = {
                    'entry_idx': idx,
                    'entry_price': price,
                    'shares': shares,
                    'entry_capital': self.capital
                }
                
            # Exit logic
            elif self.position is not None:
                entry_price = self.position['entry_price']
                
                # Check risk management
                should_exit, reason = self.apply_risk_management(
                    entry_price, price, stop_loss, take_profit
                )
                
                # Also exit on sell signal
                if signal == -1 and reason is None:
                    should_exit = True
                    reason = 'SIGNAL_EXIT'
                
                if should_exit:
                    # Sell
                    shares = self.position['shares']
                    exit_value = shares * price
                    entry_value = shares * entry_price
                    pnl = exit_value - entry_value
                    pnl_pct = (price - entry_price) / entry_price
                    
                    self.capital = exit_value
                    
                    # Record trade
                    trade = {
                        'entry_time': self.position['entry_idx'],
                        'exit_time': idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': reason,
                        'capital': self.capital
                    }
                    self.trades.append(trade)
                    self.position = None
            
            # Track equity
            if self.position is not None:
                current_value = self.position['shares'] * price
                self.equity_curve.append(current_value)
            else:
                self.equity_curve.append(self.capital)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        return metrics
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(self.trades) == 0:
            logger.warning("No trades executed")
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns
        final_capital = self.equity_curve[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Win/Loss stats
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        largest_win = trades_df['pnl'].max()
        largest_loss = trades_df['pnl'].min()
        
        # Sharpe Ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit Factor
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_df': trades_df
        }
        
        self.print_metrics(metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print backtest metrics in a formatted way."""
        logger.info("\n" + "="*70)
        logger.info("BACKTEST RESULTS")
        logger.info("="*70)
        logger.info(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
        logger.info(f"Final Capital:      ${metrics['final_capital']:,.2f}")
        logger.info(f"Total Return:       {metrics['total_return_pct']:+.2f}%")
        logger.info(f"\nTotal Trades:       {metrics['total_trades']}")
        logger.info(f"Winning Trades:     {metrics['winning_trades']}")
        logger.info(f"Losing Trades:      {metrics['losing_trades']}")
        logger.info(f"Win Rate:           {metrics['win_rate']:.1%}")
        logger.info(f"\nAverage Win:        ${metrics['avg_win']:,.2f}")
        logger.info(f"Average Loss:       ${metrics['avg_loss']:,.2f}")
        logger.info(f"Largest Win:        ${metrics['largest_win']:,.2f}")
        logger.info(f"Largest Loss:       ${metrics['largest_loss']:,.2f}")
        logger.info(f"Profit Factor:      {metrics['profit_factor']:.2f}")
        logger.info(f"\nSharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:       {metrics['max_drawdown']:.2%}")
        logger.info("="*70 + "\n")
    
    def save_trades(self, filepath: str = 'logs/backtest_trades.csv'):
        """Save trade history to CSV."""
        if not self.trades:
            logger.warning("No trades to save")
            return
        
        trades_df = pd.DataFrame(self.trades)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved trades to {filepath}")


class RuleBasedStrategy(TradingStrategy):
    """
    Rule-based trading strategy using technical indicators.
    Does not require ML model.
    """
    
    def generate_technical_signals(
        self,
        df: pd.DataFrame,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70
    ) -> pd.DataFrame:
        """
        Generate signals from technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            rsi_oversold: RSI level for oversold
            rsi_overbought: RSI level for overbought
            
        Returns:
            DataFrame with signals
        """
        df_signals = df.copy()
        df_signals['Signal'] = 0
        
        # Buy signals
        buy_conditions = (
            (df_signals['RSI_14'] < rsi_oversold) &
            (df_signals['MACD'] > df_signals['MACD_Signal']) &
            (df_signals['Close'] < df_signals['BB_Lower'])
        )
        df_signals.loc[buy_conditions, 'Signal'] = 1
        
        # Sell signals
        sell_conditions = (
            (df_signals['RSI_14'] > rsi_overbought) &
            (df_signals['MACD'] < df_signals['MACD_Signal']) &
            (df_signals['Close'] > df_signals['BB_Upper'])
        )
        df_signals.loc[sell_conditions, 'Signal'] = -1
        
        return df_signals


# Example usage
if __name__ == "__main__":
    from src.data.collector import DataCollector
    from src.data.processor import DataProcessor
    from src.models.baseline import BaselineModelTrainer
    
    # Collect and process data
    collector = DataCollector()
    df = collector.fetch_historical_data('AAPL', interval='5m', period='60d')
    
    processor = DataProcessor()
    df_processed, features = processor.process_pipeline(df)
    
    # Train model
    trainer = BaselineModelTrainer()
    trainer.prepare_data(df_processed, features, test_size=0.2)
    trainer.train_random_forest()
    
    # Get predictions
    model = trainer.models['Random Forest']
    predictions = trainer.results['Random Forest']['predictions']
    probabilities = trainer.results['Random Forest']['probabilities']
    
    # Get test data
    test_df = df_processed.loc[trainer.X_test.index]
    
    # Backtest strategy
    strategy = TradingStrategy(initial_capital=10000)
    metrics = strategy.execute_backtest(
        test_df,
        predictions,
        probabilities,
        prob_threshold=0.6,
        stop_loss=0.02,
        take_profit=0.03
    )
    
    # Save trades
    strategy.save_trades()