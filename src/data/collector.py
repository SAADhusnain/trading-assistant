"""
Data Collection Module
Handles fetching market data from various sources
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects market data from Yahoo Finance and other sources.
    Saves data to data/raw/ directory.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize data collector.
        
        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = '5m',
        period: str = '60d',
        save: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD')
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            save: Whether to save data to disk
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {symbol} data: {interval} interval, {period} period")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Remove timezone info
            df.index = df.index.tz_localize(None)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['interval'] = interval
            df.attrs['period'] = period
            
            logger.info(f"✅ Downloaded {len(df)} records")
            
            if save:
                self.save_data(df, symbol, interval)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = '5m',
        period: str = '60d'
    ) -> dict:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            interval: Data interval
            period: Time period
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_historical_data(symbol, interval, period)
                data[symbol] = df
                logger.info(f"✅ {symbol}: {len(df)} records")
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                continue
        
        return data
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        Save data to CSV file.
        
        Args:
            df: DataFrame to save
            symbol: Symbol name
            interval: Data interval
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol.replace('-', '_')}_{interval}_{timestamp}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"💾 Saved to {filepath}")
        
    def load_latest_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load most recent saved data for a symbol.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval
            
        Returns:
            DataFrame if found, None otherwise
        """
        pattern = f"{symbol.replace('-', '_')}_{interval}_*.csv"
        files = sorted(self.data_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No saved data found for {symbol}")
            return None
        
        latest_file = files[-1]
        logger.info(f"Loading {latest_file}")
        
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        return df
    
    def get_real_time_data(self, symbol: str) -> dict:
        """
        Get current real-time price and info.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with current price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'previous_close': info.get('previousClose'),
                'open': info.get('open'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {}
    
    def update_data(self, symbol: str, interval: str = '5m'):
        """
        Update existing data with latest candles.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval
        """
        # Load existing data
        existing_df = self.load_latest_data(symbol, interval)
        
        if existing_df is None:
            logger.info("No existing data, fetching fresh data")
            self.fetch_historical_data(symbol, interval, save=True)
            return
        
        # Fetch new data
        last_date = existing_df.index[-1]
        days_since = (datetime.now() - last_date).days + 1
        period = f"{min(days_since, 60)}d"
        
        new_df = self.fetch_historical_data(symbol, interval, period, save=False)
        
        # Merge and remove duplicates
        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        # Save updated data
        self.save_data(combined_df, symbol, interval)
        
        logger.info(f"✅ Updated data: {len(existing_df)} -> {len(combined_df)} records")


class CryptoDataCollector(DataCollector):
    """
    Specialized collector for cryptocurrency data.
    Extends base collector with crypto-specific features.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        super().__init__(data_dir)
        
    def fetch_crypto_pairs(
        self,
        pairs: List[str] = None,
        interval: str = '5m',
        period: str = '60d'
    ) -> dict:
        """
        Fetch data for multiple crypto pairs.
        
        Args:
            pairs: List of crypto pairs (default: BTC-USD, ETH-USD)
            interval: Data interval
            period: Time period
            
        Returns:
            Dictionary of DataFrames
        """
        if pairs is None:
            pairs = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
        
        return self.fetch_multiple_symbols(pairs, interval, period)


class StockDataCollector(DataCollector):
    """
    Specialized collector for stock market data.
    Extends base collector with stock-specific features.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        super().__init__(data_dir)
        
    def fetch_sp500_stocks(
        self,
        top_n: int = 10,
        interval: str = '5m',
        period: str = '60d'
    ) -> dict:
        """
        Fetch data for top S&P 500 stocks.
        
        Args:
            top_n: Number of top stocks to fetch
            interval: Data interval
            period: Time period
            
        Returns:
            Dictionary of DataFrames
        """
        # Popular S&P 500 stocks
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'V', 'JPM', 'WMT'
        ]
        
        stocks = popular_stocks[:top_n]
        return self.fetch_multiple_symbols(stocks, interval, period)


# Example usage
if __name__ == "__main__":
    # Stock example
    stock_collector = StockDataCollector()
    aapl_data = stock_collector.fetch_historical_data('AAPL', interval='5m', period='7d')
    print(f"AAPL data shape: {aapl_data.shape}")
    
    # Crypto example
    crypto_collector = CryptoDataCollector()
    btc_data = crypto_collector.fetch_historical_data('BTC-USD', interval='5m', period='7d')
    print(f"BTC data shape: {btc_data.shape}")
    
    # Real-time data
    collector = DataCollector()
    real_time = collector.get_real_time_data('AAPL')
    print(f"Real-time price: ${real_time.get('current_price')}")