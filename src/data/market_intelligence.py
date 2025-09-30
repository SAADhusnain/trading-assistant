"""
Market Intelligence Module
Fetches market metrics
"""

import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketIntelligence:
    """Get market metrics like volume, open interest, etc."""
    
    def __init__(self):
        self.crypto_api = "https://api.coingecko.com/api/v3"
        self.binance_api = "https://fapi.binance.com"
    
    def get_market_metrics(self, symbol: str) -> dict:
        """Get comprehensive market metrics"""
        metrics = {}
        
        if '-USD' in symbol or symbol.startswith('BTC') or symbol.startswith('ETH'):
            metrics = self._get_crypto_metrics(symbol)
        else:
            metrics = self._get_stock_metrics(symbol)
        
        return metrics
    
    def _get_crypto_metrics(self, symbol: str) -> dict:
        """Get crypto metrics"""
        try:
            coin_id = symbol.replace('-USD', '').lower()
            url = f"{self.crypto_api}/coins/{coin_id}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            market_data = data.get('market_data', {})
            
            return {
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'circulating_supply': market_data.get('circulating_supply', 0),
            }
        except Exception as e:
            logger.warning(f"Could not get crypto metrics: {e}")
            return {}
    
    def _get_stock_metrics(self, symbol: str) -> dict:
        """Get stock metrics"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'market_cap': info.get('marketCap', 0),
                'total_volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
            }
        except Exception as e:
            logger.warning(f"Could not get stock metrics: {e}")
            return {}