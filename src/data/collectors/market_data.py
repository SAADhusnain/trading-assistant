import requests
import pandas as pd

class MarketDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.example.com/marketdata"

    def get_market_data(self, symbol):
        url = f"{self.base_url}?symbol={symbol}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching market data: {response.status_code}")

    def save_data_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

# Example usage:
# collector = MarketDataCollector(api_key='your_api_key_here')
# data = collector.get_market_data('AAPL')
# collector.save_data_to_csv(data, 'market_data.csv')