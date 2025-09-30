# Configuration settings for the trading assistant

class Config:
    def __init__(self):
        self.api_key = "YOUR_API_KEY"
        self.api_secret = "YOUR_API_SECRET"
        self.base_url = "https://api.yourtradingplatform.com"
        self.data_refresh_interval = 60  # in seconds
        self.model_path = "src/models/trained_model.pkl"
        self.log_file = "logs/trading_assistant.log"
        self.strategy = "mean_reversion"  # or any other strategy you want to implement

config = Config()