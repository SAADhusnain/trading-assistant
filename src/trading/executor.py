class TradeExecutor:
    def __init__(self, trading_strategy):
        self.trading_strategy = trading_strategy

    def execute_trade(self, signal):
        if signal == "buy":
            self.buy()
        elif signal == "sell":
            self.sell()
        else:
            print("No valid trading signal received.")

    def buy(self):
        print("Executing buy order...")

    def sell(self):
        print("Executing sell order...")