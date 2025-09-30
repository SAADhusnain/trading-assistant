# test_trading.py

import unittest
from src.trading.executor import TradingExecutor
from src.trading.strategy import TradingStrategy

class TestTradingExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = TradingExecutor()
        self.strategy = TradingStrategy()

    def test_execute_trade(self):
        result = self.executor.execute_trade("BUY", 100)
        self.assertTrue(result)

    def test_strategy_decision(self):
        decision = self.strategy.decide("bullish")
        self.assertEqual(decision, "BUY")

if __name__ == '__main__':
    unittest.main()