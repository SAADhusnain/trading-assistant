#!/usr/bin/env python3
"""
Analyze any symbol with news sentiment
Usage: python analyze_with_news.py BTC-USD
"""

import sys
from src.data.news_scraper import NewsScraperPro

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_with_news.py SYMBOL")
        print("Example: python analyze_with_news.py BTC-USD")
        return
    
    symbol = sys.argv[1]
    scraper = NewsScraperPro()
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing {symbol}...")
    print(f"{'='*60}\n")
    
    # Get market sentiment
    sentiment = scraper.get_market_sentiment(symbol)
    print(f"📰 News Sentiment: {sentiment['overall_sentiment']}")
    print(f"   Articles analyzed: {sentiment['article_count']}")
    print(f"   Confidence: {sentiment['confidence']:.1f}%")
    
    # Get prediction
    prediction = scraper.predict_market_direction(symbol)
    
    print(f"\n🎯 PREDICTION")
    print(f"   Direction: {prediction['direction']}")
    print(f"   Action: {prediction['action']}")
    print(f"   Confidence: {prediction['confidence']:.1f}%")
    
    print(f"\n📊 FACTORS")
    print(f"   Price Trend: {prediction['factors']['price_trend']}")
    print(f"   RSI: {prediction['factors']['rsi']:.1f}")
    print(f"   Volume: {prediction['factors']['volume_signal']}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()