#!/usr/bin/env python3
"""
Monitor top cryptocurrencies
Usage: python monitor_cryptos.py --top 20
"""

import argparse
from src.data.news_scraper import NewsScraperPro
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Monitor top cryptocurrencies')
    parser.add_argument('--top', type=int, default=10, help='Number of cryptos to monitor')
    args = parser.parse_args()
    
    print(f"\n🚀 Monitoring Top {args.top} Cryptocurrencies")
    print("="*60)
    
    scraper = NewsScraperPro()
    df = scraper.monitor_all_cryptos(top_n=args.top)
    
    if not df.empty:
        print("\n📊 Market Predictions:\n")
        for idx, row in df.iterrows():
            emoji = "🟢" if row['action'] == 'BUY' else "🔴" if row['action'] == 'SELL' else "⚪"
            print(f"{emoji} {row['symbol']:12} | {row['direction']:12} | Confidence: {row['confidence']:.1f}%")
    
    print("\n" + "="*60)
    
    # Show summary
    if not df.empty:
        buy_signals = len(df[df['action'] == 'BUY'])
        sell_signals = len(df[df['action'] == 'SELL'])
        
        print(f"\n📈 Summary:")
        print(f"   Buy Signals:  {buy_signals}")
        print(f"   Sell Signals: {sell_signals}")
        print(f"   Avg Confidence: {df['confidence'].mean():.1f}%")

if __name__ == "__main__":
    main()