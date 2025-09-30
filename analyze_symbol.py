"""
Simple Symbol Analysis Script
Usage: python analyze_symbol.py BTC-USD
       python analyze_symbol.py AAPL
       python analyze_symbol.py TSLA --interval 1h
"""

import sys
import argparse
from src.data.collector import DataCollector
from src.data.processor import DataProcessor


def analyze_symbol(symbol: str, interval: str = '1h', period: str = '30d'):
    """
    Analyze any trading symbol and provide entry recommendations.
    
    Args:
        symbol: Trading symbol (e.g., BTC-USD, AAPL, TSLA)
        interval: Data interval (1m, 5m, 15m, 1h, 1d)
        period: Historical period (7d, 30d, 90d, 1y)
    """
    
    print(f"\n{'='*80}")
    print(f"🤖 ANALYZING {symbol}")
    print(f"{'='*80}\n")
    
    # 1. Fetch Data
    print(f"📊 Fetching {interval} data for last {period}...")
    try:
        collector = DataCollector()
        df = collector.fetch_historical_data(symbol, interval, period, save=False)
        print(f"✅ Loaded {len(df)} candles\n")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # 2. Process Data
    print("🔧 Calculating technical indicators...")
    try:
        processor = DataProcessor()
        df_clean = processor.clean_data(df)
        df_processed = processor.add_technical_indicators(df_clean)
        df_processed.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        df_processed.dropna(inplace=True)
        print(f"✅ Processed {len(df_processed)} candles with {len(df_processed.columns)} features\n")
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return
    
    # 3. Analyze Current State
    latest = df_processed.iloc[-1]
    current_price = latest['Close']
    
    print(f"💰 Current Price: ${current_price:,.4f}\n")
    
    # 4. Technical Analysis
    print("📈 TECHNICAL ANALYSIS")
    print("-" * 80)
    
    # RSI
    if 'RSI_14' in df_processed.columns:
        rsi = latest['RSI_14']
        if rsi < 30:
            rsi_signal = "🟢 OVERSOLD (Bullish)"
        elif rsi > 70:
            rsi_signal = "🔴 OVERBOUGHT (Bearish)"
        else:
            rsi_signal = "⚪ NEUTRAL"
        print(f"RSI (14):        {rsi:.2f} - {rsi_signal}")
    
    # MACD
    if 'MACD' in df_processed.columns:
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        if macd > macd_signal:
            macd_trend = "🟢 BULLISH"
        else:
            macd_trend = "🔴 BEARISH"
        print(f"MACD:            {macd:.4f} - {macd_trend}")
    
    # Bollinger Bands
    if 'BB_Position' in df_processed.columns:
        bb_pos = latest['BB_Position']
        if bb_pos < 0.2:
            bb_signal = "🟢 Near Lower Band (Bullish)"
        elif bb_pos > 0.8:
            bb_signal = "🔴 Near Upper Band (Bearish)"
        else:
            bb_signal = "⚪ Middle Zone"
        print(f"BB Position:     {bb_pos:.2f} - {bb_signal}")
    
    # Moving Averages
    if 'EMA_20' in df_processed.columns and 'EMA_50' in df_processed.columns:
        ema20 = latest['EMA_20']
        ema50 = latest['EMA_50']
        if ema20 > ema50:
            ma_trend = "🟢 UPTREND"
        else:
            ma_trend = "🔴 DOWNTREND"
        print(f"EMA Trend:       {ma_trend}")
    
    # 5. Volume Analysis
    print(f"\n📊 VOLUME ANALYSIS")
    print("-" * 80)
    
    current_volume = latest['Volume']
    avg_volume = df_processed['Volume'].rolling(20).mean().iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5:
        volume_signal = "🟢 HIGH (Strong interest)"
    elif volume_ratio < 0.5:
        volume_signal = "🔴 LOW (Weak interest)"
    else:
        volume_signal = "⚪ NORMAL"
    
    print(f"Current Volume:  {current_volume:,.0f}")
    print(f"Average Volume:  {avg_volume:,.0f}")
    print(f"Volume Ratio:    {volume_ratio:.2f}x - {volume_signal}")
    
    # 6. Calculate Overall Signal
    print(f"\n🎯 RECOMMENDATION")
    print("=" * 80)
    
    # Simple scoring
    score = 50  # Start neutral
    
    # RSI contribution
    if 'RSI_14' in df_processed.columns:
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15
    
    # MACD contribution
    if 'MACD' in df_processed.columns:
        if macd > macd_signal:
            score += 10
        else:
            score -= 10
    
    # BB contribution
    if 'BB_Position' in df_processed.columns:
        if bb_pos < 0.2:
            score += 10
        elif bb_pos > 0.8:
            score -= 10
    
    # Volume contribution
    if volume_ratio > 1.5:
        # High volume with price increase = bullish
        if df_processed['Close'].pct_change().iloc[-1] > 0:
            score += 10
        else:
            score -= 5
    
    # Determine signal
    if score >= 70:
        signal = "🚀 STRONG BUY"
        action = "BUY"
    elif score >= 55:
        signal = "📈 BUY"
        action = "BUY"
    elif score <= 30:
        signal = "🔻 STRONG SELL"
        action = "SELL"
    elif score <= 45:
        signal = "📉 SELL"
        action = "SELL"
    else:
        signal = "⏸️  HOLD/WAIT"
        action = "WAIT"
    
    print(f"\nSignal:          {signal}")
    print(f"Confidence:      {score:.1f}/100")
    print(f"Action:          {action}")
    
    # 7. Calculate Levels
    if action != "WAIT":
        print(f"\n💰 TRADING LEVELS")
        print("-" * 80)
        
        # Use ATR for stop loss and take profit
        if 'ATR_14' in df_processed.columns:
            atr = latest['ATR_14']
        else:
            atr = current_price * 0.02  # 2% default
        
        if action == "BUY":
            entry = current_price
            stop_loss = entry - (2 * atr)
            take_profit = entry + (3 * atr)
        else:  # SELL
            entry = current_price
            stop_loss = entry + (2 * atr)
            take_profit = entry - (3 * atr)
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        print(f"Entry Price:     ${entry:,.4f}")
        print(f"Stop Loss:       ${stop_loss:,.4f} ({((stop_loss/entry-1)*100):+.2f}%)")
        print(f"Take Profit:     ${take_profit:,.4f} ({((take_profit/entry-1)*100):+.2f}%)")
        print(f"Risk/Reward:     1:{rr_ratio:.2f}")
        
        # Position sizing suggestion
        account_size = 10000  # Example
        risk_pct = 0.02  # 2%
        risk_amount = account_size * risk_pct
        position_size = risk_amount / abs(entry - stop_loss)
        
        print(f"\n📊 POSITION SIZING (Based on $10,000 account, 2% risk)")
        print(f"Risk Amount:     ${risk_amount:,.2f}")
        print(f"Position Size:   {position_size:.4f} units")
        print(f"Position Value:  ${position_size * entry:,.2f}")
    
    # 8. Summary
    print(f"\n{'='*80}")
    print("📝 SUMMARY")
    print(f"{'='*80}")
    
    if score >= 60:
        print("✅ Strong bullish signals detected")
        print("   Consider entering a long position")
        print("   Use provided stop loss and take profit levels")
    elif score <= 40:
        print("⚠️  Bearish signals detected")
        print("   Consider staying out or shorting (if experienced)")
        print("   Market may continue lower")
    else:
        print("⏸️  Mixed signals detected")
        print("   Recommend waiting for clearer setup")
        print("   Monitor price action for better entry")
    
    print(f"\n⚠️  Always use proper risk management!")
    print(f"    Never risk more than 1-2% per trade\n")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze trading symbols')
    parser.add_argument('symbol', type=str, help='Trading symbol (e.g., BTC-USD, AAPL)')
    parser.add_argument('--interval', default='1h', help='Data interval (default: 1h)')
    parser.add_argument('--period', default='30d', help='Historical period (default: 30d)')
    
    args = parser.parse_args()
    
    analyze_symbol(args.symbol, args.interval, args.period)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments, run example
        print("Usage: python analyze_symbol.py SYMBOL [--interval 1h] [--period 30d]")
        print("\nExample:")
        print("  python analyze_symbol.py BTC-USD")
        print("  python analyze_symbol.py AAPL --interval 1d --period 90d")
        print("\nRunning demo with BTC-USD...")
        analyze_symbol('BTC-USD')
    else:
        main()