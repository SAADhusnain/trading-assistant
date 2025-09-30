"""
ENHANCED STREAMLIT DASHBOARD
With Market Intelligence, Sentiment Analysis, and Entry Point Detection
Run: streamlit run app/enhanced_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.data.processor import DataProcessor

# Import new modules (from the code above - save them first)
# from src.data.sentiment_analyzer import NewsSentimentAnalyzer
# from src.data.market_intelligence import MarketIntelligence
# from src.strategies.entry_detector import EntryPointDetector

# Page config
st.set_page_config(
    page_title="AI Trading Intelligence",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .signal-buy {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
    }
    .signal-sell {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
    }
    .signal-hold {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_process_data(symbol, interval, period):
    """Load and process market data"""
    collector = DataCollector()
    df = collector.fetch_historical_data(symbol, interval, period, save=False)
    
    processor = DataProcessor()
    df_clean = processor.clean_data(df)
    df_processed = processor.add_technical_indicators(df_clean)
    df_processed.dropna(inplace=True)
    
    return df_processed


def analyze_market_intelligence(symbol):
    """Run complete market intelligence analysis"""
    # Placeholder - implement based on your saved modules
    intelligence = {
        'sentiment': {
            'overall_signal': 'BULLISH',
            'confidence': 72,
            'news_count': 15,
            'bullish_pct': 73,
            'bearish_pct': 27
        },
        'metrics': {
            'market_cap': 1200000000000,
            'total_volume': 45000000000,
            'long_ratio': 68.5,
            'short_ratio': 31.5,
            'ls_signal': 'BULLISH',
            'buy_sell_ratio': 1.24,
            'taker_signal': 'BULLISH',
            'funding_rate': 0.008,
            'funding_signal': 'BULLISH',
            'open_interest': 15000000000
        }
    }
    return intelligence


def calculate_entry_signals(df, intelligence):
    """Calculate entry point recommendation"""
    latest = df.iloc[-1]
    
    # Technical score
    tech_score = 50
    if 'RSI_14' in df.columns:
        rsi = latest['RSI_14']
        if rsi < 30:
            tech_score += 20
        elif rsi > 70:
            tech_score -= 20
    
    if 'MACD' in df.columns:
        if latest['MACD'] > latest['MACD_Signal']:
            tech_score += 15
        else:
            tech_score -= 15
    
    # Combine with sentiment
    sent_score = intelligence['sentiment']['confidence']
    if intelligence['sentiment']['overall_signal'] == 'BEARISH':
        sent_score = 100 - sent_score
    
    # Combine scores
    total_score = (tech_score * 0.5 + sent_score * 0.5)
    
    # Determine signal
    if total_score >= 70:
        signal = "STRONG BUY"
        action = "BUY"
        color_class = "signal-buy"
        emoji = "🚀"
    elif total_score >= 55:
        signal = "BUY"
        action = "BUY"
        color_class = "signal-buy"
        emoji = "📈"
    elif total_score <= 30:
        signal = "STRONG SELL"
        action = "SELL"
        color_class = "signal-sell"
        emoji = "🔻"
    elif total_score <= 45:
        signal = "SELL"
        action = "SELL"
        color_class = "signal-sell"
        emoji = "📉"
    else:
        signal = "HOLD"
        action = "WAIT"
        color_class = "signal-hold"
        emoji = "⏸️"
    
    current_price = latest['Close']
    atr = latest.get('ATR_14', current_price * 0.02)
    
    if action == "BUY":
        entry = current_price
        stop = entry - (2 * atr)
        target = entry + (3 * atr)
    elif action == "SELL":
        entry = current_price
        stop = entry + (2 * atr)
        target = entry - (3 * atr)
    else:
        entry = current_price
        stop = current_price
        target = current_price
    
    return {
        'signal': signal,
        'action': action,
        'emoji': emoji,
        'color_class': color_class,
        'confidence': total_score,
        'current_price': current_price,
        'entry_price': entry,
        'stop_loss': stop,
        'take_profit': target,
        'tech_score': tech_score,
        'sent_score': sent_score
    }


def main():
    # Header
    st.title("🚀 AI Trading Intelligence Dashboard")
    st.markdown("*Advanced Market Analysis with Sentiment, Metrics & Entry Detection*")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    symbol = st.sidebar.text_input("Symbol", value="BTC-USD")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        interval = st.selectbox("Interval", ['5m', '15m', '30m', '1h', '4h', '1d'], index=3)
    with col2:
        period = st.selectbox("Period", ['7d', '30d', '90d', '1y'], index=1)
    
    if st.sidebar.button("🔄 Analyze", type="primary"):
        with st.spinner("🔍 Running deep analysis..."):
            try:
                # Load data
                df = load_and_process_data(symbol, interval, period)
                st.session_state['df'] = df
                st.session_state['symbol'] = symbol
                
                # Get intelligence
                intelligence = analyze_market_intelligence(symbol)
                st.session_state['intelligence'] = intelligence
                
                # Calculate signals
                recommendation = calculate_entry_signals(df, intelligence)
                st.session_state['recommendation'] = recommendation
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return
    
    if 'df' not in st.session_state:
        st.info("👈 Configure settings and click 'Analyze' to begin")
        
        # Show features
        st.markdown("## 🎯 What This Dashboard Analyzes:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📰 News & Sentiment
            - Real-time news analysis
            - Social media sentiment
            - Bullish/Bearish indicators
            - Confidence scores
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Market Metrics
            - Market cap & volume
            - Open interest
            - Long/Short ratios
            - Funding rates
            - Taker buy/sell volume
            """)
        
        with col3:
            st.markdown("""
            ### 🎯 Entry Detection
            - Optimal entry points
            - Stop loss levels
            - Take profit targets
            - Risk/Reward ratios
            - Multi-signal confirmation
            """)
        
        return
    
    df = st.session_state['df']
    symbol = st.session_state['symbol']
    intelligence = st.session_state.get('intelligence', {})
    recommendation = st.session_state.get('recommendation', {})
    
    # Main Recommendation
    if recommendation:
        signal_html = f"""
        <div class="{recommendation['color_class']}">
            {recommendation['emoji']} {recommendation['signal']}
            <br><span style="font-size: 18px;">Confidence: {recommendation['confidence']:.1f}%</span>
        </div>
        """
        st.markdown(signal_html, unsafe_allow_html=True)
    
    # Entry Levels
    if recommendation and recommendation['action'] != 'WAIT':
        st.markdown("## 💰 Trading Levels")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${recommendation['current_price']:.4f}")
        with col2:
            st.metric("Entry Price", f"${recommendation['entry_price']:.4f}")
        with col3:
            st.metric("Stop Loss", f"${recommendation['stop_loss']:.4f}", 
                     delta=f"{((recommendation['stop_loss']/recommendation['entry_price']-1)*100):.2f}%")
        with col4:
            st.metric("Take Profit", f"${recommendation['take_profit']:.4f}",
                     delta=f"{((recommendation['take_profit']/recommendation['entry_price']-1)*100):.2f}%")
        
        # Risk/Reward
        risk = abs(recommendation['entry_price'] - recommendation['stop_loss'])
        reward = abs(recommendation['take_profit'] - recommendation['entry_price'])
        rr_ratio = reward / risk if risk > 0 else 0
        
        st.info(f"📊 **Risk/Reward Ratio:** 1:{rr_ratio:.2f}")
    
    st.markdown("---")
    
    # Score Breakdown
    st.markdown("## 📊 Analysis Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score visualization
        if recommendation:
            scores = {
                'Technical': recommendation['tech_score'],
                'Sentiment': recommendation['sent_score'],
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=['#667eea', '#764ba2'],
                text=[f"{v:.1f}%" for v in scores.values()],
                textposition='auto'
            ))
            fig.update_layout(
                title="Signal Strength by Category",
                yaxis_title="Score (0-100)",
                yaxis_range=[0, 100],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment gauge
        if 'sentiment' in intelligence:
            sent = intelligence['sentiment']
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sent['confidence'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Sentiment: {sent['overall_signal']}"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Market Intelligence
    st.markdown("---")
    st.markdown("## 🔍 Market Intelligence")
    
    if 'metrics' in intelligence:
        metrics = intelligence['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Market Cap", f"${metrics.get('market_cap', 0)/1e9:.2f}B")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("24h Volume", f"${metrics.get('total_volume', 0)/1e9:.2f}B")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            long_pct = metrics.get('long_ratio', 50)
            st.metric("Long Ratio", f"{long_pct:.1f}%", 
                     delta="Bullish" if long_pct > 60 else "Bearish" if long_pct < 40 else "Neutral")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            bs_ratio = metrics.get('buy_sell_ratio', 1.0)
            st.metric("Buy/Sell Ratio", f"{bs_ratio:.2f}",
                     delta="Bullish" if bs_ratio > 1.1 else "Bearish" if bs_ratio < 0.9 else "Neutral")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Open Interest & Funding
        col1, col2 = st.columns(2)
        
        with col1:
            oi = metrics.get('open_interest', 0)
            st.info(f"📊 **Open Interest:** ${oi/1e9:.2f}B")
        
        with col2:
            funding = metrics.get('funding_rate', 0)
            funding_signal = "Bearish (longs paying shorts)" if funding > 0.01 else "Bullish (shorts paying longs)" if funding < -0.01 else "Neutral"
            st.info(f"💰 **Funding Rate:** {funding:.4f}% ({funding_signal})")
    
    # Price Chart
    st.markdown("---")
    st.markdown("## 📈 Price Action")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Entry point marker
    if recommendation and recommendation['action'] != 'WAIT':
        fig.add_hline(
            y=recommendation['entry_price'],
            line_dash="solid",
            line_color="blue",
            annotation_text="Entry",
            row=1, col=1
        )
        fig.add_hline(
            y=recommendation['stop_loss'],
            line_dash="dash",
            line_color="red",
            annotation_text="Stop Loss",
            row=1, col=1
        )
        fig.add_hline(
            y=recommendation['take_profit'],
            line_dash="dash",
            line_color="green",
            annotation_text="Take Profit",
            row=1, col=1
        )
    
    # EMAs
    if 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if c < o else 'green' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple'), showlegend=False),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p style='font-size: 20px;'>🚀 AI Trading Intelligence Platform</p>
        <p>Multi-Signal Analysis • Market Intelligence • Entry Point Detection</p>
        <p style='font-size: 12px;'>⚠️ For educational and informational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()