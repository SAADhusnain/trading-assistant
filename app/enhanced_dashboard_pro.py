"""
PROFESSIONAL AI TRADING DASHBOARD
With Live News, Sentiment Analysis & 100 Cryptocurrency Support
Run: streamlit run enhanced_dashboard_pro.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your modules
from src.data.collector import DataCollector
from src.data.processor import DataProcessor
from src.data.news_scraper import NewsScraperPro

# Page configuration
st.set_page_config(
    page_title="AI Trading Pro - Market Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }
    .crypto-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .news-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .bullish-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .bearish-signal {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .neutral-signal {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: transparent;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'news_scraper' not in st.session_state:
    st.session_state.news_scraper = NewsScraperPro()
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'selected_cryptos' not in st.session_state:
    st.session_state.selected_cryptos = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Trading Intelligence Pro</h1>
    <p>Live News Analysis • 100 Cryptocurrencies • Market Direction Prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Market Selection
market_type = st.sidebar.radio(
    "Select Market",
    ["📊 Top Cryptocurrencies", "🎯 Custom Selection", "📈 Traditional Stocks"]
)

if market_type == "📊 Top Cryptocurrencies":
    # Top crypto selection
    top_n = st.sidebar.slider("Number of Cryptos to Monitor", 5, 50, 20)
    cryptos_to_monitor = st.session_state.news_scraper.crypto_symbols[:top_n]
    
elif market_type == "🎯 Custom Selection":
    # Custom crypto selection
    st.sidebar.markdown("**Select Cryptocurrencies:**")
    
    # Group cryptos by category
    categories = {
        "🏆 Top 10": ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 
                     'DOGE-USD', 'ADA-USD', 'TON-USD', 'AVAX-USD', 'DOT-USD'],
        "🔗 DeFi": ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'CRV-USD', 'COMP-USD', 
                   'SUSHI-USD', 'BAL-USD', '1INCH-USD', 'LDO-USD', 'GMX-USD'],
        "🎮 Gaming/NFT": ['AXS-USD', 'GALA-USD', 'ENJ-USD', 'MANA-USD', 'SAND-USD',
                         'ILV-USD', 'APE-USD', 'FLOW-USD', 'CHZ-USD', 'THETA-USD'],
        "🐕 Memecoins": ['DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'FLOKI-USD'],
        "⚡ Layer 2": ['MATIC-USD', 'ARB-USD', 'OP-USD', 'IMX-USD'],
        "🌐 Infrastructure": ['LINK-USD', 'GRT-USD', 'FIL-USD', 'AR-USD', 'OCEAN-USD']
    }
    
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        list(categories.keys()),
        default=["🏆 Top 10"]
    )
    
    cryptos_to_monitor = []
    for cat in selected_categories:
        cryptos_to_monitor.extend(categories[cat])
    cryptos_to_monitor = list(set(cryptos_to_monitor))[:30]  # Limit to 30
    
else:  # Traditional Stocks
    stock_symbols = st.sidebar.text_input(
        "Enter Stock Symbols (comma-separated)",
        value="AAPL,MSFT,GOOGL,TSLA,NVDA"
    ).split(',')
    cryptos_to_monitor = [s.strip() for s in stock_symbols]

# Refresh settings
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)

if st.sidebar.button("🔄 Refresh Now", type="primary"):
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# Auto refresh logic
if auto_refresh:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
    if time_since_refresh > refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Show countdown
    time_to_refresh = refresh_interval - time_since_refresh
    st.sidebar.info(f"⏱️ Next refresh in {time_to_refresh}s")

# Main Content Area
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Overview",
    "📰 Live News & Sentiment",
    "🎯 Predictions",
    "📈 Technical Analysis",
    "🚨 Alerts"
])

with tab1:
    st.header("📊 Market Overview")
    
    # Show monitored cryptos
    st.info(f"📍 Monitoring {len(cryptos_to_monitor)} assets")
    
    # Get predictions for all monitored assets
    with st.spinner("🔍 Analyzing markets..."):
        predictions_df = st.session_state.news_scraper.monitor_all_cryptos(
            top_n=min(len(cryptos_to_monitor), 20)
        )
    
    if not predictions_df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bullish_count = len(predictions_df[predictions_df['action'] == 'BUY'])
            st.metric("🟢 Bullish Signals", bullish_count)
        
        with col2:
            bearish_count = len(predictions_df[predictions_df['action'] == 'SELL'])
            st.metric("🔴 Bearish Signals", bearish_count)
        
        with col3:
            avg_confidence = predictions_df['confidence'].mean()
            st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col4:
            strong_signals = len(predictions_df[predictions_df['confidence'] > 70])
            st.metric("💪 Strong Signals", strong_signals)
        
        # Top opportunities
        st.subheader("🏆 Top Opportunities")
        
        for idx, row in predictions_df.head(5).iterrows():
            signal_class = "bullish-signal" if row['action'] == 'BUY' else "bearish-signal"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="crypto-card">
                    <h3>{row['symbol']}</h3>
                    <p>Direction: {row['direction']}</p>
                    <p>Confidence: {row['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if row['action'] == 'BUY':
                    st.success(f"🟢 {row['action']}")
                else:
                    st.error(f"🔴 {row['action']}")
            
            with col3:
                st.metric("Rank", f"#{int(row['rank'])}")
        
        # Full table
        st.subheader("📋 All Predictions")
        
        display_df = predictions_df[['symbol', 'direction', 'action', 'confidence', 'rank']]
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "direction": st.column_config.TextColumn("Direction", width="medium"),
                "action": st.column_config.TextColumn("Action", width="small"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "rank": st.column_config.NumberColumn("Rank", width="small", format="%d")
            }
        )

with tab2:
    st.header("📰 Live News & Market Sentiment")
    
    # Get latest news
    with st.spinner("📡 Fetching latest news..."):
        articles = st.session_state.news_scraper.scrape_all_sources(max_articles=50)
    
    if articles:
        # Overall market sentiment
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Overall Market Sentiment")
            
            # Analyze all articles
            sentiments = []
            for article in articles[:20]:
                text = f"{article['title']} {article['summary']}"
                sentiment = st.session_state.news_scraper.analyze_sentiment(text)
                sentiments.append(sentiment['signal'])
            
            bullish = sentiments.count('BULLISH')
            bearish = sentiments.count('BEARISH')
            neutral = sentiments.count('NEUTRAL')
            
            if bullish > bearish + neutral:
                st.markdown('<div class="bullish-signal">🚀 BULLISH MARKET</div>', 
                          unsafe_allow_html=True)
            elif bearish > bullish + neutral:
                st.markdown('<div class="bearish-signal">⚠️ BEARISH MARKET</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown('<div class="neutral-signal">⏸️ NEUTRAL MARKET</div>', 
                          unsafe_allow_html=True)
            
            # Sentiment pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Bullish', 'Bearish', 'Neutral'],
                values=[bullish, bearish, neutral],
                marker=dict(colors=['#38ef7d', '#ff6a00', '#00f2fe']),
                hole=0.3
            )])
            fig_pie.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📰 Latest Headlines")
            
            for article in articles[:10]:
                text = f"{article['title']} {article['summary']}"
                sentiment = st.session_state.news_scraper.analyze_sentiment(text)
                
                # Sentiment emoji
                if sentiment['signal'] == 'BULLISH':
                    emoji = "🟢"
                elif sentiment['signal'] == 'BEARISH':
                    emoji = "🔴"
                else:
                    emoji = "⚪"
                
                with st.expander(f"{emoji} {article['title'][:80]}..."):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Sentiment:** {sentiment['signal']} ({sentiment['confidence']:.1f}% confidence)")
                    if article['summary']:
                        st.write(f"**Summary:** {article['summary'][:200]}...")
                    if article['link']:
                        st.write(f"[Read More]({article['link']})")

with tab3:
    st.header("🎯 Market Direction Predictions")
    
    # Select symbol for detailed prediction
    selected_symbol = st.selectbox(
        "Select Asset for Detailed Analysis",
        cryptos_to_monitor,
        index=0
    )
    
    with st.spinner(f"🔮 Predicting market direction for {selected_symbol}..."):
        prediction = st.session_state.news_scraper.predict_market_direction(selected_symbol)
    
    # Display prediction
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if prediction['action'] == 'BUY':
            st.markdown('<div class="bullish-signal">📈 BUY SIGNAL</div>', 
                      unsafe_allow_html=True)
        elif prediction['action'] == 'SELL':
            st.markdown('<div class="bearish-signal">📉 SELL SIGNAL</div>', 
                      unsafe_allow_html=True)
        else:
            st.markdown('<div class="neutral-signal">⏸️ HOLD SIGNAL</div>', 
                      unsafe_allow_html=True)
    
    with col2:
        st.metric("Prediction", prediction['direction'])
        st.metric("Confidence", f"{prediction['confidence']:.1f}%")
    
    with col3:
        st.metric("Timeframe", prediction['timeframe'])
        st.metric("Score", f"{prediction['prediction_score']:.1f}")
    
    # Factors breakdown
    st.subheader("📊 Analysis Factors")
    
    factors = prediction['factors']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("News Sentiment", factors['news_sentiment'])
        st.metric("News Confidence", f"{factors['news_confidence']:.1f}%")
    
    with col2:
        st.metric("Price Trend", factors['price_trend'])
        st.metric("RSI", f"{factors['rsi']:.1f}")
    
    with col3:
        st.metric("Volume Signal", factors['volume_signal'])
        st.metric("Articles Analyzed", factors['article_count'])
    
    with col4:
        # Get current price
        try:
            ticker = yf.Ticker(selected_symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.4f}")
        except:
            st.metric("Current Price", "N/A")

with tab4:
    st.header("📈 Technical Analysis")
    
    # Select symbol
    ta_symbol = st.selectbox(
        "Select Asset for Technical Analysis",
        cryptos_to_monitor,
        index=0,
        key="ta_symbol"
    )
    
    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
    with col2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=3)
    
    # Fetch data
    with st.spinner(f"Loading data for {ta_symbol}..."):
        try:
            ticker = yf.Ticker(ta_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                # Create technical chart
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price', 'Volume', 'RSI', 'MACD'),
                    row_heights=[0.5, 0.15, 0.15, 0.2]
                )
                
                # Candlestick chart
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
                
                # Moving averages
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA50'] = df['Close'].rolling(50).mean()
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA20'], name='MA20', 
                             line=dict(color='orange', width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                             line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                
                # Volume
                colors = ['red' if c < o else 'green' 
                         for c, o in zip(df['Close'], df['Open'])]
                fig.add_trace(
                    go.Bar(x=df.index, y=df['Volume'], marker_color=colors,
                          showlegend=False),
                    row=2, col=1
                )
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'),
                             showlegend=False),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                # MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['Histogram'] = df['MACD'] - df['Signal']
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                             line=dict(color='blue')),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['Signal'], name='Signal',
                             line=dict(color='orange')),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Bar(x=df.index, y=df['Histogram'], name='Histogram',
                          marker_color='gray'),
                    row=4, col=1
                )
                
                fig.update_layout(height=900, showlegend=True)
                fig.update_xaxes(rangeslider_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics
                st.subheader("📊 Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${df['Close'].iloc[-1]:.4f}")
                    st.metric("24h Change", f"{((df['Close'].iloc[-1] - df['Close'].iloc[0])/df['Close'].iloc[0]*100):.2f}%")
                
                with col2:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                    rsi_signal = "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral"
                    st.metric("RSI Signal", rsi_signal)
                
                with col3:
                    st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                    avg_vol = df['Volume'].mean()
                    vol_ratio = df['Volume'].iloc[-1] / avg_vol
                    st.metric("Vol vs Avg", f"{vol_ratio:.2f}x")
                
                with col4:
                    macd_signal = "Bullish" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "Bearish"
                    st.metric("MACD Signal", macd_signal)
                    st.metric("MACD Value", f"{df['MACD'].iloc[-1]:.4f}")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")

with tab5:
    st.header("🚨 Breaking News & Alerts")
    
    # Get breaking news alerts
    with st.spinner("🔍 Scanning for breaking news..."):
        alerts = st.session_state.news_scraper.get_breaking_news_alerts()
    
    if alerts:
        st.warning(f"⚠️ {len(alerts)} Breaking News Alerts!")
        
        for alert in alerts[:10]:
            impact_color = "🔴" if alert['impact'] == 'HIGH' else "🟡"
            sentiment_emoji = "🟢" if alert['sentiment'] == 'BULLISH' else "🔴" if alert['sentiment'] == 'BEARISH' else "⚪"
            
            st.markdown(f"""
            <div class="news-alert">
                <h4>{impact_color} {alert['title']}</h4>
                <p><b>Source:</b> {alert['source']} | <b>Impact:</b> {alert['impact']} | <b>Sentiment:</b> {sentiment_emoji} {alert['sentiment']}</p>
                <p><b>Keywords:</b> {', '.join(alert['matched_keywords'][:3])}</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("✅ No breaking news alerts at this time")
    
    # High impact predictions
    st.subheader("⚡ High Confidence Predictions")
    
    if not predictions_df.empty:
        high_conf = predictions_df[predictions_df['confidence'] > 70]
        
        if not high_conf.empty:
            for idx, row in high_conf.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{row['symbol']}**")
                
                with col2:
                    if row['action'] == 'BUY':
                        st.success(f"BUY - {row['confidence']:.1f}%")
                    else:
                        st.error(f"SELL - {row['confidence']:.1f}%")
                
                with col3:
                    st.write(row['direction'])
        else:
            st.info("No high confidence predictions at this time")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <h3>🚀 AI Trading Intelligence Pro</h3>
    <p>Real-time analysis of 100+ cryptocurrencies • 26 news sources • Advanced ML predictions</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        ⚠️ <b>DISCLAIMER:</b> This tool is for educational and informational purposes only. 
        Not financial advice. Always do your own research and consult with financial professionals.
        Cryptocurrency trading carries high risk and you can lose all your investment.
    </p>
</div>
""", unsafe_allow_html=True)

# Debug info in sidebar (optional)
with st.sidebar.expander("🔧 Debug Info"):
    st.write(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    st.write(f"Monitoring: {len(cryptos_to_monitor)} assets")
    st.write(f"Session ID: {id(st.session_state)}")