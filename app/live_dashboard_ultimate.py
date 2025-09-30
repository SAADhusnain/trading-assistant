"""
ULTIMATE LIVE TRADING DASHBOARD
Real-time auto-updating with ML predictions and news analysis
Run: streamlit run live_dashboard_ultimate.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import threading
import queue
from pathlib import Path
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from src.data.news_scraper import NewsScraperPro
from src.data.news_ml_trainer import NewsMLTrainer

# Page configuration
st.set_page_config(
    page_title="🚀 Live Crypto Trading AI",
    page_icon="💹",
    layout="wide"
)

# Initialize session state
if 'news_scraper' not in st.session_state:
    st.session_state.news_scraper = NewsScraperPro()
if 'ml_trainer' not in st.session_state:
    st.session_state.ml_trainer = NewsMLTrainer()
if 'selected_crypto' not in st.session_state:
    st.session_state.selected_crypto = 'BTC-USD'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = {}

# Complete list of 100 cryptocurrencies with proper names
CRYPTO_LIST = {
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD',
    'Binance Coin (BNB)': 'BNB-USD',
    'Solana (SOL)': 'SOL-USD',
    'XRP (XRP)': 'XRP-USD',
    'Dogecoin (DOGE)': 'DOGE-USD',
    'Cardano (ADA)': 'ADA-USD',
    'Toncoin (TON)': 'TON-USD',
    'Avalanche (AVAX)': 'AVAX-USD',
    'Polkadot (DOT)': 'DOT-USD',
    'Chainlink (LINK)': 'LINK-USD',
    'Polygon (MATIC)': 'MATIC-USD',
    'Litecoin (LTC)': 'LTC-USD',
    'Shiba Inu (SHIB)': 'SHIB-USD',
    'Bitcoin Cash (BCH)': 'BCH-USD',
    'Uniswap (UNI)': 'UNI4-USD',
    'Internet Computer (ICP)': 'ICP-USD',
    'Stellar (XLM)': 'XLM-USD',
    'NEAR Protocol (NEAR)': 'NEAR-USD',
    'Aptos (APT)': 'APT21794-USD',
    'Monero (XMR)': 'XMR-USD',
    'OKB (OKB)': 'OKB-USD',
    'Hedera (HBAR)': 'HBAR-USD',
    'Arbitrum (ARB)': 'ARB11841-USD',
    'Cosmos (ATOM)': 'ATOM-USD',
    'VeChain (VET)': 'VET-USD',
    'Mantle (MNT)': 'MNT27075-USD',
    'Immutable (IMX)': 'IMX10603-USD',
    'Render (RNDR)': 'RNDR-USD',
    'Maker (MKR)': 'MKR-USD',
    'Lido DAO (LDO)': 'LDO-USD',
    'Aave (AAVE)': 'AAVE-USD',
    'The Graph (GRT)': 'GRT-USD',
    'Optimism (OP)': 'OP-USD',
    'Algorand (ALGO)': 'ALGO-USD',
    'Fantom (FTM)': 'FTM-USD',
    'Sui (SUI)': 'SUI20947-USD',
    'Injective (INJ)': 'INJ-USD',
    'Thorchain (RUNE)': 'RUNE-USD',
    'dYdX (DYDX)': 'DYDX-USD',
    'Pepe (PEPE)': 'PEPE24478-USD',
    'Floki (FLOKI)': 'FLOKI-USD',
    'Gala (GALA)': 'GALA-USD',
    'Theta Network (THETA)': 'THETA-USD',
    'Axie Infinity (AXS)': 'AXS-USD',
    'Tezos (XTZ)': 'XTZ-USD',
    'ApeCoin (APE)': 'APE18876-USD',
    'Flow (FLOW)': 'FLOW-USD',
    'Chiliz (CHZ)': 'CHZ-USD',
    'Mina Protocol (MINA)': 'MINA-USD',
    'Kava (KAVA)': 'KAVA-USD',
    '1inch (1INCH)': '1INCH-USD',
    'Curve DAO (CRV)': 'CRV-USD',
    'Enjin Coin (ENJ)': 'ENJ-USD',
    'EOS (EOS)': 'EOS-USD',
    'Rocket Pool (RPL)': 'RPL-USD',
    'Ocean Protocol (OCEAN)': 'OCEAN-USD',
    'Frax Share (FXS)': 'FXS-USD',
    'Harmony (ONE)': 'ONE-USD',
    'Basic Attention Token (BAT)': 'BAT-USD',
    'Kusama (KSM)': 'KSM-USD',
    'NEO (NEO)': 'NEO-USD',
    'WOO Network (WOO)': 'WOO-USD',
    'SushiSwap (SUSHI)': 'SUSHI-USD',
    'Gnosis (GNO)': 'GNO-USD',
    'Compound (COMP)': 'COMP-USD',
    'GMX (GMX)': 'GMX11857-USD',
    'Balancer (BAL)': 'BAL-USD',
    'Serum (SRM)': 'SRM-USD',
    'Terra Classic (LUNC)': 'LUNC-USD',
    'Helium (HNT)': 'HNT-USD',
    'Celo (CELO)': 'CELO-USD',
    'Qtum (QTUM)': 'QTUM-USD',
    'ICON (ICX)': 'ICX-USD',
    'Ankr (ANKR)': 'ANKR-USD',
    'SafePal (SFP)': 'SFP-USD',
    'Reserve Rights (RSR)': 'RSR-USD',
    'Oasis Network (ROSE)': 'ROSE-USD',
    'StormX (STMX)': 'STMX-USD',
    'Vulcan Forged (PYR)': 'PYR-USD',
    'Fetch.ai (FET)': 'FET-USD',
    'Illuvium (ILV)': 'ILV-USD',
    'Bluzelle (BLZ)': 'BLZ-USD',
    'Nervos Network (CKB)': 'CKB-USD',
    'SingularityNET (AGIX)': 'AGIX-USD',
    'Aragon (ANT)': 'ANT-USD',
    'Ren (REN)': 'REN-USD',
    'Dent (DENT)': 'DENT-USD',
    'SKALE (SKL)': 'SKL-USD',
    'Orion Protocol (ORN)': 'ORN-USD',
    'COTI (COTI)': 'COTI-USD',
    'Mask Network (MASK)': 'MASK-USD',
    'MXC (MXC)': 'MXC-USD',
    'Civic (CVC)': 'CVC-USD',
    'Terra (LUNA)': 'LUNA-USD',
    'Power Ledger (POWR)': 'POWR-USD',
    'Ergo (ERG)': 'ERG-USD',
    'Badger DAO (BADGER)': 'BADGER-USD',
    'Ribbon Finance (RBN)': 'RBN-USD',
    'Polymath (POLY)': 'POLY-USD'
}

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #090979, #020024);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .live-indicator {
        background: #ff0000;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        animation: pulse 1.5s infinite;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .prediction-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        animation: slideIn 0.5s;
    }
    .prediction-sell {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        animation: slideIn 0.5s;
    }
    .prediction-hold {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        animation: slideIn 0.5s;
    }
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .news-item {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #00d4ff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .news-item:hover {
        transform: translateX(10px);
        background: rgba(255,255,255,0.1);
    }
    div[data-testid="stSelectbox"] > div:first-child {
        background: linear-gradient(90deg, #00d4ff, #090979);
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_live_price(symbol: str) -> dict:
    """Get live price data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if not data.empty:
            current = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[0]
            change = current - prev_close
            change_pct = (change / prev_close) * 100
            
            return {
                'price': current,
                'change': change,
                'change_pct': change_pct,
                'volume': data['Volume'].iloc[-1],
                'high': data['High'].max(),
                'low': data['Low'].min(),
                'timestamp': datetime.now()
            }
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        return None

def get_ml_prediction(symbol: str) -> dict:
    """Get ML model prediction"""
    try:
        # Check if model exists, if not train it
        if not st.session_state.ml_trainer.load_models(symbol):
            with st.spinner(f"🤖 Training AI model for {symbol}..."):
                training_data = st.session_state.ml_trainer.collect_training_data(
                    symbol, 
                    st.session_state.news_scraper,
                    lookback_days=7
                )
                if not training_data.empty:
                    st.session_state.ml_trainer.train_models(symbol, training_data)
        
        # Make prediction
        prediction = st.session_state.ml_trainer.predict_realtime(
            symbol,
            st.session_state.news_scraper
        )
        return prediction
    except Exception as e:
        # Fallback to simple prediction
        return st.session_state.news_scraper.predict_market_direction(symbol)

def update_data():
    """Function to update data in background"""
    while st.session_state.auto_refresh:
        try:
            symbol = st.session_state.selected_crypto
            
            # Get live price
            price_data = get_live_price(symbol)
            if price_data:
                if symbol not in st.session_state.price_history:
                    st.session_state.price_history[symbol] = []
                st.session_state.price_history[symbol].append(price_data)
                
                # Keep only last 100 points
                if len(st.session_state.price_history[symbol]) > 100:
                    st.session_state.price_history[symbol].pop(0)
            
            # Update prediction every minute
            if (datetime.now() - st.session_state.last_update).seconds > 60:
                prediction = get_ml_prediction(symbol)
                if symbol not in st.session_state.predictions_history:
                    st.session_state.predictions_history[symbol] = []
                st.session_state.predictions_history[symbol].append(prediction)
                st.session_state.last_update = datetime.now()
            
            # Removed time.sleep(5) to avoid blocking Streamlit's responsiveness
            # Periodic updates are handled by Streamlit's rerun logic in the main script
            
        except Exception as e:
            print(f"Update error: {e}")
            time.sleep(10)

# Start background thread for updates
if 'update_thread' not in st.session_state:
    st.session_state.update_thread = threading.Thread(target=update_data, daemon=True)
    st.session_state.update_thread.start()

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🚀 LIVE AI CRYPTO TRADING</h1>
    <span class="live-indicator">● LIVE</span>
</div>
""", unsafe_allow_html=True)

# Sidebar with crypto selection
st.sidebar.title("⚙️ Control Panel")

# Crypto dropdown
selected_name = st.sidebar.selectbox(
    "🪙 Select Cryptocurrency",
    options=list(CRYPTO_LIST.keys()),
    index=0,
    help="Choose from 100 cryptocurrencies"
)

# Update selected crypto
if CRYPTO_LIST[selected_name] != st.session_state.selected_crypto:
    st.session_state.selected_crypto = CRYPTO_LIST[selected_name]
    st.session_state.last_update = datetime.now() - timedelta(minutes=2)  # Force update

# Auto-refresh toggle
st.session_state.auto_refresh = st.sidebar.checkbox(
    "🔄 Auto Refresh", 
    value=st.session_state.auto_refresh,
    help="Automatically update data every 5 seconds"
)

# Refresh interval
refresh_rate = st.sidebar.slider(
    "⏱️ Refresh Rate (seconds)",
    min_value=5,
    max_value=60,
    value=5,
    step=5
)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now", type="primary", use_container_width=True):
    st.rerun()

# Training controls
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Model Controls")

if st.sidebar.button("🧠 Train/Retrain Model", use_container_width=True):
    with st.spinner("Training AI model..."):
        training_data = st.session_state.ml_trainer.collect_training_data(
            st.session_state.selected_crypto,
            st.session_state.news_scraper,
            lookback_days=7
        )
        if not training_data.empty:
            results = st.session_state.ml_trainer.train_models(
                st.session_state.selected_crypto,
                training_data
            )
            st.sidebar.success("✅ Model trained successfully!")
            st.sidebar.json(results)

# Display selected crypto info
symbol = st.session_state.selected_crypto

# Main content area with auto-refresh
placeholder = st.empty()

with placeholder.container():
    # Get current data
    price_data = get_live_price(symbol)
    prediction = get_ml_prediction(symbol)
    sentiment = st.session_state.news_scraper.get_market_sentiment(symbol)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if price_data:
            st.markdown(f"""
            <div class="price-card">
                <h2>{selected_name}</h2>
                <h1>${price_data['price']:.4f}</h1>
                <p>{price_data['change_pct']:+.2f}% ({price_data['change']:+.4f})</p>
                <small>Updated: {price_data['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if prediction and 'signal' in prediction:
            signal_class = "prediction-buy" if 'BUY' in prediction['signal'] else "prediction-sell" if 'SELL' in prediction['signal'] else "prediction-hold"
            st.markdown(f"""
            <div class="{signal_class}">
                <h3>AI SIGNAL</h3>
                <h1>{prediction['signal']}</h1>
                <p>Confidence: {prediction.get('confidence', 0):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.metric(
            "📰 News Sentiment",
            sentiment['overall_sentiment'],
            f"{sentiment['confidence']:.1f}% confidence"
        )
        st.metric(
            "📊 Articles Analyzed",
            sentiment['article_count']
        )
    
    with col4:
        if price_data:
            st.metric("📈 24h High", f"${price_data['high']:.4f}")
            st.metric("📉 24h Low", f"${price_data['low']:.4f}")
            st.metric("📊 Volume", f"{price_data['volume']:,.0f}")
    
    # Charts section
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Chart", "📰 News Feed", "🤖 AI Analysis", "📊 Technical"])
    
    with tab1:
        # Live price chart
        st.subheader(f"📈 Live Price Chart - {selected_name}")
        
        # Get more detailed price history
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period='1d', interval='1m')
        
        if not hist_data.empty:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['red' if c < o else 'green' 
                     for c, o in zip(hist_data['Close'], hist_data['Open'])]
            fig.add_trace(
                go.Bar(x=hist_data.index, y=hist_data['Volume'],
                      marker_color=colors, showlegend=False),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Price history sparkline
        if symbol in st.session_state.price_history and len(st.session_state.price_history[symbol]) > 1:
            prices = [p['price'] for p in st.session_state.price_history[symbol]]
            times = [p['timestamp'] for p in st.session_state.price_history[symbol]]
            
            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(
                x=times, y=prices,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#00d4ff', width=2)
            ))
            fig_spark.update_layout(
                height=200,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_spark, use_container_width=True)
    
    with tab2:
        # Live news feed
        st.subheader("📰 Live News Feed")
        
        # Get latest news
        articles = st.session_state.news_scraper.scrape_all_sources(max_articles=20)
        
        # Filter relevant news
        symbol_keywords = st.session_state.news_scraper._get_symbol_keywords(symbol)
        relevant_articles = []
        
        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()
            if any(keyword in text for keyword in symbol_keywords) or len(relevant_articles) < 5:
                sentiment_analysis = st.session_state.news_scraper.analyze_sentiment(text)
                article['sentiment'] = sentiment_analysis
                relevant_articles.append(article)
        
        # Display news
        for article in relevant_articles[:10]:
            sentiment_emoji = "🟢" if article['sentiment']['signal'] == 'BULLISH' else "🔴" if article['sentiment']['signal'] == 'BEARISH' else "⚪"
            
            st.markdown(f"""
            <div class="news-item">
                <h4>{sentiment_emoji} {article['title']}</h4>
                <p><b>Source:</b> {article['source']} | <b>Sentiment:</b> {article['sentiment']['signal']}</p>
                <p>{article['summary'][:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # AI Analysis
        st.subheader("🤖 AI Analysis & Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Prediction Factors")
            if prediction and 'factors' in prediction:
                factors = prediction['factors']
                
                # Create gauge chart for confidence
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction.get('confidence', 0),
                    title={'text': "AI Confidence"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Display factors
                st.metric("News Sentiment", factors.get('news_sentiment', 'N/A'))
                st.metric("Price Trend", factors.get('price_trend', 'N/A'))
                st.metric("RSI", f"{factors.get('rsi', 50):.1f}")
                st.metric("Volume Signal", factors.get('volume_signal', 'N/A'))
        
        with col2:
            st.markdown("### 📈 Probability Distribution")
            if prediction and 'probability_up' in prediction:
                # Pie chart of probabilities
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Probability UP', 'Probability DOWN'],
                    values=[prediction['probability_up'], prediction['probability_down']],
                    marker=dict(colors=['#38ef7d', '#ff6a00']),
                    hole=0.3
                )])
                fig_pie.update_layout(
                    height=400,
                    template='plotly_dark',
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Model performance
        st.markdown("### 🎯 Model Performance")
        if st.session_state.ml_trainer.training_history:
            for history in st.session_state.ml_trainer.training_history:
                if history['symbol'] == symbol:
                    st.write(f"**Last Training:** {history['timestamp']}")
                    st.write(f"**Samples Used:** {history['samples']}")
                    st.write(f"**Features:** {history['features']}")
                    
                    # Display accuracy
                    if 'results' in history:
                        results_df = pd.DataFrame(history['results']).T
                        st.dataframe(results_df)
    
    with tab4:
        # Technical indicators
        st.subheader("📊 Technical Analysis")
        
        # Get longer history for technical analysis
        hist_1d = ticker.history(period='1d', interval='5m')
        
        if not hist_1d.empty:
            # Calculate indicators
            # RSI
            delta = hist_1d['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist_1d['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = hist_1d['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist_1d['Close'].ewm(span=26, adjust=False).mean()
            hist_1d['MACD'] = exp1 - exp2
            hist_1d['Signal'] = hist_1d['MACD'].ewm(span=9, adjust=False).mean()
            
            # Create subplots
            fig_tech = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price with MA', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price with moving averages
            fig_tech.add_trace(
                go.Scatter(x=hist_1d.index, y=hist_1d['Close'],
                          name='Price', line=dict(color='#00d4ff')),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(x=hist_1d.index, y=hist_1d['Close'].rolling(20).mean(),
                          name='MA20', line=dict(color='orange')),
                row=1, col=1
            )
            
            # RSI
            fig_tech.add_trace(
                go.Scatter(x=hist_1d.index, y=hist_1d['RSI'],
                          name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            fig_tech.add_trace(
                go.Scatter(x=hist_1d.index, y=hist_1d['MACD'],
                          name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig_tech.add_trace(
                go.Scatter(x=hist_1d.index, y=hist_1d['Signal'],
                          name='Signal', line=dict(color='red')),
                row=3, col=1
            )
            
            fig_tech.update_layout(
                height=800,
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)
    
    # Status bar at bottom
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.auto_refresh:
            st.success("🟢 Auto-refresh: ON")
        else:
            st.error("🔴 Auto-refresh: OFF")
    
    with col2:
        st.info(f"⏱️ Refresh rate: {refresh_rate}s")
    
    with col3:
        st.info(f"🕐 Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    with col4:
        st.info(f"📊 Monitoring: {selected_name}")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><b>🚀 Live AI Crypto Trading System</b></p>
    <p>Real-time predictions • 100 cryptocurrencies • News analysis • ML models</p>
    <p style='font-size: 0.8rem;'>⚠️ Not financial advice. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)