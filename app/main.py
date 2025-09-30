"""
Streamlit Dashboard for AI Trading Assistant
Run with: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.data.processor import DataProcessor

# Page config
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(symbol, interval, period):
    """Load and cache data"""
    collector = DataCollector()
    df = collector.fetch_historical_data(symbol, interval, period, save=False)
    return df


@st.cache_data
def process_data(df):
    """Process and cache data"""
    processor = DataProcessor()
    df_clean = processor.clean_data(df)
    df_indicators = processor.add_technical_indicators(df_clean)
    # Don't create labels for dashboard, just return indicators
    df_indicators.dropna(inplace=True)
    return df_indicators


def create_candlestick_chart(df):
    """Create candlestick chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Bollinger Bands', 'Volume', 'RSI'),
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
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='gray', dash='dash', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle',
                      line=dict(color='blue', dash='dash', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='gray', dash='dash', width=1)),
            row=1, col=1
        )
    
    # EMAs
    if 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open_ else 'green' 
              for close, open_ in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI_14'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Trading Assistant Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Symbol selection
    symbol = st.sidebar.text_input("Symbol", value="AAPL", 
                                    help="Stock ticker (AAPL, TSLA) or crypto (BTC-USD)")
    
    # Timeframe
    interval = st.sidebar.selectbox(
        "Interval",
        options=['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d'],
        index=4  # Default to 30m
    )
    
    period = st.sidebar.selectbox(
        "Period",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y'],
        index=2  # Default to 1mo
    )
    
    st.sidebar.markdown("---")
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch Latest Data", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                df = load_data(symbol, interval, period)
                st.session_state['df'] = df
                st.session_state['symbol'] = symbol
                st.success(f"✅ Fetched {len(df)} records for {symbol}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return
    
    # Check if data exists
    if 'df' not in st.session_state:
        st.info("👆 Click 'Fetch Latest Data' to begin")
        
        # Show example
        st.markdown("## 📚 How to Use")
        st.markdown("""
        1. **Enter a symbol** (e.g., AAPL, TSLA, BTC-USD)
        2. **Select interval** (1m, 5m, 15m, 30m, 1h, 1d)
        3. **Select period** (1d, 5d, 1mo, 3mo, etc.)
        4. **Click 'Fetch Latest Data'**
        5. View charts and analysis below
        """)
        return
    
    df = st.session_state['df']
    symbol = st.session_state['symbol']
    
    # Process data
    with st.spinner("Calculating indicators..."):
        try:
            df_processed = process_data(df)
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return
    
    # Current metrics
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{symbol} Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].mean()
        volume_ratio = (volume / avg_volume - 1) * 100
        st.metric(
            label="Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_ratio:+.1f}% vs avg"
        )
    
    with col3:
        if 'RSI_14' in df_processed.columns:
            rsi = df_processed['RSI_14'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                label="RSI (14)",
                value=f"{rsi:.1f}",
                delta=rsi_status
            )
    
    with col4:
        if 'MACD' in df_processed.columns:
            macd = df_processed['MACD'].iloc[-1]
            macd_signal = df_processed['MACD_Signal'].iloc[-1]
            macd_diff = macd - macd_signal
            st.metric(
                label="MACD",
                value=f"{macd:.2f}",
                delta=f"{macd_diff:+.2f}"
            )
    
    st.markdown("---")
    
    # Chart
    st.markdown("## 📈 Technical Analysis")
    chart = create_candlestick_chart(df_processed)
    st.plotly_chart(chart, use_container_width=True)
    
    # MACD Chart
    st.markdown("## 📊 MACD Indicator")
    if 'MACD' in df_processed.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_processed.index, y=df_processed['MACD'], 
                                       name='MACD', line=dict(color='blue', width=2)))
        fig_macd.add_trace(go.Scatter(x=df_processed.index, y=df_processed['MACD_Signal'], 
                                       name='Signal', line=dict(color='orange', width=2)))
        fig_macd.add_trace(go.Bar(x=df_processed.index, y=df_processed['MACD_Hist'], 
                                   name='Histogram', marker_color='gray'))
        fig_macd.update_layout(height=400, title='MACD')
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("## 📊 Market Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Price Statistics")
        price_stats = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        st.dataframe(price_stats, use_container_width=True)
    
    with col2:
        st.markdown("### 🔢 Technical Indicators (Current)")
        if len(df_processed) > 0:
            current_indicators = {
                'RSI (14)': df_processed['RSI_14'].iloc[-1] if 'RSI_14' in df_processed.columns else 'N/A',
                'MACD': df_processed['MACD'].iloc[-1] if 'MACD' in df_processed.columns else 'N/A',
                'BB Upper': df_processed['BB_Upper'].iloc[-1] if 'BB_Upper' in df_processed.columns else 'N/A',
                'BB Lower': df_processed['BB_Lower'].iloc[-1] if 'BB_Lower' in df_processed.columns else 'N/A',
                'EMA 20': df_processed['EMA_20'].iloc[-1] if 'EMA_20' in df_processed.columns else 'N/A',
                'Volume Ratio': df_processed['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df_processed.columns else 'N/A',
            }
            indicators_df = pd.DataFrame(list(current_indicators.items()), 
                                         columns=['Indicator', 'Value'])
            st.dataframe(indicators_df, use_container_width=True, hide_index=True)
    
    # Recent data
    st.markdown("---")
    st.markdown("## 📋 Recent Data")
    st.dataframe(df.tail(20), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🤖 AI Trading Assistant Dashboard</p>
        <p style='font-size: 0.8rem;'>
            Built with Streamlit • Real-time Market Data<br>
            ⚠️ For educational purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()