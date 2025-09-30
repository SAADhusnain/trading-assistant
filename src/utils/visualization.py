"""
Visualization utilities for trading data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_price_chart(df: pd.DataFrame, title: str = "Price Chart", show_volume: bool = True):
    """
    Plot candlestick chart with volume.
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        show_volume: Whether to show volume subplot
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, 'Volume'),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # Candlestick
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
        
        # Volume bars
        colors = ['red' if c < o else 'green' 
                 for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                   marker_color=colors, showlegend=False),
            row=2, col=1
        )
    else:
        fig.add_trace(candlestick)
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig


def plot_indicators(df: pd.DataFrame, indicators: list):
    """
    Plot technical indicators.
    
    Args:
        df: DataFrame with indicator columns
        indicators: List of indicator names to plot
    """
    fig = make_subplots(
        rows=len(indicators), cols=1,
        shared_xaxes=True,
        subplot_titles=indicators,
        vertical_spacing=0.05
    )
    
    for i, indicator in enumerate(indicators, 1):
        if indicator in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[indicator], name=indicator),
                row=i, col=1
            )
    
    fig.update_layout(height=200*len(indicators), showlegend=False)
    return fig


def plot_backtest_results(equity_curve: list, trades_df: pd.DataFrame):
    """
    Plot backtest results including equity curve and trade distribution.
    
    Args:
        equity_curve: List of equity values over time
        trades_df: DataFrame with trade records
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Equity Curve',
            'Trade P&L Distribution',
            'Win/Loss Distribution',
            'Cumulative Returns'
        )
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(y=equity_curve, name='Equity', line=dict(color='blue')),
        row=1, col=1
    )
    
    # P&L distribution
    fig.add_trace(
        go.Histogram(x=trades_df['pnl'], name='P&L', nbinsx=30),
        row=1, col=2
    )
    
    # Win/Loss pie chart
    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] <= 0])
    fig.add_trace(
        go.Pie(labels=['Wins', 'Losses'], values=[wins, losses],
               marker=dict(colors=['green', 'red'])),
        row=2, col=1
    )
    
    # Cumulative returns
    cumulative = trades_df['pnl'].cumsum()
    fig.add_trace(
        go.Scatter(y=cumulative, name='Cumulative P&L', line=dict(color='green')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20):
    """
    Plot feature importance from tree-based models.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
    """
    top_features = feature_importance.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, features: list = None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with features
        features: List of features to include (None = all numeric columns)
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns
    
    corr = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=800,
        width=800
    )
    
    return fig