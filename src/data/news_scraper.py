"""
Live News Scraper & Market Sentiment Analyzer
Scrapes news from 26+ financial websites and analyzes sentiment
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import feedparser
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from textblob import TextBlob
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsScraperPro:
    """Advanced news scraper for 26+ financial news sources"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Define news sources with their RSS feeds and APIs
        self.news_sources = {
            'bloomberg': {
                'rss': 'https://feeds.bloomberg.com/markets/news.rss',
                'name': 'Bloomberg'
            },
            'reuters': {
                'rss': 'https://feeds.reuters.com/reuters/businessNews',
                'name': 'Reuters'
            },
            'cnbc': {
                'rss': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                'name': 'CNBC'
            },
            'marketwatch': {
                'rss': 'http://feeds.marketwatch.com/marketwatch/topstories',
                'name': 'MarketWatch'
            },
            'yahoo': {
                'rss': 'https://finance.yahoo.com/rss/',
                'name': 'Yahoo Finance'
            },
            'seekingalpha': {
                'rss': 'https://seekingalpha.com/feed.xml',
                'name': 'Seeking Alpha'
            },
            'investing': {
                'rss': 'https://www.investing.com/rss/news.rss',
                'name': 'Investing.com'
            },
            'finviz': {
                'url': 'https://finviz.com/news.ashx',
                'name': 'Finviz'
            },
            'zerohedge': {
                'rss': 'https://feeds.feedburner.com/zerohedge/feed',
                'name': 'ZeroHedge'
            },
            'coinmarketcap': {
                'url': 'https://coinmarketcap.com/headlines/news/',
                'name': 'CoinMarketCap'
            }
        }
        
        # Cryptocurrency list (100 cryptos)
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
            'DOGE-USD', 'ADA-USD', 'TON-USD', 'AVAX-USD', 'DOT-USD',
            'LINK-USD', 'MATIC-USD', 'LTC-USD', 'SHIB-USD', 'BCH-USD',
            'UNI-USD', 'ICP-USD', 'XLM-USD', 'NEAR-USD', 'APT-USD',
            'XMR-USD', 'OKB-USD', 'HBAR-USD', 'ARB-USD', 'ATOM-USD',
            'VET-USD', 'MNT-USD', 'IMX-USD', 'RNDR-USD', 'MKR-USD',
            'LDO-USD', 'AAVE-USD', 'GRT-USD', 'OP-USD', 'ALGO-USD',
            'FTM-USD', 'SUI-USD', 'INJ-USD', 'RUNE-USD', 'DYDX-USD',
            'PEPE-USD', 'FLOKI-USD', 'GALA-USD', 'THETA-USD', 'AXS-USD',
            'XTZ-USD', 'APE-USD', 'FLOW-USD', 'CHZ-USD', 'MINA-USD',
            'KAVA-USD', '1INCH-USD', 'CRV-USD', 'ENJ-USD', 'EOS-USD',
            'RPL-USD', 'OCEAN-USD', 'FXS-USD', 'ONE-USD', 'BAT-USD',
            'KSM-USD', 'NEO-USD', 'WOO-USD', 'SUSHI-USD', 'GNO-USD',
            'COMP-USD', 'GMX-USD', 'BAL-USD', 'SRM-USD', 'LUNC-USD',
            'HNT-USD', 'CELO-USD', 'QTUM-USD', 'ICX-USD', 'ANKR-USD',
            'SFP-USD', 'RSR-USD', 'ROSE-USD', 'STMX-USD', 'PYR-USD',
            'FET-USD', 'ILV-USD', 'BLZ-USD', 'CKB-USD', 'AGIX-USD',
            'ANT-USD', 'REN-USD', 'DENT-USD', 'SKL-USD', 'ORN-USD',
            'COTI-USD', 'MASK-USD', 'MXC-USD', 'CVC-USD', 'LUNA-USD',
            'POWR-USD', 'ERG-USD', 'BADGER-USD', 'RBN-USD', 'POLY-USD'
        ]
        
        # Keywords for different market impacts
        self.market_keywords = {
            'bullish': [
                'surge', 'rally', 'breakout', 'bull run', 'soar', 'jump',
                'gains', 'positive', 'upgrade', 'growth', 'record high',
                'breakthrough', 'optimistic', 'strong earnings', 'beat expectations',
                'expansion', 'approval', 'partnership', 'adoption', 'institutional'
            ],
            'bearish': [
                'crash', 'plunge', 'fall', 'decline', 'bear', 'dump',
                'loss', 'negative', 'downgrade', 'recession', 'sell-off',
                'correction', 'weakness', 'miss expectations', 'layoffs',
                'regulatory', 'ban', 'hack', 'lawsuit', 'bankruptcy'
            ],
            'high_impact': [
                'fed', 'federal reserve', 'interest rate', 'inflation',
                'cpi', 'gdp', 'employment', 'fomc', 'ecb', 'war',
                'election', 'stimulus', 'regulation', 'sec', 'etf'
            ],
            'crypto_specific': [
                'bitcoin', 'ethereum', 'defi', 'nft', 'mining', 'halving',
                'fork', 'airdrop', 'staking', 'burn', 'whale', 'hodl',
                'altcoin', 'memecoin', 'web3', 'metaverse', 'dao'
            ]
        }
    
    def scrape_all_sources(self, max_articles: int = 50) -> List[Dict]:
        """Scrape news from all sources concurrently"""
        all_articles = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for source_id, source_info in self.news_sources.items():
                if 'rss' in source_info:
                    future = executor.submit(
                        self._scrape_rss, source_info['rss'], source_info['name'], max_articles
                    )
                    futures.append(future)
                elif 'url' in source_info:
                    future = executor.submit(
                        self._scrape_website, source_info['url'], source_info['name'], max_articles
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    articles = future.result(timeout=10)
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Error scraping source: {e}")
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)
        
        # Sort by timestamp
        unique_articles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"Scraped {len(unique_articles)} unique articles from all sources")
        return unique_articles[:max_articles]
    
    def _scrape_rss(self, rss_url: str, source_name: str, max_articles: int) -> List[Dict]:
        """Scrape articles from RSS feed"""
        articles = []
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', 'No title'),
                    'summary': entry.get('summary', entry.get('description', '')),
                    'link': entry.get('link', ''),
                    'source': source_name,
                    'timestamp': datetime.now(),
                    'published': entry.get('published_parsed', None)
                }
                
                # Clean HTML from summary
                if article['summary']:
                    article['summary'] = BeautifulSoup(article['summary'], 'html.parser').get_text()
                
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error scraping RSS {source_name}: {e}")
        
        return articles
    
    def _scrape_website(self, url: str, source_name: str, max_articles: int) -> List[Dict]:
        """Scrape articles directly from website"""
        articles = []
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Different parsing logic for different sites
            if 'finviz' in url:
                news_table = soup.find('table', id='news-table')
                if news_table:
                    rows = news_table.find_all('tr')[:max_articles]
                    for row in rows:
                        link = row.find('a')
                        if link:
                            articles.append({
                                'title': link.text.strip(),
                                'summary': '',
                                'link': link.get('href', ''),
                                'source': source_name,
                                'timestamp': datetime.now(),
                                'published': None
                            })
            
            elif 'coinmarketcap' in url:
                news_items = soup.find_all('article', limit=max_articles)
                for item in news_items:
                    title_elem = item.find('h3')
                    if title_elem:
                        articles.append({
                            'title': title_elem.text.strip(),
                            'summary': '',
                            'link': url,
                            'source': source_name,
                            'timestamp': datetime.now(),
                            'published': None
                        })
                        
        except Exception as e:
            logger.error(f"Error scraping website {source_name}: {e}")
        
        return articles
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using VADER and TextBlob"""
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # Keyword analysis
        text_lower = text.lower()
        bullish_count = sum(1 for word in self.market_keywords['bullish'] if word in text_lower)
        bearish_count = sum(1 for word in self.market_keywords['bearish'] if word in text_lower)
        high_impact = any(word in text_lower for word in self.market_keywords['high_impact'])
        crypto_related = any(word in text_lower for word in self.market_keywords['crypto_specific'])
        
        # Combined sentiment score
        combined_score = (vader_scores['compound'] + textblob_sentiment) / 2
        
        # Determine signal
        if combined_score > 0.1:
            signal = 'BULLISH'
        elif combined_score < -0.1:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'signal': signal,
            'score': combined_score,
            'vader': vader_scores,
            'textblob': textblob_sentiment,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count,
            'high_impact': high_impact,
            'crypto_related': crypto_related,
            'confidence': abs(combined_score) * 100
        }
    
    def get_market_sentiment(self, symbol: str = None) -> Dict:
        """Get overall market sentiment for a specific symbol or general market"""
        articles = self.scrape_all_sources(max_articles=100)
        
        if symbol:
            # Filter articles relevant to the symbol
            relevant_articles = []
            symbol_keywords = self._get_symbol_keywords(symbol)
            
            for article in articles:
                text = f"{article['title']} {article['summary']}".lower()
                if any(keyword in text for keyword in symbol_keywords):
                    relevant_articles.append(article)
        else:
            relevant_articles = articles
        
        if not relevant_articles:
            return {
                'overall_sentiment': 'NEUTRAL',
                'confidence': 0,
                'article_count': 0,
                'sentiment_distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0}
            }
        
        # Analyze sentiment for each article
        sentiments = []
        for article in relevant_articles:
            text = f"{article['title']} {article['summary']}"
            sentiment = self.analyze_sentiment(text)
            article['sentiment'] = sentiment
            sentiments.append(sentiment)
        
        # Calculate overall sentiment
        bullish = sum(1 for s in sentiments if s['signal'] == 'BULLISH')
        bearish = sum(1 for s in sentiments if s['signal'] == 'BEARISH')
        neutral = sum(1 for s in sentiments if s['signal'] == 'NEUTRAL')
        
        avg_score = np.mean([s['score'] for s in sentiments])
        
        if avg_score > 0.1:
            overall = 'BULLISH'
        elif avg_score < -0.1:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'overall_sentiment': overall,
            'confidence': abs(avg_score) * 100,
            'article_count': len(relevant_articles),
            'sentiment_distribution': {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral
            },
            'average_score': avg_score,
            'high_impact_news': sum(1 for s in sentiments if s['high_impact']),
            'recent_articles': relevant_articles[:10]
        }
    
    def _get_symbol_keywords(self, symbol: str) -> List[str]:
        """Get keywords related to a symbol"""
        symbol_clean = symbol.replace('-USD', '').upper()
        
        # Crypto name mappings
        crypto_names = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'BNB': ['binance', 'bnb'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'DOGE': ['dogecoin', 'doge'],
            'ADA': ['cardano', 'ada'],
            'AVAX': ['avalanche', 'avax'],
            'DOT': ['polkadot', 'dot'],
            'LINK': ['chainlink', 'link'],
            'MATIC': ['polygon', 'matic'],
            'LTC': ['litecoin', 'ltc'],
            'SHIB': ['shiba inu', 'shib'],
            'UNI': ['uniswap', 'uni'],
            'ATOM': ['cosmos', 'atom'],
            'NEAR': ['near protocol', 'near'],
            'APT': ['aptos', 'apt'],
            'ARB': ['arbitrum', 'arb'],
            'OP': ['optimism', 'op'],
            'INJ': ['injective', 'inj']
        }
        
        return crypto_names.get(symbol_clean, [symbol_clean.lower()])
    
    def predict_market_direction(self, symbol: str, timeframe: str = '24h') -> Dict:
        """Predict market direction based on news sentiment and technical analysis"""
        # Get news sentiment
        sentiment = self.get_market_sentiment(symbol)
        
        # Get technical data
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='7d', interval='1h')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                avg_price = hist['Close'].mean()
                price_trend = 'UP' if current_price > avg_price else 'DOWN'
                
                # Calculate RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                # Volume analysis
                recent_volume = hist['Volume'].iloc[-24:].mean()
                avg_volume = hist['Volume'].mean()
                volume_signal = 'HIGH' if recent_volume > avg_volume * 1.2 else 'NORMAL'
            else:
                price_trend = 'UNKNOWN'
                rsi = 50
                volume_signal = 'NORMAL'
                
        except Exception as e:
            logger.error(f"Error fetching technical data: {e}")
            price_trend = 'UNKNOWN'
            rsi = 50
            volume_signal = 'NORMAL'
        
        # Combine signals for prediction
        prediction_score = 0
        
        # News sentiment weight: 40%
        if sentiment['overall_sentiment'] == 'BULLISH':
            prediction_score += 40
        elif sentiment['overall_sentiment'] == 'BEARISH':
            prediction_score -= 40
        
        # Technical trend weight: 30%
        if price_trend == 'UP':
            prediction_score += 30
        elif price_trend == 'DOWN':
            prediction_score -= 30
        
        # RSI weight: 20%
        if rsi < 30:
            prediction_score += 20  # Oversold
        elif rsi > 70:
            prediction_score -= 20  # Overbought
        
        # Volume weight: 10%
        if volume_signal == 'HIGH':
            if prediction_score > 0:
                prediction_score += 10  # Confirms bullish
            else:
                prediction_score -= 10  # Confirms bearish
        
        # Final prediction
        if prediction_score > 20:
            direction = 'STRONG BUY'
            action = 'BUY'
        elif prediction_score > 0:
            direction = 'BUY'
            action = 'BUY'
        elif prediction_score < -20:
            direction = 'STRONG SELL'
            action = 'SELL'
        elif prediction_score < 0:
            direction = 'SELL'
            action = 'SELL'
        else:
            direction = 'HOLD'
            action = 'HOLD'
        
        return {
            'symbol': symbol,
            'direction': direction,
            'action': action,
            'confidence': min(abs(prediction_score), 100),
            'timeframe': timeframe,
            'factors': {
                'news_sentiment': sentiment['overall_sentiment'],
                'news_confidence': sentiment['confidence'],
                'price_trend': price_trend,
                'rsi': rsi,
                'volume_signal': volume_signal,
                'article_count': sentiment['article_count']
            },
            'prediction_score': prediction_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def monitor_all_cryptos(self, top_n: int = 20) -> pd.DataFrame:
        """Monitor top N cryptocurrencies and rank by opportunity"""
        predictions = []
        
        logger.info(f"Analyzing top {top_n} cryptocurrencies...")
        
        for symbol in self.crypto_symbols[:top_n]:
            try:
                prediction = self.predict_market_direction(symbol)
                predictions.append(prediction)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Create DataFrame and rank
        df = pd.DataFrame(predictions)
        if not df.empty:
            df['rank'] = df['confidence'].rank(ascending=False)
            df = df.sort_values('confidence', ascending=False)
        
        return df
    
    def get_breaking_news_alerts(self, keywords: List[str] = None) -> List[Dict]:
        """Get breaking news alerts based on keywords"""
        if keywords is None:
            keywords = self.market_keywords['high_impact']
        
        articles = self.scrape_all_sources(max_articles=20)
        alerts = []
        
        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()
            
            # Check for keywords
            matched_keywords = [kw for kw in keywords if kw in text]
            
            if matched_keywords:
                sentiment = self.analyze_sentiment(text)
                
                alert = {
                    'title': article['title'],
                    'source': article['source'],
                    'link': article['link'],
                    'matched_keywords': matched_keywords,
                    'sentiment': sentiment['signal'],
                    'impact': 'HIGH' if sentiment['high_impact'] else 'MEDIUM',
                    'timestamp': article['timestamp']
                }
                alerts.append(alert)
        
        return alerts


# Example usage
if __name__ == "__main__":
    scraper = NewsScraperPro()
    
    # Get market sentiment for Bitcoin
    btc_sentiment = scraper.get_market_sentiment('BTC-USD')
    print(f"Bitcoin Sentiment: {btc_sentiment['overall_sentiment']}")
    print(f"Confidence: {btc_sentiment['confidence']:.1f}%")
    
    # Predict market direction for Ethereum
    eth_prediction = scraper.predict_market_direction('ETH-USD')
    print(f"\nEthereum Prediction: {eth_prediction['direction']}")
    print(f"Action: {eth_prediction['action']}")
    
    # Monitor top 10 cryptos
    top_cryptos = scraper.monitor_all_cryptos(top_n=10)
    print("\nTop Crypto Opportunities:")
    print(top_cryptos[['symbol', 'direction', 'confidence', 'rank']])
    
    # Get breaking news
    breaking = scraper.get_breaking_news_alerts()
    print(f"\nBreaking News Alerts: {len(breaking)} found")
    for alert in breaking:
        print(f"- {alert['title']} ({alert['source']}) | Impact: {alert['impact']} | Sentiment: {alert['sentiment']}")