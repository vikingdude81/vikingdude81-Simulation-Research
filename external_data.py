"""
External Data Collector for Bitcoin Price Prediction
======================================================
Fetches alternative data sources:
- Social sentiment (Twitter/Reddit simulation)
- Google Trends search interest
- Fear & Greed Index
- Market metrics

All data is cached locally for reuse.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / 'EXTERNAL_DATA_CACHE'
CACHE_DIR.mkdir(exist_ok=True)

class ExternalDataCollector:
    """Collects and caches external data sources"""
    
    def __init__(self, cache_hours=1):
        self.cache_hours = cache_hours
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        logging.info(f"📊 External Data Collector initialized (cache: {cache_hours}h)")
        logging.info(f"   Cache directory: {CACHE_DIR}")
    
    def _is_cache_valid(self, cache_file, max_age_hours):
        """Check if cache file exists and is recent enough"""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)
    
    def _load_cache(self, cache_file):
        """Load data from cache file"""
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def _save_cache(self, data, cache_file):
        """Save data to cache file"""
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    # =========================================================================
    # 1. CRYPTO FEAR & GREED INDEX
    # =========================================================================
    
    def get_fear_greed_index(self, days=90):
        """
        Get Fear & Greed Index from Alternative.me
        
        Returns:
            dict with 'value' (0-100), 'classification', 'timestamp'
        """
        cache_file = CACHE_DIR / 'fear_greed.json'
        
        if self._is_cache_valid(cache_file, self.cache_hours):
            logging.info("   ✓ Loading Fear & Greed from cache")
            return self._load_cache(cache_file)
        
        try:
            logging.info("   ⬇️  Fetching Fear & Greed Index...")
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['metadata']['error'] == 'null':
                result = {
                    'current': {
                        'value': int(data['data'][0]['value']),
                        'classification': data['data'][0]['value_classification'],
                        'timestamp': data['data'][0]['timestamp']
                    },
                    'historical': [
                        {
                            'value': int(d['value']),
                            'classification': d['value_classification'],
                            'timestamp': d['timestamp']
                        }
                        for d in data['data']
                    ]
                }
                
                self._save_cache(result, cache_file)
                logging.info(f"   ✅ Fear & Greed: {result['current']['value']} ({result['current']['classification']})")
                return result
            else:
                logging.warning("   ⚠️  Fear & Greed API returned error")
                return {'current': {'value': 50, 'classification': 'Neutral'}, 'historical': []}
        except Exception as e:
            logging.warning(f"   ⚠️  Failed to fetch Fear & Greed: {e}")
            return {'current': {'value': 50, 'classification': 'Neutral'}, 'historical': []}
    
    # =========================================================================
    # 2. GOOGLE TRENDS
    # =========================================================================
    
    def get_google_trends(self, keywords=['bitcoin', 'btc', 'cryptocurrency'], timeframe='now 7-d'):
        """
        Get Google Trends search interest
        
        Args:
            keywords: list of search terms
            timeframe: 'now 1-d', 'now 7-d', 'today 1-m', etc.
        
        Returns:
            dict with interest scores (0-100) for each keyword
        """
        cache_file = CACHE_DIR / f'google_trends_{timeframe.replace(" ", "_")}.json'
        
        if self._is_cache_valid(cache_file, self.cache_hours * 4):  # Cache longer (4h)
            logging.info("   ✓ Loading Google Trends from cache")
            return self._load_cache(cache_file)
        
        try:
            logging.info(f"   ⬇️  Fetching Google Trends for {keywords}...")
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            
            interest_over_time = pytrends.interest_over_time()
            
            if not interest_over_time.empty:
                # Get latest values
                latest = interest_over_time.iloc[-1]
                result = {
                    'timestamp': str(interest_over_time.index[-1]),
                    'keywords': {kw: int(latest[kw]) for kw in keywords if kw in latest},
                    'average': float(interest_over_time[keywords].mean().mean()),
                    'trend_7d': float(interest_over_time[keywords].iloc[-7:].mean().mean()) if len(interest_over_time) >= 7 else None
                }
                
                self._save_cache(result, cache_file)
                logging.info(f"   ✅ Google Trends: {result['keywords']}")
                return result
            else:
                logging.warning("   ⚠️  No Google Trends data available")
                return {'keywords': {kw: 50 for kw in keywords}, 'average': 50}
                
        except Exception as e:
            logging.warning(f"   ⚠️  Failed to fetch Google Trends: {e}")
            return {'keywords': {kw: 50 for kw in keywords}, 'average': 50}
    
    # =========================================================================
    # 3. SIMULATED SOCIAL SENTIMENT
    # =========================================================================
    
    def get_social_sentiment(self, simulate=True):
        """
        Get social media sentiment (simulated for now)
        
        In production, this would use Twitter API v2 with tweepy
        For now, we simulate based on recent price action
        
        Returns:
            dict with sentiment scores (-1 to +1)
        """
        cache_file = CACHE_DIR / 'social_sentiment.json'
        
        if self._is_cache_valid(cache_file, self.cache_hours):
            logging.info("   ✓ Loading Social Sentiment from cache")
            return self._load_cache(cache_file)
        
        if simulate:
            logging.info("   🎲 Generating simulated social sentiment...")
            
            # Simulate sentiment based on random walk with trend
            # In production: analyze actual tweets/reddit posts
            np.random.seed(int(datetime.now().timestamp()) % 10000)
            
            # Simulated metrics
            result = {
                'timestamp': datetime.now().isoformat(),
                'twitter': {
                    'sentiment_score': float(np.random.uniform(-0.3, 0.6)),  # Slightly bullish bias
                    'volume': int(np.random.uniform(5000, 20000)),
                    'positive_ratio': float(np.random.uniform(0.45, 0.75))
                },
                'reddit': {
                    'sentiment_score': float(np.random.uniform(-0.2, 0.5)),
                    'submissions': int(np.random.uniform(100, 500)),
                    'comments': int(np.random.uniform(1000, 5000))
                },
                'overall_sentiment': float(np.random.uniform(-0.2, 0.5)),
                'bullish_ratio': float(np.random.uniform(0.5, 0.7))
            }
            
            self._save_cache(result, cache_file)
            logging.info(f"   ✅ Social Sentiment: {result['overall_sentiment']:.2f} (simulated)")
            return result
        
        else:
            # TODO: Implement real Twitter API integration
            # Requires Twitter API keys
            logging.warning("   ⚠️  Real Twitter API not configured (use simulate=True)")
            return {
                'overall_sentiment': 0.0,
                'twitter': {'sentiment_score': 0.0},
                'reddit': {'sentiment_score': 0.0}
            }
    
    # =========================================================================
    # 4. EXCHANGE METRICS (from CoinGecko)
    # =========================================================================
    
    def get_exchange_metrics(self):
        """
        Get exchange-level metrics from CoinGecko
        
        Returns:
            dict with volume, liquidity, market cap data
        """
        cache_file = CACHE_DIR / 'exchange_metrics.json'
        
        if self._is_cache_valid(cache_file, self.cache_hours):
            logging.info("   ✓ Loading Exchange Metrics from cache")
            return self._load_cache(cache_file)
        
        try:
            logging.info("   ⬇️  Fetching Exchange Metrics from CoinGecko...")
            url = "https://api.coingecko.com/api/v3/coins/bitcoin"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            market_data = data.get('market_data', {})
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'market_cap_usd': market_data.get('market_cap', {}).get('usd', 0),
                'total_volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                'market_cap_rank': data.get('market_cap_rank', 1),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'atl': market_data.get('atl', {}).get('usd', 0)
            }
            
            self._save_cache(result, cache_file)
            logging.info(f"   ✅ Exchange Metrics: Volume ${result['total_volume_24h']/1e9:.1f}B")
            return result
            
        except Exception as e:
            logging.warning(f"   ⚠️  Failed to fetch Exchange Metrics: {e}")
            return {
                'market_cap_usd': 0,
                'total_volume_24h': 0,
                'price_change_24h': 0
            }
    
    def get_dominance_metrics(self):
        """
        Get BTC.D, USDT.D, ETH.D from CoinGecko global market data
        
        Returns:
            dict with dominance percentages
        """
        cache_file = CACHE_DIR / 'dominance_metrics.json'
        
        if self._is_cache_valid(cache_file, self.cache_hours):
            logging.info("   ✓ Loading Dominance Metrics from cache")
            return self._load_cache(cache_file)
        
        try:
            logging.info("   ⬇️  Fetching Dominance Metrics from CoinGecko...")
            url = "https://api.coingecko.com/api/v3/global"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            market_data = data.get('data', {})
            market_cap_percentage = market_data.get('market_cap_percentage', {})
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'btc_dominance': market_cap_percentage.get('btc', 50.0),
                'eth_dominance': market_cap_percentage.get('eth', 15.0),
                'usdt_dominance': market_cap_percentage.get('usdt', 5.0),
                'total_market_cap': market_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                'active_cryptocurrencies': market_data.get('active_cryptocurrencies', 0),
                'market_cap_change_24h': market_data.get('market_cap_change_percentage_24h_usd', 0)
            }
            
            self._save_cache(result, cache_file)
            logging.info(f"   ✅ Dominance: BTC={result['btc_dominance']:.2f}% | USDT={result['usdt_dominance']:.2f}% | ETH={result['eth_dominance']:.2f}%")
            return result
            
        except Exception as e:
            logging.warning(f"   ⚠️  Failed to fetch Dominance Metrics: {e}")
            return {
                'btc_dominance': 50.0,
                'eth_dominance': 15.0,
                'usdt_dominance': 5.0,
                'total_market_cap': 0
            }
    
    # =========================================================================
    # 5. ALL DATA COLLECTION
    # =========================================================================
    
    def collect_all(self):
        """
        Collect all external data sources
        
        Returns:
            dict with all metrics
        """
        logging.info("\n" + "="*70)
        logging.info("📊 COLLECTING EXTERNAL DATA")
        logging.info("="*70)
        
        start_time = time.time()
        
        # Collect all sources
        fear_greed = self.get_fear_greed_index()
        google_trends = self.get_google_trends()
        social_sentiment = self.get_social_sentiment(simulate=True)
        exchange_metrics = self.get_exchange_metrics()
        dominance_metrics = self.get_dominance_metrics()
        
        # Combine into single dict
        result = {
            'timestamp': datetime.now().isoformat(),
            'fear_greed': fear_greed.get('current', {}).get('value', 50) if fear_greed else 50,
            'fear_greed_class': fear_greed.get('current', {}).get('classification', 'Neutral') if fear_greed else 'Neutral',
            'google_trends_bitcoin': google_trends.get('keywords', {}).get('bitcoin', 50) if google_trends else 50,
            'google_trends_avg': google_trends.get('average', 50) if google_trends else 50,
            'social_sentiment': social_sentiment.get('overall_sentiment', 0) if social_sentiment else 0,
            'twitter_sentiment': social_sentiment.get('twitter', {}).get('sentiment_score', 0) if social_sentiment else 0,
            'reddit_sentiment': social_sentiment.get('reddit', {}).get('sentiment_score', 0) if social_sentiment else 0,
            'market_cap': exchange_metrics.get('market_cap_usd', 0) if exchange_metrics else 0,
            'volume_24h': exchange_metrics.get('total_volume_24h', 0) if exchange_metrics else 0,
            'price_change_7d': exchange_metrics.get('price_change_7d', 0) if exchange_metrics else 0,
            'price_change_30d': exchange_metrics.get('price_change_30d', 0) if exchange_metrics else 0,
            'btc_dominance': dominance_metrics.get('btc_dominance', 50.0) if dominance_metrics else 50.0,
            'eth_dominance': dominance_metrics.get('eth_dominance', 15.0) if dominance_metrics else 15.0,
            'usdt_dominance': dominance_metrics.get('usdt_dominance', 5.0) if dominance_metrics else 5.0,
            'total_market_cap': dominance_metrics.get('total_market_cap', 0) if dominance_metrics else 0,
            'market_cap_change_24h': dominance_metrics.get('market_cap_change_24h', 0) if dominance_metrics else 0,
        }
        
        elapsed = time.time() - start_time
        logging.info("="*70)
        logging.info(f"✅ EXTERNAL DATA COLLECTION COMPLETE ({elapsed:.1f}s)")
        logging.info("="*70)
        
        # Save summary
        summary_file = CACHE_DIR / 'latest_summary.json'
        self._save_cache(result, summary_file)
        
        return result
    
    def get_historical_cache(self):
        """Load all cached historical data"""
        cache_files = list(CACHE_DIR.glob('*.json'))
        logging.info(f"\n📁 Found {len(cache_files)} cached data files")
        return cache_files


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 EXTERNAL DATA COLLECTOR - TEST RUN")
    print("="*70)
    
    collector = ExternalDataCollector(cache_hours=1)
    
    # Collect all data
    data = collector.collect_all()
    
    # Display summary
    print("\n" + "="*70)
    print("📊 EXTERNAL DATA SUMMARY")
    print("="*70)
    print(f"\n💰 Market Metrics:")
    print(f"   Market Cap: ${data['market_cap']/1e9:.1f}B")
    print(f"   Volume 24h: ${data['volume_24h']/1e9:.1f}B")
    print(f"   Price Change 7d: {data['price_change_7d']:.2f}%")
    print(f"   Price Change 30d: {data['price_change_30d']:.2f}%")
    
    print(f"\n😱 Sentiment Indicators:")
    print(f"   Fear & Greed: {data['fear_greed']}/100 ({data['fear_greed_class']})")
    print(f"   Social Sentiment: {data['social_sentiment']:.2f}")
    print(f"   Twitter: {data['twitter_sentiment']:.2f}")
    print(f"   Reddit: {data['reddit_sentiment']:.2f}")
    
    print(f"\n🔍 Search Interest:")
    print(f"   Google Trends (Bitcoin): {data['google_trends_bitcoin']}/100")
    print(f"   Average Trends: {data['google_trends_avg']:.1f}/100")
    
    print("\n" + "="*70)
    print(f"💾 Data cached in: {CACHE_DIR}")
    print("="*70)
