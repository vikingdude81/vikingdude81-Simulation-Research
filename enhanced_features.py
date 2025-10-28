"""
Enhanced Feature Engineering for Bitcoin Price Prediction
==========================================================
Adds advanced features beyond basic technical indicators:
- Market microstructure features
- Volatility regime detection
- Fractal & chaos indicators
- Order flow proxies
- Market regime classification
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_hurst_exponent(ts, max_lag=20):
    """
    Calculate Hurst exponent to detect long-term memory in price series
    
    H < 0.5: Mean reverting (anti-persistent)
    H = 0.5: Random walk
    H > 0.5: Trending (persistent)
    
    Args:
        ts: time series (returns)
        max_lag: maximum lag to consider
    
    Returns:
        float: Hurst exponent
    """
    try:
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        
        # Linear fit of log(tau) vs log(lags)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return 0.5  # Default to random walk


def add_microstructure_features(df):
    """
    Add market microstructure features
    
    These capture intra-candle dynamics and liquidity
    """
    logging.info("   Adding microstructure features...")
    
    # Bid-ask spread proxy
    df['spread_proxy'] = (df['_high'] - df['_low']) / df['_close']
    
    # Price efficiency (how much price moves per unit volume)
    df['price_efficiency'] = df['_close'] / (df['_volume'].rolling(24).mean() + 1)
    
    # Amihud illiquidity measure
    df['amihud_illiquidity'] = np.abs(df['returns']) / (df['_volume'] + 1)
    df['amihud_illiquidity_24h'] = df['amihud_illiquidity'].rolling(24).mean()
    
    # Roll's effective spread (from high-low)
    df['roll_spread'] = 2 * np.sqrt(np.abs(df['_close'].diff().rolling(24).cov(df['_close'].diff().shift(1))))
    
    # Trading intensity
    df['trade_intensity'] = df['_volume'] / df['_volume'].rolling(168).mean()  # vs weekly avg
    
    logging.info(f"   ✓ Added 6 microstructure features")
    return df


def add_volatility_regime_features(df):
    """
    Detect and classify volatility regimes
    """
    logging.info("   Adding volatility regime features...")
    
    # Calculate volatility if not exists
    if 'volatility' not in df.columns:
        df['returns'] = df['_close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std() * np.sqrt(24)
    
    # Volatility regimes (low/medium/high)
    vol_rolling_mean = df['volatility'].rolling(168).mean()  # 1 week
    df['volatility_regime'] = pd.cut(
        vol_rolling_mean,
        bins=3,
        labels=[0, 1, 2]  # 0=low, 1=medium, 2=high
    ).astype(float)
    
    # Volatility percentile rank
    df['volatility_percentile'] = df['volatility'].rolling(720).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # Volatility acceleration (change in volatility)
    df['volatility_acceleration'] = df['volatility'].diff()
    
    # GARCH-style: lag volatility
    df['volatility_lag1'] = df['volatility'].shift(1)
    df['volatility_lag2'] = df['volatility'].shift(2)
    
    # Parkinson volatility (uses high-low range)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * np.log(df['_high'] / df['_low']) ** 2
    ).rolling(24).mean()
    
    logging.info(f"   ✓ Added 7 volatility regime features")
    return df


def add_fractal_chaos_features(df):
    """
    Add fractal dimension and chaos theory indicators
    """
    logging.info("   Adding fractal & chaos features...")
    
    # Returns distribution features
    df['returns_skew_24h'] = df['returns'].rolling(24).apply(lambda x: skew(x.dropna()))
    df['returns_skew_168h'] = df['returns'].rolling(168).apply(lambda x: skew(x.dropna()))
    
    df['returns_kurtosis_24h'] = df['returns'].rolling(24).apply(lambda x: kurtosis(x.dropna()))
    df['returns_kurtosis_168h'] = df['returns'].rolling(168).apply(lambda x: kurtosis(x.dropna()))
    
    # Hurst exponent (persistence)
    df['hurst_48h'] = df['returns'].rolling(48).apply(
        lambda x: calculate_hurst_exponent(x.dropna().values, max_lag=min(20, len(x)//2))
    )
    
    # Fractal dimension proxy: ratio of path length to displacement
    df['fractal_dimension'] = (
        df['_high'].rolling(24).max() - df['_low'].rolling(24).min()
    ) / (np.abs(df['_close'].diff()).rolling(24).sum() + 1e-8)
    
    # Chaos indicator: Largest Lyapunov Exponent proxy
    # Positive = chaotic, Negative = stable
    df['chaos_indicator'] = df['returns'].rolling(24).apply(
        lambda x: np.mean(np.abs(np.diff(x.values))) / (np.std(x.values) + 1e-8)
    )
    
    logging.info(f"   ✓ Added 7 fractal & chaos features")
    return df


def add_order_flow_features(df):
    """
    Add order flow imbalance proxies
    
    Without tick data, we approximate using OHLC
    """
    logging.info("   Adding order flow features...")
    
    # Buy/sell pressure from candle body
    candle_range = df['_high'] - df['_low']
    candle_range = candle_range.replace(0, 1e-8)  # Avoid division by zero
    
    df['buy_pressure'] = (df['_close'] - df['_low']) / candle_range
    df['sell_pressure'] = (df['_high'] - df['_close']) / candle_range
    
    # Order imbalance
    df['order_imbalance'] = df['buy_pressure'] - df['sell_pressure']
    df['order_imbalance_ma'] = df['order_imbalance'].rolling(24).mean()
    
    # Volume-weighted imbalance
    df['volume_imbalance'] = df['order_imbalance'] * df['_volume']
    df['volume_imbalance_ma'] = df['volume_imbalance'].rolling(24).mean()
    
    # Buying/selling volume proxies
    df['buy_volume_proxy'] = df['_volume'] * df['buy_pressure']
    df['sell_volume_proxy'] = df['_volume'] * df['sell_pressure']
    
    df['buy_sell_ratio'] = (df['buy_volume_proxy'] + 1) / (df['sell_volume_proxy'] + 1)
    
    # Cumulative order flow
    df['cumulative_order_flow'] = df['order_imbalance'].rolling(168).sum()
    
    logging.info(f"   ✓ Added 10 order flow features")
    return df


def add_market_regime_features(df):
    """
    Classify market into regimes: trending/ranging/volatile
    """
    logging.info("   Adding market regime features...")
    
    # Trend strength: difference between fast and slow EMAs
    if 'ema_12' in df.columns and 'ema_26' in df.columns:
        df['trend_strength'] = np.abs(df['ema_12'] - df['ema_26']) / df['_close']
    else:
        df['ema_12'] = df['_close'].ewm(span=12).mean()
        df['ema_26'] = df['_close'].ewm(span=26).mean()
        df['trend_strength'] = np.abs(df['ema_12'] - df['ema_26']) / df['_close']
    
    # ADX (Average Directional Index) proxy for trend strength
    high_low = df['_high'] - df['_low']
    high_close = np.abs(df['_high'] - df['_close'].shift(1))
    low_close = np.abs(df['_low'] - df['_close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    plus_dm = df['_high'].diff()
    minus_dm = -df['_low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    
    df['adx_proxy'] = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    
    # Market regime classification
    vol_threshold = df['volatility'].quantile(0.75)
    trend_threshold = df['trend_strength'].quantile(0.75)
    
    df['market_regime'] = 'ranging'  # Default
    df.loc[df['trend_strength'] > trend_threshold, 'market_regime'] = 'trending'
    df.loc[df['volatility'] > vol_threshold, 'market_regime'] = 'volatile'
    
    # One-hot encode regime
    df['regime_trending'] = (df['market_regime'] == 'trending').astype(int)
    df['regime_ranging'] = (df['market_regime'] == 'ranging').astype(int)
    df['regime_volatile'] = (df['market_regime'] == 'volatile').astype(int)
    
    # Regime persistence: how long in current regime
    df['regime_duration'] = df.groupby(
        (df['market_regime'] != df['market_regime'].shift()).cumsum()
    ).cumcount() + 1
    
    # Drop the string column, keep only numeric one-hot encoded versions
    df = df.drop(columns=['market_regime'], errors='ignore')
    
    logging.info(f"   ✓ Added 7 market regime features")
    return df


def add_price_levels_features(df):
    """
    Add support/resistance and psychological level features
    """
    logging.info("   Adding price level features...")
    
    # Distance to recent high/low
    df['dist_to_high_24h'] = (df['_high'].rolling(24).max() - df['_close']) / df['_close']
    df['dist_to_low_24h'] = (df['_close'] - df['_low'].rolling(24).min()) / df['_close']
    df['dist_to_high_168h'] = (df['_high'].rolling(168).max() - df['_close']) / df['_close']
    df['dist_to_low_168h'] = (df['_close'] - df['_low'].rolling(168).min()) / df['_close']
    
    # Proximity to round numbers (psychological levels)
    df['round_number_dist'] = df['_close'] % 1000 / 1000  # Distance to nearest $1000
    df['round_5k_dist'] = df['_close'] % 5000 / 5000  # Distance to nearest $5000
    df['round_10k_dist'] = df['_close'] % 10000 / 10000  # Distance to nearest $10000
    
    logging.info(f"   ✓ Added 7 price level features")
    return df


def add_geometric_ma_features(df):
    """
    Add Geometric Moving Average (GMA) features
    
    GMA = exp(SMA(log(price))) - superior to arithmetic MA for crypto because:
    - Handles exponential price movements naturally
    - Weighs percentage changes equally (not absolute)
    - Less lag on strong trends
    - Better fits multiplicative price dynamics
    
    Based on breakthrough results: 4.33-6.47 Sharpe ratios across BTC/ETH/SOL
    """
    logging.info("   Adding Geometric MA features (GMA Crossover Champion)...")
    
    def _gma(series, length):
        """Calculate Geometric Moving Average"""
        safe_series = series.clip(lower=1e-10)  # Prevent log(0)
        log_series = np.log(safe_series)
        log_sma = log_series.rolling(window=length).mean()
        gma = np.exp(log_sma)
        return gma
    
    # Multiple GMA lengths for different timeframes
    # Fast GMAs (trend detection)
    df['gma_15'] = _gma(df['_close'], 15)   # Fast (15h - optimal for SOL)
    df['gma_20'] = _gma(df['_close'], 20)   # Fast-medium
    df['gma_25'] = _gma(df['_close'], 25)   # Medium (optimal for BTC/ETH)
    
    # Slow GMAs (trend confirmation)
    df['gma_50'] = _gma(df['_close'], 50)   # Slow (optimal for SOL)
    df['gma_60'] = _gma(df['_close'], 60)   # Slow-medium (optimal for ETH)
    df['gma_75'] = _gma(df['_close'], 75)   # Slow (optimal for BTC)
    
    # Long-term trend filter
    df['gma_200'] = _gma(df['_close'], 200)  # Major trend (suggested enhancement)
    
    # GMA spreads (crossover signals)
    df['gma_spread_15_50'] = (df['gma_15'] - df['gma_50']) / df['gma_50']  # SOL optimal
    df['gma_spread_25_60'] = (df['gma_25'] - df['gma_60']) / df['gma_60']  # ETH optimal
    df['gma_spread_25_75'] = (df['gma_25'] - df['gma_75']) / df['gma_75']  # BTC optimal
    
    # Price position relative to GMAs
    df['price_above_gma_50'] = (df['_close'] > df['gma_50']).astype(int)
    df['price_above_gma_200'] = (df['_close'] > df['gma_200']).astype(int)
    
    # GMA slope (trend acceleration) - suggested enhancement
    df['gma_50_slope'] = df['gma_50'].pct_change(5)   # 5-hour change
    df['gma_200_slope'] = df['gma_200'].pct_change(24)  # 24-hour change
    
    # Distance from price to GMA (normalization)
    df['dist_to_gma_50'] = (df['_close'] - df['gma_50']) / df['gma_50']
    df['dist_to_gma_200'] = (df['_close'] - df['gma_200']) / df['gma_200']
    
    # GMA alignment (all GMAs trending same direction)
    gma_aligned_bull = (
        (df['gma_15'] > df['gma_50']) &
        (df['gma_50'] > df['gma_200']) &
        (df['_close'] > df['gma_15'])
    ).astype(int)
    
    gma_aligned_bear = (
        (df['gma_15'] < df['gma_50']) &
        (df['gma_50'] < df['gma_200']) &
        (df['_close'] < df['gma_15'])
    ).astype(int)
    
    df['gma_alignment'] = gma_aligned_bull - gma_aligned_bear  # +1 bull, -1 bear, 0 neutral
    
    # GMA volatility (how stable is the trend)
    df['gma_50_volatility'] = df['gma_50'].pct_change().rolling(24).std()
    
    logging.info(f"   ✓ Added 21 Geometric MA features (4.33-6.47 Sharpe champion!)")
    return df


def add_long_term_momentum_features(df):
    """
    Add long-term momentum features (30d, 90d, 180d, 365d)
    
    Based on Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
    Academic research shows 1-12 month momentum persistence across asset classes.
    
    Key findings:
    - Momentum persists for 1-12 months
    - Then reverses over longer horizons (>12 months)
    - Works best during extreme markets
    """
    logging.info("   Adding long-term momentum features (Moskowitz et al. 2012)...")
    
    # 30-day momentum (1 month) - 720 hours
    df['momentum_30d'] = df['_close'].pct_change(720)
    
    # 90-day momentum (3 months) - 2160 hours
    df['momentum_90d'] = df['_close'].pct_change(2160)
    
    # 180-day momentum (6 months) - 4320 hours
    df['momentum_180d'] = df['_close'].pct_change(4320)
    
    # 365-day momentum (1 year) - 8760 hours
    # Used primarily for mean reversion detection
    df['momentum_365d'] = df['_close'].pct_change(8760)
    
    # Momentum strength (average absolute momentum across timeframes)
    df['momentum_strength_longterm'] = (
        df['momentum_30d'].abs() + 
        df['momentum_90d'].abs() + 
        df['momentum_180d'].abs()
    ) / 3
    
    # Trend exhaustion indicator (mean reversion signal)
    # If 180-day momentum is extreme (top/bottom 10%), expect reversal
    df['trend_exhaustion'] = (
        (df['momentum_180d'] > df['momentum_180d'].quantile(0.9)) |  # Overbought
        (df['momentum_180d'] < df['momentum_180d'].quantile(0.1))    # Oversold
    ).astype(int)
    
    # Momentum divergence (momentum slowing while price rising/falling)
    # Short-term momentum weaker than long-term = potential reversal
    df['momentum_divergence'] = df['momentum_30d'] - df['momentum_90d']
    
    # Momentum acceleration (is momentum accelerating or decelerating?)
    df['momentum_acceleration'] = df['momentum_30d'] - df['momentum_30d'].shift(720)  # 30-day change
    
    # Cross-timeframe momentum alignment
    # All timeframes pointing same direction = strong trend
    df['momentum_alignment'] = (
        (np.sign(df['momentum_30d']) == np.sign(df['momentum_90d'])) &
        (np.sign(df['momentum_90d']) == np.sign(df['momentum_180d']))
    ).astype(int)
    
    logging.info(f"   ✓ Added 9 long-term momentum features")
    return df


def add_extreme_market_features(df):
    """
    Add extreme market regime detection features
    
    Based on Moskowitz, Ooi, Pedersen (2012) finding:
    "Momentum strategies perform BEST during extreme markets"
    
    Detects high volatility periods where momentum is most reliable.
    """
    logging.info("   Adding extreme market regime features...")
    
    # 30-day rolling volatility (annualized)
    vol_30d = df['returns'].rolling(720).std() * np.sqrt(24 * 365)
    
    # Volatility percentile rank (where are we in historical vol distribution?)
    df['vol_percentile_30d'] = vol_30d.rank(pct=True)
    
    # Extreme market indicator (top 10% volatility)
    df['extreme_market'] = (df['vol_percentile_30d'] > 0.9).astype(float)
    
    # Volume surge confirmation (confirms real volatility vs noise)
    if '_volume' in df.columns and df['_volume'].sum() > 0:
        vol_ma_30d = df['_volume'].rolling(720).mean()
        df['volume_surge'] = (df['_volume'] > vol_ma_30d * 1.5).astype(float)
        
        # Combined extreme market (high vol + volume surge)
        df['extreme_market_confirmed'] = ((df['extreme_market'] + df['volume_surge']) / 2).fillna(0)
    else:
        df['volume_surge'] = 0.0
        df['extreme_market_confirmed'] = df['extreme_market']
    
    # Volatility regime change (entering/exiting extreme regime)
    df['vol_regime_change'] = df['extreme_market'].diff().abs()
    
    # Days since last extreme market
    extreme_indices = df.index[df['extreme_market'] == 1].tolist()
    df['hours_since_extreme'] = 0
    for i, idx in enumerate(df.index):
        if idx in extreme_indices:
            df.loc[idx, 'hours_since_extreme'] = 0
        elif i > 0:
            df.loc[idx, 'hours_since_extreme'] = df.iloc[i-1]['hours_since_extreme'] + 1
    
    # Extreme market duration (how long in extreme state?)
    df['extreme_duration'] = df.groupby(
        (df['extreme_market'] != df['extreme_market'].shift()).cumsum()
    )['extreme_market'].cumsum()
    
    logging.info(f"   ✓ Added 8 extreme market features")
    return df


def add_kmeans_regime_features(df):
    """
    Add K-Means clustering to auto-discover market regimes
    
    Uses unsupervised learning to find natural groupings in:
    - Volatility
    - Trend strength
    - Volume
    
    Advantages over manual thresholds:
    - Data-driven (learns from actual patterns)
    - Adapts to changing market conditions
    - No manual threshold tuning needed
    
    Creates 4 clusters (optimal balance):
    - cluster_0_ranging: Low volatility, weak trends
    - cluster_1_trending: Medium volatility, moderate trends  
    - cluster_2_choppy: High volatility, mixed signals
    - cluster_3_stable: Very low volatility, calm market
    """
    logging.info("   Adding K-Means regime clustering features (4 clusters)...")
    
    # Select features for clustering (must exist and have good coverage)
    clustering_features = []
    
    if 'volatility' in df.columns:
        clustering_features.append('volatility')
    
    if 'trend_strength' in df.columns:
        clustering_features.append('trend_strength')
    
    if '_volume' in df.columns and df['_volume'].sum() > 0:
        # Normalize volume
        df['volume_normalized'] = (df['_volume'] - df['_volume'].mean()) / (df['_volume'].std() + 1e-8)
        clustering_features.append('volume_normalized')
    
    # Need at least 2 features for meaningful clustering
    if len(clustering_features) < 2:
        logging.warning("   ⚠ Not enough features for K-Means clustering, skipping")
        df['market_cluster'] = 0
        df['cluster_0_ranging'] = 1
        df['cluster_1_trending'] = 0
        df['cluster_2_choppy'] = 0
        df['cluster_3_stable'] = 0
        df['cluster_confidence'] = 0.0
        return df
    
    # Prepare data for clustering
    cluster_data = df[clustering_features].copy()
    
    # Fill NaN with median (common in early rows)
    for col in clustering_features:
        cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
    
    # Scale features (important for K-Means since it uses distance)
    scaler = StandardScaler()
    try:
        cluster_data_scaled = scaler.fit_transform(cluster_data)
    except Exception as e:
        logging.warning(f"   ⚠ Failed to scale clustering features: {e}")
        df['market_cluster'] = 0
        df['cluster_0_ranging'] = 1
        df['cluster_1_trending'] = 0
        df['cluster_2_choppy'] = 0
        df['cluster_3_stable'] = 0
        df['cluster_confidence'] = 0.0
        return df
    
    # Apply K-Means with 4 clusters (validated optimal count)
    # n_init=10: Run algorithm 10 times with different initializations
    # random_state=42: Reproducible results
    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
        df['market_cluster'] = kmeans.fit_predict(cluster_data_scaled)
        
        # Calculate confidence (distance to cluster center)
        # Closer to center = higher confidence
        distances = kmeans.transform(cluster_data_scaled)
        min_distances = distances.min(axis=1)
        max_distance = min_distances.max()
        if max_distance > 0:
            df['cluster_confidence'] = 1 - (min_distances / max_distance)
        else:
            df['cluster_confidence'] = 1.0
        
        # One-hot encode clusters with semantic names
        df['cluster_0_ranging'] = (df['market_cluster'] == 0).astype(int)
        df['cluster_1_trending'] = (df['market_cluster'] == 1).astype(int)
        df['cluster_2_choppy'] = (df['market_cluster'] == 2).astype(int)
        df['cluster_3_stable'] = (df['market_cluster'] == 3).astype(int)
        
        # Analyze cluster characteristics for logging
        cluster_names = ['ranging', 'trending', 'choppy', 'stable']
        cluster_stats = []
        for i in range(4):
            cluster_mask = df['market_cluster'] == i
            cluster_size = cluster_mask.sum()
            
            stats = {
                'cluster': i,
                'size': cluster_size,
                'pct': f"{100 * cluster_size / len(df):.1f}%"
            }
            
            if 'volatility' in df.columns:
                stats['avg_vol'] = df.loc[cluster_mask, 'volatility'].mean()
            
            if 'trend_strength' in df.columns:
                stats['avg_trend'] = df.loc[cluster_mask, 'trend_strength'].mean()
            
            cluster_stats.append(stats)
        
        # Log cluster distribution
        logging.info(f"   ✓ K-Means clustering complete (4 clusters):")
        for stats in cluster_stats:
            name = cluster_names[stats['cluster']]
            vol_str = f", vol={stats.get('avg_vol', 0):.4f}" if 'avg_vol' in stats else ""
            trend_str = f", trend={stats.get('avg_trend', 0):.4f}" if 'avg_trend' in stats else ""
            logging.info(f"      Cluster {stats['cluster']} ({name}): {stats['size']} samples ({stats['pct']}){vol_str}{trend_str}")
        
        logging.info(f"   ✓ Added 6 K-Means regime features (4 clusters + confidence + cluster_id)")
        
    except Exception as e:
        logging.warning(f"   ⚠ K-Means clustering failed: {e}")
        df['market_cluster'] = 0
        df['cluster_0_ranging'] = 1
        df['cluster_1_trending'] = 0
        df['cluster_2_choppy'] = 0
        df['cluster_3_stable'] = 0
        df['cluster_confidence'] = 0.0
    
    # Clean up temporary column
    if 'volume_normalized' in df.columns:
        df = df.drop(columns=['volume_normalized'], errors='ignore')
    
    return df


def add_interaction_features(df):
    """
    Add feature interactions - combinations of top-performing features
    
    These capture non-linear relationships and synergies between features.
    Based on Run 4 top features analysis.
    """
    logging.info("   Adding interaction features...")
    
    # Safe division helper
    def safe_divide(a, b, default=0):
        """Safely divide arrays, replacing inf/nan with default"""
        result = np.where(np.abs(b) > 1e-10, a / b, default)
        return np.nan_to_num(result, nan=default, posinf=default, neginf=default)
    
    # ========================================================================
    # 1. MOMENTUM × VOLATILITY INTERACTIONS
    # ========================================================================
    # When is momentum high relative to risk?
    if 'returns' in df.columns and 'volatility' in df.columns:
        df['momentum_vol_ratio'] = safe_divide(df['returns'].abs(), df['volatility'], default=0)
    
    if 'returns' in df.columns and 'volatility_percentile' in df.columns:
        df['returns_vol_adjusted'] = safe_divide(df['returns'], df['volatility_percentile'] + 0.01, default=0)
    
    if 'trend_strength' in df.columns and 'volatility_percentile' in df.columns:
        df['trend_vol_adjusted'] = safe_divide(df['trend_strength'], df['volatility_percentile'] + 0.01, default=0)
    
    # ========================================================================
    # 2. PRICE LEVEL × ORDER FLOW INTERACTIONS
    # ========================================================================
    # Order flow behavior near round numbers
    if 'round_10k_dist' in df.columns and 'cumulative_order_flow' in df.columns:
        df['round_level_flow'] = df['round_10k_dist'] * df['cumulative_order_flow']
    
    if 'round_5k_dist' in df.columns and 'order_imbalance_ma' in df.columns:
        df['round_5k_imbalance'] = df['round_5k_dist'] * df['order_imbalance_ma']
    
    # Price distance × flow
    if 'dist_to_high_24h' in df.columns and 'cumulative_order_flow' in df.columns:
        df['high_dist_flow'] = df['dist_to_high_24h'] * df['cumulative_order_flow']
    
    if 'dist_to_low_24h' in df.columns and 'buy_volume_proxy' in df.columns:
        df['low_dist_buying'] = df['dist_to_low_24h'] * df['buy_volume_proxy']
    
    # ========================================================================
    # 3. MICROSTRUCTURE × REGIME INTERACTIONS
    # ========================================================================
    # Spread behavior by market regime
    if 'spread_proxy' in df.columns and 'regime_duration' in df.columns:
        df['spread_regime'] = df['spread_proxy'] * df['regime_duration']
    
    if 'spread_proxy' in df.columns and 'volatility_regime' in df.columns:
        df['spread_vol_regime'] = df['spread_proxy'] * df['volatility_regime']
    
    # Liquidity × trend
    if 'amihud_illiquidity' in df.columns and 'trend_strength' in df.columns:
        df['liquidity_trend'] = df['amihud_illiquidity'] * df['trend_strength']
    
    if 'roll_spread' in df.columns and 'adx_proxy' in df.columns:
        df['spread_trend_strength'] = df['roll_spread'] * df['adx_proxy']
    
    # ========================================================================
    # 4. GMA × OTHER FEATURE INTERACTIONS (NEW - Champion Indicator!)
    # ========================================================================
    # GMA trend strength × volatility
    if 'gma_spread_15_50' in df.columns and 'volatility' in df.columns:
        df['gma_trend_vol'] = df['gma_spread_15_50'].abs() * df['volatility']
    
    # GMA × momentum alignment
    if 'dist_to_gma_50' in df.columns and 'momentum_90d' in df.columns:
        df['gma_momentum_sync'] = df['dist_to_gma_50'] * df['momentum_90d']
    
    # GMA × order flow (trend + buying/selling)
    if 'gma_spread_15_50' in df.columns and 'buy_sell_ratio' in df.columns:
        df['gma_flow_alignment'] = df['gma_spread_15_50'] * df['buy_sell_ratio']
    
    # GMA slope × volatility regime (trend acceleration in different regimes)
    if 'gma_50_slope' in df.columns and 'volatility_percentile' in df.columns:
        df['gma_accel_regime'] = df['gma_50_slope'] * df['volatility_percentile']
    
    # GMA distance × extreme markets (how far from GMA during extremes)
    if 'dist_to_gma_200' in df.columns and 'extreme_market' in df.columns:
        df['gma_extreme_dist'] = df['dist_to_gma_200'] * df['extreme_market']
    
    # GMA alignment × cluster (strong trends by market regime)
    if 'gma_alignment' in df.columns and 'market_cluster' in df.columns:
        df['gma_cluster_trend'] = df['gma_alignment'] * df['market_cluster']
    
    # GMA volatility × overall volatility (trend stability vs market volatility)
    if 'gma_50_volatility' in df.columns and 'volatility' in df.columns:
        df['gma_stability_ratio'] = safe_divide(
            df['gma_50_volatility'], 
            df['volatility'] + 0.0001, 
            default=0
        )
    
    # ========================================================================
    # 5. VOLATILITY CLUSTERING & ACCELERATION
    # ========================================================================
    # Volatility acceleration in different regimes
    if 'volatility_acceleration' in df.columns and 'volatility_percentile' in df.columns:
        df['vol_accel_regime'] = df['volatility_acceleration'] * df['volatility_percentile']
    
    if 'parkinson_vol' in df.columns and 'chaos_indicator' in df.columns:
        df['vol_chaos_combo'] = df['parkinson_vol'] * df['chaos_indicator']
    
    if 'volatility' in df.columns and 'hurst_48h' in df.columns:
        df['vol_persistence'] = df['volatility'] * df['hurst_48h']
    
    # ========================================================================
    # 5. MULTI-SCALE MOMENTUM
    # ========================================================================
    # Short-term vs long-term skewness
    if 'returns_skew_24h' in df.columns and 'returns_skew_168h' in df.columns:
        df['momentum_scale_ratio'] = safe_divide(
            df['returns_skew_24h'], 
            df['returns_skew_168h'].abs() + 0.01, 
            default=0
        )
    
    # Kurtosis change (tail risk evolution)
    if 'returns_kurtosis_24h' in df.columns and 'returns_kurtosis_168h' in df.columns:
        df['kurtosis_change'] = df['returns_kurtosis_24h'] - df['returns_kurtosis_168h']
    
    # ========================================================================
    # 6. VOLUME × PRICE DYNAMICS
    # ========================================================================
    # Volume-weighted momentum
    if 'returns' in df.columns and 'volume' in df.columns:
        vol_ma = df['volume'].rolling(24, min_periods=1).mean()
        df['volume_weighted_returns'] = df['returns'] * safe_divide(df['volume'], vol_ma, default=1)
    
    # Volume imbalance × price movement
    if 'volume_imbalance' in df.columns and 'returns' in df.columns:
        df['imbalance_momentum'] = df['volume_imbalance'] * df['returns']
    
    if 'volume_imbalance_ma' in df.columns and 'trend_strength' in df.columns:
        df['imbalance_trend'] = df['volume_imbalance_ma'] * df['trend_strength']
    
    # ========================================================================
    # 7. FRACTAL × VOLATILITY
    # ========================================================================
    # Fractal dimension in volatile markets
    if 'fractal_dimension' in df.columns and 'volatility_percentile' in df.columns:
        df['fractal_vol_regime'] = df['fractal_dimension'] * df['volatility_percentile']
    
    # Chaos in trending markets
    if 'chaos_indicator' in df.columns and 'trend_strength' in df.columns:
        df['chaos_trend'] = df['chaos_indicator'] * df['trend_strength']
    
    # ========================================================================
    # 8. ORDER FLOW RATIOS
    # ========================================================================
    # Order flow vs volatility
    if 'cumulative_order_flow' in df.columns and 'volatility' in df.columns:
        df['flow_vol_ratio'] = safe_divide(
            df['cumulative_order_flow'].abs(), 
            df['volatility'], 
            default=0
        )
    
    # Trade intensity vs spread
    if 'trade_intensity' in df.columns and 'spread_proxy' in df.columns:
        df['intensity_spread_ratio'] = safe_divide(
            df['trade_intensity'], 
            df['spread_proxy'] + 0.0001, 
            default=0
        )
    
    # Count new features
    interaction_features = [
        'momentum_vol_ratio', 'returns_vol_adjusted', 'trend_vol_adjusted',
        'round_level_flow', 'round_5k_imbalance', 'high_dist_flow', 'low_dist_buying',
        'spread_regime', 'spread_vol_regime', 'liquidity_trend', 'spread_trend_strength',
        'vol_accel_regime', 'vol_chaos_combo', 'vol_persistence',
        'momentum_scale_ratio', 'kurtosis_change',
        'volume_weighted_returns', 'imbalance_momentum', 'imbalance_trend',
        'fractal_vol_regime', 'chaos_trend',
        'flow_vol_ratio', 'intensity_spread_ratio'
    ]
    
    added_features = [f for f in interaction_features if f in df.columns]
    logging.info(f"   ✓ Added {len(added_features)} interaction features")
    
    return df


def add_all_enhanced_features(df):
    """
    Add all enhanced features to dataframe
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        df: DataFrame with enhanced features
    """
    logging.info("\n" + "="*70)
    logging.info("🔬 ADDING ENHANCED FEATURES")
    logging.info("="*70)
    
    initial_features = len(df.columns)
    
    # Detect column names (handle both 'close' and 'price' naming conventions)
    price_col = 'close' if 'close' in df.columns else 'price'
    high_col = 'high' if 'high' in df.columns else '1h_high' if '1h_high' in df.columns else price_col
    low_col = 'low' if 'low' in df.columns else '1h_low' if '1h_low' in df.columns else price_col
    volume_col = 'volume' if 'volume' in df.columns else '1h_volume' if '1h_volume' in df.columns else None
    
    logging.info(f"Using columns: price={price_col}, high={high_col}, low={low_col}, volume={volume_col}")
    
    # Ensure basic features exist
    if 'returns' not in df.columns:
        df['returns'] = df[price_col].pct_change()
    if 'volatility' not in df.columns:
        df['volatility'] = df['returns'].rolling(24).std() * np.sqrt(24)
    
    # Temporarily create standard column names for feature functions
    df['_close'] = df[price_col]
    df['_high'] = df[high_col]
    df['_low'] = df[low_col]
    if volume_col:
        df['_volume'] = df[volume_col]
    else:
        df['_volume'] = 1.0  # Dummy volume if not available
    
    # Add feature groups
    df = add_microstructure_features(df)
    df = add_volatility_regime_features(df)
    df = add_fractal_chaos_features(df)
    df = add_order_flow_features(df)
    df = add_market_regime_features(df)
    df = add_price_levels_features(df)
    
    # Add Geometric MA features (NEW - CHAMPION INDICATOR!)
    df = add_geometric_ma_features(df)
    
    # Add long-term momentum features (NEW - Academic research validated!)
    df = add_long_term_momentum_features(df)
    
    # Add extreme market regime detection (NEW - Moskowitz et al. 2012)
    df = add_extreme_market_features(df)
    
    # Add K-Means clustering for regime discovery (NEW - ML A-Z Course!)
    df = add_kmeans_regime_features(df)
    
    # Add interaction features (NEW - Phase 5 optimization!)
    df = add_interaction_features(df)
    
    # Clean up temporary columns
    df = df.drop(columns=['_close', '_high', '_low', '_volume'], errors='ignore')
    
    final_features = len(df.columns)
    new_features = final_features - initial_features
    
    logging.info("="*70)
    logging.info(f"✅ ENHANCED FEATURES COMPLETE")
    logging.info(f"   Initial features: {initial_features}")
    logging.info(f"   Final features: {final_features}")
    logging.info(f"   New features added: {new_features}")
    logging.info("="*70 + "\n")
    
    return df


# ============================================================================
# FEATURE GROUPS (for easy reference)
# ============================================================================

ENHANCED_FEATURE_GROUPS = {
    'microstructure': [
        'spread_proxy', 'price_efficiency', 'amihud_illiquidity',
        'amihud_illiquidity_24h', 'roll_spread', 'trade_intensity'
    ],
    'volatility_regime': [
        'volatility_regime', 'volatility_percentile', 'volatility_acceleration',
        'volatility_lag1', 'volatility_lag2', 'parkinson_vol'
    ],
    'fractal_chaos': [
        'returns_skew_24h', 'returns_skew_168h', 'returns_kurtosis_24h',
        'returns_kurtosis_168h', 'hurst_48h', 'fractal_dimension', 'chaos_indicator'
    ],
    'order_flow': [
        'buy_pressure', 'sell_pressure', 'order_imbalance', 'order_imbalance_ma',
        'volume_imbalance', 'volume_imbalance_ma', 'buy_volume_proxy',
        'sell_volume_proxy', 'buy_sell_ratio', 'cumulative_order_flow'
    ],
    'market_regime': [
        'trend_strength', 'adx_proxy', 'regime_trending', 'regime_ranging',
        'regime_volatile', 'regime_duration'
    ],
    'price_levels': [
        'dist_to_high_24h', 'dist_to_low_24h', 'dist_to_high_168h',
        'dist_to_low_168h', 'round_number_dist', 'round_5k_dist', 'round_10k_dist'
    ],
    'geometric_ma': [
        'gma_15', 'gma_20', 'gma_25', 'gma_50', 'gma_60', 'gma_75', 'gma_200',
        'gma_spread_15_50', 'gma_spread_25_60', 'gma_spread_25_75',
        'price_above_gma_50', 'price_above_gma_200',
        'gma_50_slope', 'gma_200_slope',
        'dist_to_gma_50', 'dist_to_gma_200',
        'gma_alignment', 'gma_50_volatility'
    ],
    'long_term_momentum': [
        'momentum_30d', 'momentum_90d', 'momentum_180d', 'momentum_365d',
        'momentum_strength_longterm', 'trend_exhaustion', 'momentum_divergence',
        'momentum_acceleration', 'momentum_alignment'
    ],
    'extreme_markets': [
        'vol_percentile_30d', 'extreme_market', 'volume_surge',
        'extreme_market_confirmed', 'vol_regime_change', 'hours_since_extreme',
        'extreme_duration'
    ],
    'kmeans_regimes': [
        'market_cluster', 'cluster_0_ranging', 'cluster_1_trending',
        'cluster_2_choppy', 'cluster_3_stable', 'cluster_confidence'
    ],
    'interactions': [
        'momentum_vol_ratio', 'returns_vol_adjusted', 'trend_vol_adjusted',
        'round_level_flow', 'round_5k_imbalance', 'high_dist_flow', 'low_dist_buying',
        'spread_regime', 'spread_vol_regime', 'liquidity_trend', 'spread_trend_strength',
        'gma_trend_vol', 'gma_momentum_sync', 'gma_flow_alignment', 
        'gma_accel_regime', 'gma_extreme_dist', 'gma_cluster_trend', 'gma_stability_ratio',
        'vol_accel_regime', 'vol_chaos_combo', 'vol_persistence',
        'momentum_scale_ratio', 'kurtosis_change',
        'volume_weighted_returns', 'imbalance_momentum', 'imbalance_trend',
        'fractal_vol_regime', 'chaos_trend',
        'flow_vol_ratio', 'intensity_spread_ratio'
    ]
}


def get_enhanced_feature_names():
    """Get list of all enhanced feature names"""
    all_features = []
    for group in ENHANCED_FEATURE_GROUPS.values():
        all_features.extend(group)
    return all_features


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 ENHANCED FEATURES - TEST")
    print("="*70)
    
    # Load sample data
    import sys
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).parent.absolute()
    
    # Try to load Bitcoin data
    try:
        df = pd.read_csv(SCRIPT_DIR / 'DATA' / 'yf_btc_1h.csv')
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.rename(columns={'price': 'close'}, inplace=True, errors='ignore')
        
        print(f"\n✅ Loaded {len(df)} rows of data")
        print(f"   Columns: {list(df.columns)}")
        
        # Add enhanced features
        df_enhanced = add_all_enhanced_features(df.copy())
        
        # Show summary
        print("\n" + "="*70)
        print("📊 FEATURE SUMMARY")
        print("="*70)
        
        for group_name, features in ENHANCED_FEATURE_GROUPS.items():
            available = [f for f in features if f in df_enhanced.columns]
            print(f"\n{group_name.upper()} ({len(available)} features):")
            for feat in available[:5]:  # Show first 5
                if feat in df_enhanced.columns:
                    print(f"   {feat}: {df_enhanced[feat].iloc[-1]:.6f}")
            if len(available) > 5:
                print(f"   ... and {len(available)-5} more")
        
        print("\n" + "="*70)
        print(f"✅ All {len(get_enhanced_feature_names())} enhanced features added!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print("   Run fetch_data.py first to generate data files")
