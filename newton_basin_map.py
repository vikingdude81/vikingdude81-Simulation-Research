"""
Newton Basin-of-Attraction Map - Python Implementation

Based on Newton-Raphson iteration to find which moving average "root" 
price is converging toward. Identifies regime shifts and attractor changes.

Original: Pine Script v6
Purpose: Detect which MA basin price is in (not signal generation directly)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class NewtonBasinMap:
    """
    Newton-Raphson basin-of-attraction analyzer
    
    Uses Newton-Raphson iteration to solve:
    f(x) = (x - ma1)(x - ma2)(x - ma3) = 0
    
    The solution x converges to one of three "roots" (moving averages),
    revealing which attractor basin price is currently in.
    """
    
    def __init__(
        self,
        ma1_len: int = 20,      # EMA length (fast root)
        ma2_len: int = 50,      # SMA length (medium root)
        ma3_len: int = 200,     # SMA length (slow root)
        newton_iterations: int = 6,   # Max iterations
        tolerance: float = 0.01,      # Convergence tolerance
        use_relative_distance: bool = True  # Normalize by MA value
    ):
        self.ma1_len = ma1_len
        self.ma2_len = ma2_len
        self.ma3_len = ma3_len
        self.newton_iterations = newton_iterations
        self.tolerance = tolerance
        self.use_relative_distance = use_relative_distance
        self.eps = 1e-10  # Prevent division by zero
    
    def _newton_raphson_step(self, x: float, ma1: float, ma2: float, ma3: float) -> Tuple[float, float]:
        """
        Single Newton-Raphson iteration
        
        f(x) = (x - ma1)(x - ma2)(x - ma3)
        f'(x) = (x - ma2)(x - ma3) + (x - ma1)(x - ma3) + (x - ma1)(x - ma2)
        
        Returns: (new_x, f_value)
        """
        # Function value
        f = (x - ma1) * (x - ma2) * (x - ma3)
        
        # Derivative
        fp = (x - ma2)*(x - ma3) + (x - ma1)*(x - ma3) + (x - ma1)*(x - ma2)
        
        # Prevent division by zero
        if abs(fp) < self.eps:
            fp = self.eps
        
        # Newton step: x_new = x - f/f'
        x_new = x - (f / fp)
        
        return x_new, abs(f)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Newton basin map for entire dataframe"""
        df = df.copy()
        
        # Calculate moving averages (the "roots")
        df['ma1_ema'] = df['close'].ewm(span=self.ma1_len, adjust=False).mean()
        df['ma2_sma'] = df['close'].rolling(window=self.ma2_len).mean()
        df['ma3_sma'] = df['close'].rolling(window=self.ma3_len).mean()
        
        # Initialize Newton result arrays
        df['newton_x'] = np.nan
        df['newton_iterations_used'] = 0
        df['newton_converged'] = False
        df['basin_id'] = 0  # 1=fast, 2=medium, 3=slow
        df['basin_distance'] = np.nan
        
        # Apply Newton-Raphson to each bar
        for idx in df.index:
            if pd.isna(df.loc[idx, 'ma1_ema']) or pd.isna(df.loc[idx, 'ma2_sma']) or pd.isna(df.loc[idx, 'ma3_sma']):
                continue
            
            # Initial guess: current price
            x = df.loc[idx, 'close']
            ma1 = df.loc[idx, 'ma1_ema']
            ma2 = df.loc[idx, 'ma2_sma']
            ma3 = df.loc[idx, 'ma3_sma']
            
            # Newton-Raphson iterations
            converged = False
            iterations_used = 0
            
            for i in range(self.newton_iterations):
                x_new, f_val = self._newton_raphson_step(x, ma1, ma2, ma3)
                iterations_used = i + 1
                
                # Check convergence
                if f_val < self.tolerance:
                    converged = True
                    break
                
                x = x_new
            
            # Store final Newton value
            df.loc[idx, 'newton_x'] = x
            df.loc[idx, 'newton_iterations_used'] = iterations_used
            df.loc[idx, 'newton_converged'] = converged
            
            # Calculate distances to each root
            if self.use_relative_distance:
                # Relative distance (normalized by MA value)
                dist1 = abs(x - ma1) / (ma1 + self.eps)
                dist2 = abs(x - ma2) / (ma2 + self.eps)
                dist3 = abs(x - ma3) / (ma3 + self.eps)
            else:
                # Absolute distance
                dist1 = abs(x - ma1)
                dist2 = abs(x - ma2)
                dist3 = abs(x - ma3)
            
            # Classify which basin (closest root)
            if dist1 < dist2 and dist1 < dist3:
                basin_id = 1  # Fast MA basin
                basin_distance = dist1
            elif dist2 < dist3:
                basin_id = 2  # Medium MA basin
                basin_distance = dist2
            else:
                basin_id = 3  # Slow MA basin
                basin_distance = dist3
            
            df.loc[idx, 'basin_id'] = basin_id
            df.loc[idx, 'basin_distance'] = basin_distance
        
        # Detect basin shifts (regime changes)
        df['basin_shift'] = df['basin_id'].diff().fillna(0).astype(int)
        df['shift_to_fast'] = (df['basin_shift'] < 0) & (df['basin_id'] == 1)
        df['shift_to_slow'] = (df['basin_shift'] > 0) & (df['basin_id'] == 3)
        
        # Additional analytics
        df['basin_strength'] = 1.0 / (df['basin_distance'] + self.eps)  # Inverse of distance
        df['in_fast_basin'] = (df['basin_id'] == 1).astype(int)
        df['in_medium_basin'] = (df['basin_id'] == 2).astype(int)
        df['in_slow_basin'] = (df['basin_id'] == 3).astype(int)
        
        # Persistence: How long in current basin
        df['basin_persistence'] = 0
        current_basin = 0
        count = 0
        for idx in df.index:
            if df.loc[idx, 'basin_id'] == current_basin:
                count += 1
            else:
                current_basin = df.loc[idx, 'basin_id']
                count = 1
            df.loc[idx, 'basin_persistence'] = count
        
        return df
    
    def get_current_state(self, df: pd.DataFrame) -> Dict:
        """Get current basin state"""
        if len(df) == 0:
            return {}
        
        last = df.iloc[-1]
        
        basin_names = {1: "FAST (EMA 20)", 2: "MEDIUM (SMA 50)", 3: "SLOW (SMA 200)"}
        
        return {
            'basin_id': int(last['basin_id']),
            'basin_name': basin_names.get(int(last['basin_id']), "UNKNOWN"),
            'basin_distance': last['basin_distance'],
            'basin_strength': last['basin_strength'],
            'basin_persistence': int(last['basin_persistence']),
            'newton_x': last['newton_x'],
            'converged': last['newton_converged'],
            'iterations_used': int(last['newton_iterations_used']),
            'ma1_ema': last['ma1_ema'],
            'ma2_sma': last['ma2_sma'],
            'ma3_sma': last['ma3_sma'],
            'price': last['close']
        }
    
    def analyze_regime_shifts(self, df: pd.DataFrame) -> Dict:
        """Analyze basin shift patterns"""
        
        # Count basin transitions
        shift_to_fast = df['shift_to_fast'].sum()
        shift_to_slow = df['shift_to_slow'].sum()
        total_shifts = (df['basin_shift'] != 0).sum()
        
        # Average persistence in each basin
        fast_persistence = df[df['in_fast_basin'] == 1]['basin_persistence'].mean()
        medium_persistence = df[df['in_medium_basin'] == 1]['basin_persistence'].mean()
        slow_persistence = df[df['in_slow_basin'] == 1]['basin_persistence'].mean()
        
        # Time spent in each basin
        total_bars = len(df)
        pct_fast = (df['in_fast_basin'].sum() / total_bars) * 100
        pct_medium = (df['in_medium_basin'].sum() / total_bars) * 100
        pct_slow = (df['in_slow_basin'].sum() / total_bars) * 100
        
        # Forward returns after shifts
        df['return_12h'] = df['close'].pct_change(12).shift(-12)
        
        shift_to_fast_returns = df[df['shift_to_fast']]['return_12h'].dropna()
        shift_to_slow_returns = df[df['shift_to_slow']]['return_12h'].dropna()
        
        return {
            'total_shifts': int(total_shifts),
            'shifts_to_fast': int(shift_to_fast),
            'shifts_to_slow': int(shift_to_slow),
            'fast_persistence_avg': fast_persistence,
            'medium_persistence_avg': medium_persistence,
            'slow_persistence_avg': slow_persistence,
            'pct_time_fast': pct_fast,
            'pct_time_medium': pct_medium,
            'pct_time_slow': pct_slow,
            'shift_to_fast_avg_return': shift_to_fast_returns.mean() if len(shift_to_fast_returns) > 0 else 0,
            'shift_to_slow_avg_return': shift_to_slow_returns.mean() if len(shift_to_slow_returns) > 0 else 0,
            'shift_to_fast_win_rate': (shift_to_fast_returns > 0).sum() / len(shift_to_fast_returns) if len(shift_to_fast_returns) > 0 else 0,
            'shift_to_slow_win_rate': (shift_to_slow_returns > 0).sum() / len(shift_to_slow_returns) if len(shift_to_slow_returns) > 0 else 0
        }


if __name__ == "__main__":
    print("="*80)
    print("NEWTON BASIN-OF-ATTRACTION MAP - Python Test")
    print("="*80)
    
    # Load BTC data
    print("\nLoading BTC data...")
    df = pd.read_csv('DATA/yf_btc_1h.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    if 'Close' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
    
    df = df.tail(2160).reset_index(drop=True)  # Last 90 days
    
    print(f"   Loaded {len(df)} bars")
    
    # Initialize Newton basin map
    print("\nCalculating Newton basin map...")
    newton = NewtonBasinMap(
        ma1_len=20,
        ma2_len=50,
        ma3_len=200,
        newton_iterations=6,
        tolerance=0.01,
        use_relative_distance=True
    )
    
    df_result = newton.calculate(df)
    
    # Current state
    print("\n" + "="*80)
    print("CURRENT STATE")
    print("="*80)
    
    state = newton.get_current_state(df_result)
    print(f"\nBasin: {state['basin_name']}")
    print(f"  Distance: {state['basin_distance']:.6f}")
    print(f"  Strength: {state['basin_strength']:.2f}")
    print(f"  Persistence: {state['basin_persistence']} bars")
    print(f"  Newton converged: {state['converged']} (in {state['iterations_used']} iterations)")
    print(f"\nPrice: ${state['price']:.2f}")
    print(f"  EMA 20:  ${state['ma1_ema']:.2f}")
    print(f"  SMA 50:  ${state['ma2_sma']:.2f}")
    print(f"  SMA 200: ${state['ma3_sma']:.2f}")
    print(f"  Newton X: ${state['newton_x']:.2f}")
    
    # Analyze regime shifts
    print("\n" + "="*80)
    print("REGIME SHIFT ANALYSIS (90 days)")
    print("="*80)
    
    analysis = newton.analyze_regime_shifts(df_result)
    
    print(f"\nBasin Occupancy:")
    print(f"  Fast Basin (EMA 20):   {analysis['pct_time_fast']:.1f}% of time")
    print(f"  Medium Basin (SMA 50): {analysis['pct_time_medium']:.1f}% of time")
    print(f"  Slow Basin (SMA 200):  {analysis['pct_time_slow']:.1f}% of time")
    
    print(f"\nBasin Persistence (average bars):")
    print(f"  Fast:   {analysis['fast_persistence_avg']:.1f} bars")
    print(f"  Medium: {analysis['medium_persistence_avg']:.1f} bars")
    print(f"  Slow:   {analysis['slow_persistence_avg']:.1f} bars")
    
    print(f"\nRegime Shifts:")
    print(f"  Total shifts: {analysis['total_shifts']}")
    print(f"  Shifts to FAST basin:  {analysis['shifts_to_fast']}")
    print(f"  Shifts to SLOW basin:  {analysis['shifts_to_slow']}")
    
    print(f"\nForward Returns After Shifts (12h):")
    print(f"  Shift to FAST:")
    print(f"    Avg Return:  {analysis['shift_to_fast_avg_return']*100:.3f}%")
    print(f"    Win Rate:    {analysis['shift_to_fast_win_rate']*100:.1f}%")
    print(f"  Shift to SLOW:")
    print(f"    Avg Return:  {analysis['shift_to_slow_avg_return']*100:.3f}%")
    print(f"    Win Rate:    {analysis['shift_to_slow_win_rate']*100:.1f}%")
    
    # Last 10 basin shifts
    print("\n" + "="*80)
    print("LAST 10 BASIN SHIFTS")
    print("="*80)
    
    shifts = df_result[df_result['basin_shift'] != 0].tail(10)
    basin_names = {1: "FAST", 2: "MED", 3: "SLOW"}
    
    for idx, row in shifts.iterrows():
        shift_type = "→" if row['basin_shift'] > 0 else "←"
        basin_name = basin_names.get(int(row['basin_id']), "?")
        print(f"{row['time'].strftime('%Y-%m-%d %H:%M')} {shift_type} {basin_name:4s} "
              f"@ ${row['close']:>9,.2f}  (distance: {row['basin_distance']:.6f})")
    
    print("\n" + "="*80)
    print("Test complete!")
