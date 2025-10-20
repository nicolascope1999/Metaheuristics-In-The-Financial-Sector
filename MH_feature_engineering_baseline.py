"""
Baseline Feature Engineering Module for Metaheuristics Financial Project

This module contains the reusable baseline feature engineering function extracted from the 
Combined_Metaheuristics_Workflow.ipynb notebook. It creates comprehensive technical indicators 
from raw OHLCV data and target variables for machine learning models.

Functions:
    compute_baseline_features(df_ohlcv, volume_type_ticker=True): Main feature engineering pipeline
    compute_target_rule(df_features, lookback=20, prediction_horizon=5, multiplier=1): Target variable computation
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def compute_baseline_features(df_ohlcv: pd.DataFrame, volume_type_ticker: bool = True) -> pd.DataFrame:
    """
    Reusable baseline feature engineering function extracted from the notebook.
    Input: df_ohlcv with columns ['Open','High','Low','Close','Volume'] and optional datetime index
    Output: DataFrame with the exact feature columns used during training (same names & order).
    
    Args:
        df_ohlcv (pd.DataFrame): DataFrame with OHLCV columns
        volume_type_ticker (bool): If True, treats volume as tick volume; if False, as trading volume
        
    Returns:
        pd.DataFrame: DataFrame with engineered features and target variable
    """
    # Create a deep copy to work with for feature engineering
    feature_df = df_ohlcv.copy()
    
    # Define common lookback periods
    lookback_periods = [5, 10, 20, 50, 100]

    ## 1. Basic Price Transformations

    # Calculate returns (percentage change)
    feature_df['Percentage_Change'] = feature_df['Close'].pct_change()
    feature_df['Log_Return'] = np.log(feature_df['Close'] / feature_df['Close'].shift(1))

    # Calculate price differences
    feature_df['Price_Diff'] = feature_df['Close'].diff()
    feature_df['Open_Close_Diff'] = feature_df['Close'] - feature_df['Open']
    feature_df['High_Low_Diff'] = feature_df['High'] - feature_df['Low']

    # Calculate price ratios
    feature_df['High_Close_Ratio'] = feature_df['High'] / feature_df['Close']
    feature_df['Low_Close_Ratio'] = feature_df['Low'] / feature_df['Close']
    feature_df['Open_Close_Ratio'] = feature_df['Open'] / feature_df['Close']

    # Calculate candle characteristics
    feature_df['Candle_Range'] = feature_df['High'] - feature_df['Low']  # Total range
    feature_df['Body_Size'] = abs(feature_df['Close'] - feature_df['Open'])  # Body size
    feature_df['Upper_Shadow'] = feature_df['High'] - feature_df[['Open', 'Close']].max(axis=1)  # Upper shadow
    feature_df['Lower_Shadow'] = feature_df[['Open', 'Close']].min(axis=1) - feature_df['Low']  # Lower shadow
    feature_df['Body_To_Range_Ratio'] = feature_df['Body_Size'] / feature_df['Candle_Range']  # Body to range ratio

    ## 2. Moving Averages and Derivatives

    # Simple Moving Averages (SMA)
    for period in lookback_periods:
        feature_df[f'SMA_{period}'] = feature_df['Close'].rolling(window=period).mean()
        feature_df[f'SMA_Dist_{period}'] = (feature_df['Close'] - feature_df[f'SMA_{period}']) / feature_df[f'SMA_{period}'] * 100

    # Exponential Moving Averages (EMA)
    for period in lookback_periods:
        feature_df[f'EMA_{period}'] = feature_df['Close'].ewm(span=period, adjust=False).mean()
        feature_df[f'EMA_Dist_{period}'] = (feature_df['Close'] - feature_df[f'EMA_{period}']) / feature_df[f'EMA_{period}'] * 100

    # Moving Average Convergence Divergence (MACD)
    feature_df['MACD_Line'] = feature_df['Close'].ewm(span=12, adjust=False).mean() - feature_df['Close'].ewm(span=26, adjust=False).mean()
    feature_df['MACD_Signal'] = feature_df['MACD_Line'].ewm(span=9, adjust=False).mean()
    feature_df['MACD_Histogram'] = feature_df['MACD_Line'] - feature_df['MACD_Signal']
    feature_df['MACD_CrossAbove'] = ((feature_df['MACD_Line'] > feature_df['MACD_Signal']) & 
                                    (feature_df['MACD_Line'].shift(1) <= feature_df['MACD_Signal'].shift(1))).astype(int)
    feature_df['MACD_CrossBelow'] = ((feature_df['MACD_Line'] < feature_df['MACD_Signal']) & 
                                    (feature_df['MACD_Line'].shift(1) >= feature_df['MACD_Signal'].shift(1))).astype(int)

    # Moving Average Crossovers
    feature_df['SMA_5_10_Cross'] = ((feature_df['SMA_5'] > feature_df['SMA_10']) & 
                                   (feature_df['SMA_5'].shift(1) <= feature_df['SMA_10'].shift(1))).astype(int)
    feature_df['SMA_10_20_Cross'] = ((feature_df['SMA_10'] > feature_df['SMA_20']) & 
                                    (feature_df['SMA_10'].shift(1) <= feature_df['SMA_20'].shift(1))).astype(int)

    # Triple Exponential Moving Average (TEMA)
    for period in [10, 20, 50]:
        ema1 = feature_df['Close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        feature_df[f'TEMA_{period}'] = 3 * ema1 - 3 * ema2 + ema3
        feature_df[f'TEMA_Dist_{period}'] = (feature_df['Close'] - feature_df[f'TEMA_{period}']) / feature_df[f'TEMA_{period}'] * 100

    # Bollinger Bands
    for period in [20]:
        feature_df[f'BB_Middle_{period}'] = feature_df['Close'].rolling(window=period).mean()
        feature_df[f'BB_Std_{period}'] = feature_df['Close'].rolling(window=period).std()
        feature_df[f'BB_Upper_{period}'] = feature_df[f'BB_Middle_{period}'] + 2 * feature_df[f'BB_Std_{period}']
        feature_df[f'BB_Lower_{period}'] = feature_df[f'BB_Middle_{period}'] - 2 * feature_df[f'BB_Std_{period}']
        feature_df[f'BB_Width_{period}'] = (feature_df[f'BB_Upper_{period}'] - feature_df[f'BB_Lower_{period}']) / feature_df[f'BB_Middle_{period}']
        feature_df[f'BB_Pct_B_{period}'] = (feature_df['Close'] - feature_df[f'BB_Lower_{period}']) / (feature_df[f'BB_Upper_{period}'] - feature_df[f'BB_Lower_{period}'])

    ## 3. Volatility Indicators

    # Average True Range (ATR)
    for period in [14, 20]:
        high_low = feature_df['High'] - feature_df['Low']
        high_close = abs(feature_df['High'] - feature_df['Close'].shift(1))
        low_close = abs(feature_df['Low'] - feature_df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        feature_df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        feature_df[f'ATR_Ratio_{period}'] = feature_df[f'ATR_{period}'] / feature_df['Close'] * 100

    # Volatility using standard deviation
    for period in [5, 10, 20, 50]:
        feature_df[f'Volatility_{period}'] = feature_df['Close'].pct_change().rolling(window=period).std() * np.sqrt(period)
        feature_df[f'Normalized_Vol_{period}'] = feature_df[f'Volatility_{period}'] / feature_df[f'Volatility_{period}'].rolling(window=100).mean()

    # Garman-Klass volatility estimator
    feature_df['GK_Volatility'] = np.sqrt(
        0.5 * (np.log(feature_df['High'] / feature_df['Low'])) ** 2 -
        (2 * np.log(2) - 1) * (np.log(feature_df['Close'] / feature_df['Open'])) ** 2
    )

    # Parkinson's volatility
    feature_df['Parkinson_Vol'] = np.sqrt((1 / (4 * np.log(2))) * 
                                         ((np.log(feature_df['High'] / feature_df['Low'])) ** 2))

    # Chaikin Volatility
    for period in [10, 20]:
        feature_df[f'Chaikin_Vol_{period}'] = (
            (feature_df['High'] - feature_df['Low']).rolling(window=period).mean().pct_change(periods=period) * 100
        )

    ## 4. Momentum Indicators

    # Relative Strength Index (RSI)
    for period in [2, 7, 14, 21]:
        delta = feature_df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        feature_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    for period in [14, 21]:
        feature_df[f'Stoch_K_{period}'] = 100 * ((feature_df['Close'] - feature_df['Low'].rolling(window=period).min()) / 
                                         (feature_df['High'].rolling(window=period).max() - feature_df['Low'].rolling(window=period).min()))
        feature_df[f'Stoch_D_{period}'] = feature_df[f'Stoch_K_{period}'].rolling(window=3).mean()

    # ROC (Rate of Change)
    for period in [5, 10, 20]:
        feature_df[f'ROC_{period}'] = (feature_df['Close'] / feature_df['Close'].shift(period) - 1) * 100

    # Williams %R
    for period in [14, 20]:
        feature_df[f'Williams_R_{period}'] = -100 * (
            (feature_df['High'].rolling(window=period).max() - feature_df['Close']) / 
            (feature_df['High'].rolling(window=period).max() - feature_df['Low'].rolling(window=period).min())
        )

    # Commodity Channel Index (CCI)
    for period in [20]:
        tp = (feature_df['High'] + feature_df['Low'] + feature_df['Close']) / 3
        tp_sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        feature_df[f'CCI_{period}'] = (tp - tp_sma) / (0.015 * mad)

    ## 5. Trend Indicators

    # Average Directional Index (ADX)
    for period in [14]:
        # True Range
        high_low = feature_df['High'] - feature_df['Low']
        high_close = abs(feature_df['High'] - feature_df['Close'].shift(1))
        low_close = abs(feature_df['Low'] - feature_df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Plus Directional Movement (+DM)
        up_move = feature_df['High'] - feature_df['High'].shift(1)
        down_move = feature_df['Low'].shift(1) - feature_df['Low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        plus_dm = pd.Series(plus_dm, index=feature_df.index)
        
        # Minus Directional Movement (-DM)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        minus_dm = pd.Series(minus_dm, index=feature_df.index)
        
        # Smoothed +DM and -DM
        smooth_plus_dm = plus_dm.rolling(window=period).sum()
        smooth_minus_dm = minus_dm.rolling(window=period).sum()
        
        # Directional Indicators
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        feature_df[f'ADX_{period}'] = dx.rolling(window=period).mean()
        feature_df[f'Plus_DI_{period}'] = plus_di
        feature_df[f'Minus_DI_{period}'] = minus_di
        feature_df[f'DI_Diff_{period}'] = plus_di - minus_di

    # Directional Movement Index (DMI)
    for period in [14]:
        feature_df[f'DMI_{period}'] = abs(feature_df[f'Plus_DI_{period}'] - feature_df[f'Minus_DI_{period}']) / (feature_df[f'Plus_DI_{period}'] + feature_df[f'Minus_DI_{period}']) * 100

    ## 6. Volume Indicators (conditional based on volume_type_ticker)
    
    if volume_type_ticker:
        # Treat 'Volume' as tick counts (tick volume). Use normalized/robust features rather than raw magnitude.

        # Keep original column under a clear name
        feature_df['Tick_Volume'] = feature_df['Volume'].astype(float)

        # Detect partial bars (e.g. 30, 59) — useful as a feature or for masking
        EXPECTED_TICKS_PER_HOUR = 60
        feature_df['Partial_Bar_Flag'] = (feature_df['Tick_Volume'] < EXPECTED_TICKS_PER_HOUR).astype(int)

        # Rolling statistics (mean/std) to build z-scores and relative measures
        feature_df['TickVol_RollMean_20'] = feature_df['Tick_Volume'].rolling(window=20, min_periods=1).mean()
        feature_df['TickVol_RollStd_20'] = feature_df['Tick_Volume'].rolling(window=20, min_periods=1).std().replace(0, np.nan)
        feature_df['TickVol_Z_20'] = (feature_df['Tick_Volume'] - feature_df['TickVol_RollMean_20']) / feature_df['TickVol_RollStd_20']

        # Relative volume vs prior 24 hours (captures intraday spikes)
        feature_df['TickVol_Relative_24'] = feature_df['Tick_Volume'] / (feature_df['Tick_Volume'].rolling(window=24, min_periods=1).mean().replace(0, np.nan))

        # Log transform to reduce skew / handle zeros
        feature_df['TickVol_Log1p'] = np.log1p(feature_df['Tick_Volume'])

        # Robust OBV-like measure using deviation from typical tick volume (captures activity shocks)
        sign = np.sign(feature_df['Close'].diff()).fillna(0)
        dev_from_mean = (feature_df['Tick_Volume'] - feature_df['TickVol_RollMean_20']).fillna(0)
        feature_df['OBV_TickDev'] = (sign * dev_from_mean).cumsum()

        # Chaikin Money Flow (using tick volume as a proxy), normalized over window
        mf_multiplier = ((feature_df['Close'] - feature_df['Low']) - (feature_df['High'] - feature_df['Close'])) / (feature_df['High'] - feature_df['Low']).replace(0, np.nan)
        mf_volume = mf_multiplier * feature_df['Tick_Volume']
        feature_df['CMF_20_tick'] = mf_volume.rolling(window=20, min_periods=1).sum() / feature_df['Tick_Volume'].rolling(window=20, min_periods=1).sum().replace(0, np.nan)

        # Volume oscillator and ROC computed on tick volume but normalized
        for short_period, long_period in [(5, 10), (12, 26)]:
            short_mean = feature_df['Tick_Volume'].rolling(window=short_period, min_periods=1).mean()
            long_mean = feature_df['Tick_Volume'].rolling(window=long_period, min_periods=1).mean().replace(0, np.nan)
            feature_df[f'Volume_Osc_tick_{short_period}_{long_period}'] = (short_mean - long_mean) / long_mean * 100

        for period in [10, 20]:
            feature_df[f'Volume_ROC_tick_{period}'] = (feature_df['Tick_Volume'] / feature_df['Tick_Volume'].shift(period) - 1) * 100

        # Winsorize / clip extreme tick counts (optional) to reduce outlier influence
        upper_clip = feature_df['Tick_Volume'].quantile(0.99)
        feature_df['TickVol_Clipped'] = feature_df['Tick_Volume'].clip(lower=1, upper=upper_clip)
        
    else:
        # Traditional volume indicators for trading volume
        # On Balance Volume (OBV)
        feature_df['OBV_Change'] = np.where(feature_df['Close'] > feature_df['Close'].shift(1), feature_df['Volume'],
                                np.where(feature_df['Close'] < feature_df['Close'].shift(1), -feature_df['Volume'], 0))
        feature_df['OBV'] = feature_df['OBV_Change'].cumsum()

        # Chaikin Money Flow
        for period in [20]:
            mf_multiplier = ((feature_df['Close'] - feature_df['Low']) - (feature_df['High'] - feature_df['Close'])) / (feature_df['High'] - feature_df['Low'])
            mf_volume = mf_multiplier * feature_df['Volume']
            feature_df[f'CMF_{period}'] = mf_volume.rolling(window=period).sum() / feature_df['Volume'].rolling(window=period).sum()

        # Volume Oscillator
        for short_period, long_period in [(5, 10), (12, 26)]:
            feature_df[f'Volume_Osc_{short_period}_{long_period}'] = (
                feature_df['Volume'].rolling(window=short_period).mean() - 
                feature_df['Volume'].rolling(window=long_period).mean()
            ) / feature_df['Volume'].rolling(window=long_period).mean() * 100

        # Volume Rate of Change
        for period in [10, 20]:
            feature_df[f'Volume_ROC_{period}'] = (feature_df['Volume'] / feature_df['Volume'].shift(period) - 1) * 100

        # Parabolic SAR
        def psar(df, iaf=0.02, maxaf=0.2):
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            psar = close.copy()
            bull = True
            af = iaf
            ep = low[0]
            hp = high[0]
            lp = low[0]
            
            for i in range(2, len(df)):
                if bull:
                    psar[i] = psar[i-1] + af * (hp - psar[i-1])
                else:
                    psar[i] = psar[i-1] + af * (lp - psar[i-1])
                
                reverse = False
                
                if bull:
                    if low[i] < psar[i]:
                        bull = False
                        reverse = True
                        psar[i] = hp
                        lp = low[i]
                        af = iaf
                else:
                    if high[i] > psar[i]:
                        bull = True
                        reverse = True
                        psar[i] = lp
                        hp = high[i]
                        af = iaf
                
                if not reverse:
                    if bull:
                        if high[i] > hp:
                            hp = high[i]
                            af = min(af + iaf, maxaf)
                        if low[i-1] < psar[i]:
                            psar[i] = low[i-1]
                        if low[i-2] < psar[i]:
                            psar[i] = low[i-2]
                    else:
                        if low[i] < lp:
                            lp = low[i]
                            af = min(af + iaf, maxaf)
                        if high[i-1] > psar[i]:
                            psar[i] = high[i-1]
                        if high[i-2] > psar[i]:
                            psar[i] = high[i-2]
            
            return psar

        feature_df['PSAR'] = psar(feature_df)
        feature_df['PSAR_Dist'] = (feature_df['Close'] - feature_df['PSAR']) / feature_df['Close'] * 100
        feature_df['PSAR_Bull'] = (feature_df['PSAR'] < feature_df['Close']).astype(int)

    ## 7. Statistical Features and Pattern Recognition

    # Skewness and Kurtosis
    for period in [20, 50]:
        feature_df[f'Returns_Skewness_{period}'] = feature_df['Percentage_Change'].rolling(window=period).skew()
        feature_df[f'Returns_Kurtosis_{period}'] = feature_df['Percentage_Change'].rolling(window=period).kurt()

    # Z-Score
    for period in [20, 50]:
        feature_df[f'Price_Z_Score_{period}'] = (feature_df['Close'] - feature_df['Close'].rolling(window=period).mean()) / feature_df['Close'].rolling(window=period).std()
        feature_df[f'Returns_Z_Score_{period}'] = (feature_df['Percentage_Change'] - feature_df['Percentage_Change'].rolling(window=period).mean()) / feature_df['Percentage_Change'].rolling(window=period).std()

    # Autocorrelation
    for period in [5, 10]:
        feature_df[f'Autocorr_{period}'] = feature_df['Close'].rolling(window=period*2).apply(
            lambda x: pd.Series(x).autocorr(lag=period) if len(x) > period else np.nan
        )

    return feature_df


def compute_target_rule(df_features: pd.DataFrame, lookback: int = 20, prediction_horizon: int = 5, multiplier: int = 1) -> pd.DataFrame:
    """
    Recompute the manual target/label for the dataframe using the exact rule used in training.
    
    Args:
        df_features (pd.DataFrame): DataFrame with features including 'Close' and 'Percentage_Change'
        lookback (int): Lookback period for volatility calculation (default: 20)
        prediction_horizon (int): Predict price movement N periods ahead (default: 5)
        multiplier (int): Volatility multiplier for threshold calculation (default: 1)
        
    Returns:
        pd.DataFrame: DataFrame with added 'Target' column (3-class: 0=Down, 1=Up, 2=Sideways)
    """
    feature_df = df_features.copy()
    
    # Calculate rolling volatility from the Percentage_Change column
    vol = feature_df['Percentage_Change'].rolling(window=lookback).std()

    # Calculate future values (percentage change over prediction horizon)
    future_values = feature_df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)

    # Create dynamic thresholds based on recent volatility
    upper_threshold = vol * multiplier
    lower_threshold = -vol * multiplier

    # Generate target variable (3-class classification)
    feature_df['Target'] = 2  # Default: sideways (2)
    feature_df.loc[future_values > upper_threshold, 'Target'] = 1  # Up (1)
    feature_df.loc[future_values < lower_threshold, 'Target'] = 0  # Down (0)

    # Remove future data points to avoid lookahead bias
    feature_df = feature_df[:-prediction_horizon]
    
    return feature_df


if __name__ == "__main__":
    # Example usage and testing
    print("Testing MH_feature_engineering_baseline module...")
    
    # Create sample OHLCV data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(50, 70, 100)  # Simulating tick volume
    }, index=dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    sample_data['High'] = np.maximum(sample_data['High'], 
                                   np.maximum(sample_data['Open'], sample_data['Close']))
    sample_data['Low'] = np.minimum(sample_data['Low'], 
                                  np.minimum(sample_data['Open'], sample_data['Close']))
    
    print(f"Sample data shape: {sample_data.shape}")
    print("Sample data:")
    print(sample_data.head())
    
    # Test feature engineering
    try:
        features_with_target = compute_baseline_features(sample_data, volume_type_ticker=True)
        features_with_target = compute_target_rule(features_with_target)
        
        print(f"\nFeatures created successfully!")
        print(f"Final shape: {features_with_target.shape}")
        print(f"Number of features: {len([col for col in features_with_target.columns if col != 'Target'])}")
        print(f"Target distribution: {features_with_target['Target'].value_counts().to_dict()}")
        print(f"Feature columns: {list(features_with_target.columns[:10])}...")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()