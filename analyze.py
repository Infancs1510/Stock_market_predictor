"""
Smart Data Analyzer - Stock Direction Prediction
================================================

A machine learning pipeline for predicting short-term stock price direction using:
- 30+ technical indicators and features
- HistGradientBoosting classifier with probability calibration
- Walk-forward backtesting for robust evaluation
- Ollama LLM integration for AI-powered insights

Author: Portfolio Project
License: MIT
"""

import sys
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: Time period for data ('1y', '5y', '10y', etc.)
        
    Returns:
        DataFrame with columns: [Date, Open, High, Low, Close, Volume]
        
    Raises:
        Handles multi-index flattening from yfinance API
    """
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)

    # flatten multiindex if it happens
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare raw OHLCV data.
    
    Actions:
    - Strips whitespace from column names
    - Converts Date to datetime
    - Converts OHLCV to numeric (handles errors)
    - Drops rows with missing values
    - Sorts by date
    
    Args:
        df: Raw dataframe from yfinance
        
    Returns:
        Cleaned dataframe ready for feature engineering
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    RSI measures momentum and overbought/oversold conditions (0-100 range).
    - RSI > 70: Potentially overbought
    - RSI < 30: Potentially oversold
    
    Args:
        close: Series of closing prices
        period: Lookback period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 30+ technical features for machine learning.
    
    Feature categories:
    1. Calendar: Day of week, month (seasonality)
    2. Momentum: Returns, log returns, gaps
    3. Volatility: Rolling std, ATR, Bollinger Bands
    4. Trend: Moving averages, price ratios
    5. Oscillators: MACD, RSI, Bollinger Bands %B
    6. Volume: Volume change, on-balance volume
    
    Args:
        df: Cleaned dataframe with OHLCV
        
    Returns:
        DataFrame with 30+ additional feature columns
        
    Note:
        - Computes on sorted data (preserves time-series properties)
        - Uses 1e-12 for division safety (avoids inf/nan)
        - Features are properly lagged (no lookahead bias)
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    vol = df["Volume"]
    prev_close = close.shift(1)

    # --- Calendar features (often help a tiny bit)
    df["dow"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"] = df["Date"].dt.month

    # --- Returns / momentum (different timeframes)
    df["ret_1d"] = close.pct_change()  # 1-day return %
    df["ret_5d"] = close.pct_change(5)  # 5-day return %
    df["ret_10d"] = close.pct_change(10)  # 10-day return %
    df["logret_1d"] = np.log(close).diff()  # Log return

    # --- Gaps and candle patterns
    df["gap"] = (open_ - prev_close) / (prev_close + 1e-12)  # Price gap from close to open
    df["hl_range"] = (high - low) / (close + 1e-12)  # Intraday range as % of close
    df["co_change"] = (close - open_) / (open_ + 1e-12)  # Close vs open (body %)

    # --- Volatility (annualized to 252 trading days)
    df["vol_10d"] = df["ret_1d"].rolling(10).std() * np.sqrt(252)  # 10-day rolling volatility
    df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)  # 20-day rolling volatility

    # --- Trend indicators (moving averages and deviation)
    df["sma_10"] = close.rolling(10).mean()  # 10-day simple moving average
    df["sma_20"] = close.rolling(20).mean()  # 20-day simple moving average
    df["sma_50"] = close.rolling(50).mean()  # 50-day simple moving average
    df["trend_10"] = close / (df["sma_10"] + 1e-12) - 1  # % deviation from SMA10
    df["trend_20"] = close / (df["sma_20"] + 1e-12) - 1  # % deviation from SMA20
    df["trend_50"] = close / (df["sma_50"] + 1e-12) - 1  # % deviation from SMA50

    # --- EMA + MACD (momentum indicators)
    ema12 = close.ewm(span=12, adjust=False).mean()  # 12-period exponential moving average
    ema26 = close.ewm(span=26, adjust=False).mean()  # 26-period exponential moving average
    df["macd"] = ema12 - ema26  # MACD line
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()  # Signal line
    df["macd_hist"] = df["macd"] - df["macd_signal"]  # MACD histogram

    # --- RSI (relative strength index)
    df["rsi_14"] = rsi(close, 14)

    # --- ATR (average true range - volatility measure)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()  # Average true range
    df["atr_pct"] = df["atr_14"] / (close + 1e-12)  # ATR as % of close

    # --- Bollinger Bands (volatility bands)
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    upper = mid + 2 * sd  # Upper band
    lower = mid - 2 * sd  # Lower band
    df["bb_pctb"] = (close - lower) / ((upper - lower) + 1e-12)  # % between bands
    df["bb_width"] = (upper - lower) / (mid + 1e-12)  # Band width as % of middle

    # --- Volume features (volume-based indicators)
    df["vol_chg_1d"] = vol.pct_change()  # Volume change %
    vmean = vol.rolling(20).mean()  # Mean volume
    vstd = vol.rolling(20).std()  # Std dev of volume
    df["vol_z20"] = (vol - vmean) / (vstd + 1e-12)  # Volume Z-score

    # --- OBV (on-balance volume - cumulative volume indicator)
    direction = np.sign(close.diff()).fillna(0)  # +1 if up, -1 if down
    df["obv"] = (direction * vol).cumsum()  # Cumulative OBV
    df["obv_change_5d"] = df["obv"].pct_change(5)  # 5-day OBV change %

    return df


# ============================================================================
# LABELING & MODEL TRAINING
# ============================================================================

def make_label(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """
    Create target labels for supervised learning.
    
    Label = 1 if price closes higher in N days, 0 otherwise.
    This creates a binary classification problem.
    
    Args:
        df: Feature dataframe
        horizon_days: Forward-looking period (default 5)
        
    Returns:
        DataFrame with 'y' column (0/1) and 'future_close'
    """
    df = df.copy()
    df["future_close"] = df["Close"].shift(-horizon_days)
    df["y"] = (df["future_close"] > df["Close"]).astype(int)
    return df


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Find optimal classification threshold using validation data.
    
    Tests thresholds from 0.35 to 0.65 and selects the one maximizing
    balanced_accuracy_score (fairness across classes).
    
    Args:
        y_true: True labels
        proba: Model predicted probabilities
        
    Returns:
        Optimal threshold (float)
    """
    best_t = 0.50
    best_score = -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (proba >= t).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_t = float(np.round(t, 2))
    return best_t


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute sample weights to handle class imbalance.
    
    Stock markets are often biased (more up or down days). Weights ensure
    each class contributes equally during training.
    
    Args:
        y: Class labels (0 or 1)
        
    Returns:
        Weight array (same length as y)
    """
    n = len(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    # avoid division by zero
    w_pos = n / (2 * max(pos, 1))
    w_neg = n / (2 * max(neg, 1))
    return np.where(y == 1, w_pos, w_neg)


def train_predict_signal(df: pd.DataFrame, horizon_days: int = 5) -> dict:
    """
    Main ML pipeline: train model and generate trading signal.
    
    Process:
    1. Create labels for N-day horizon
    2. Engineer 30+ technical features
    3. Time-based train/val/test split
    4. Train HistGradientBoosting with calibration
    5. Optimize threshold on validation set
    6. Evaluate on test set
    7. Generate latest signal
    
    Args:
        df: OHLCV dataframe with features from add_features()
        horizon_days: Prediction horizon (default 5)
        
    Returns:
        dict with: signal, proba_up, trend20, vol_20d, rsi_14, metrics, reasons
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df = make_label(df, horizon_days=horizon_days)

    # Feature selection - all 30+ technical indicators
    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "logret_1d",
        "gap", "hl_range", "co_change",
        "vol_10d", "vol_20d",
        "trend_10", "trend_20", "trend_50",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "atr_pct",
        "bb_pctb", "bb_width",
        "obv_change_5d",
        "vol_chg_1d", "vol_z20",
        "dow", "month",
    ]

    # Remove rows with NaN in features or target
    df = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    if len(df) < 300:
        return {"error": "Not enough usable rows after indicators. Use period='10y' or reduce features."}

    # TIME-BASED SPLIT (no shuffling to preserve temporal order)
    split = int(len(df) * 0.8)  # 80% train, 20% test
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    # VALIDATION SPLIT within training data (for threshold optimization)
    val_split = int(len(train) * 0.8)  # 80% of train for training, 20% for validation
    train_inner = train.iloc[:val_split].copy()
    val = train.iloc[val_split:].copy()

    X_train = train_inner[feature_cols]
    y_train = train_inner["y"].values

    X_val = val[feature_cols]
    y_val = val["y"].values

    X_test = test[feature_cols]
    y_test = test["y"].values

    # Handle class imbalance with sample weights
    sw = class_balance_weights(y_train)

    # Base model: HistGradientBoosting (fast, handles missing values)
    base = HistGradientBoostingClassifier(
        learning_rate=0.06,  # Moderate learning rate
        max_depth=3,  # Shallow trees prevent overfitting
        max_iter=600,  # Iterations before early stop
        l2_regularization=0.08,  # L2 penalty for regularization
        random_state=42,
        early_stopping=True,
    )

    # Train base model ONCE on train_inner
    base.fit(X_train, y_train, sample_weight=sw)

    # Calibrate probabilities using cross-validation
    # Sigmoid method is more stable than isotonic for extrapolation
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train, sample_weight=sw)

    # Find optimal threshold on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, val_proba)

    # Evaluate on test set (unseen data)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    # Compute detailed metrics
    metrics_out = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "threshold_used": float(threshold),
        "horizon_days": int(horizon_days),
        "test_size": int(len(test)),
    }
    auc = metrics_out["roc_auc"]
    has_edge = auc >= 0.52  # Model must beat random (0.5) meaningfully

    # Generate LATEST signal (most recent bar)
    latest = df.iloc[-1]
    latest_X = latest[feature_cols].to_frame().T
    proba_up = float(model.predict_proba(latest_X)[:, 1][0])

    vol20 = float(latest["vol_20d"])
    trend20 = float(latest["trend_20"])
    rsi14 = float(latest["rsi_14"])
    high_risk = vol20 > 0.60  # Volatility > 60% is considered high risk

    # Decision thresholds (adaptive based on validation)
    buy_thresh = max(0.58, threshold)
    sell_thresh = 1.0 - buy_thresh

    # Signal logic - multiple conditions for robustness
    if has_edge and (proba_up >= buy_thresh) and (trend20 > 0) and (not high_risk) and (rsi14 < 75):
        signal = "BUY (consider)"
    elif (proba_up <= sell_thresh) or ((trend20 < 0) and (proba_up < 0.52)):
        signal = "SELL / AVOID (for now)"
    else:
        signal = "WAIT"

    # Detailed reasoning for each signal component
    reasons = [
        f"Model edge check: ROC-AUC={auc:.2f} (needs >=0.52 for BUY)",
        f"Model P(up over next {horizon_days}d): {proba_up:.2f}",
        f"Trend20 (Close vs SMA20): {trend20:.3f}",
        f"Vol20 (annualized): {vol20:.2f}" + (" (high risk)" if high_risk else " (ok)"),
        f"RSI14: {rsi14:.1f}",
        f"Decision thresholds: BUY>={buy_thresh:.2f}, SELL<={sell_thresh:.2f} (val-picked threshold={threshold:.2f})",
    ]

    return {
        "signal": signal,
        "proba_up": proba_up,
        "trend20": trend20,
        "vol_20d": vol20,
        "rsi_14": rsi14,
        "metrics": metrics_out,
        "reasons": reasons,
    }


# ============================================================================
# WALK-FORWARD BACKTESTING
# ============================================================================

def walk_forward_backtest(df: pd.DataFrame, horizon_days: int = 5, folds: int = 5) -> dict:
    """
    Perform expanding-window walk-forward backtesting.
    
    Simulates realistic trading scenario:
    - Expanding training window (more data each fold)
    - Never trains on future data
    - Reports performance across all folds
    
    This is the gold standard for time-series model evaluation.
    
    Args:
        df: OHLCV dataframe
        horizon_days: Prediction horizon
        folds: Number of evaluation periods
        
    Returns:
        dict with avg metrics across folds
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df = make_label(df, horizon_days=horizon_days)

    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "logret_1d",
        "gap", "hl_range", "co_change",
        "vol_10d", "vol_20d",
        "trend_10", "trend_20", "trend_50",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "atr_pct",
        "bb_pctb", "bb_width",
        "obv_change_5d",
        "vol_chg_1d", "vol_z20",
        "dow", "month",
    ]

    df = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    n = len(df)
    if n < 400:
        return {"note": "Not enough data for walk-forward backtest."}

    fold_size = n // (folds + 1)  # Size of each fold
    results = []

    # Expanding window: each fold uses all previous data for training
    for i in range(1, folds + 1):
        train_end = fold_size * i  # Expanding window boundary
        test_end = min(train_end + fold_size, n)  # Test period

        train = df.iloc[:train_end]  # All data up to this point
        test = df.iloc[train_end:test_end]  # Future period for testing
        if len(test) < 50:
            continue

        X_train = train[feature_cols]
        y_train = train["y"].values
        X_test = test[feature_cols]
        y_test = test["y"].values

        sw = class_balance_weights(y_train)

        # Train base model
        base = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=3,
            max_iter=600,
            l2_regularization=0.08,
            random_state=42,
            early_stopping=True,
        )
        base.fit(X_train, y_train, sample_weight=sw)

        # Calibrate probabilities
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train, y_train, sample_weight=sw)

        # Predict on future period
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.50).astype(int)

        # Collect metrics for this fold
        results.append({
            "acc": accuracy_score(y_test, pred),
            "bacc": balanced_accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
        })

    if not results:
        return {"note": "Walk-forward backtest produced no folds."}

    # Return average metrics across all folds
    return {
        "folds": len(results),
        "avg_accuracy": float(np.mean([r["acc"] for r in results])),
        "avg_balanced_accuracy": float(np.mean([r["bacc"] for r in results])),
        "avg_roc_auc": float(np.mean([r["auc"] for r in results])),
    }


# ============================================================================
# AI-POWERED EXPLANATIONS (OLLAMA)
# ============================================================================

def ollama_explain(model: str, ticker: str, payload: dict) -> str:
    """
    Generate natural language explanation of trading signal using Ollama.
    
    Calls local LLM (llama3) to explain the computed signal in human-readable
    terms. This adds interpretability to the quantitative analysis.
    
    Requirements:
    - Ollama installed (https://ollama.ai)
    - Local LLM running: ollama pull llama3 && ollama serve
    
    Args:
        model: Ollama model name (e.g., 'llama3')
        ticker: Stock ticker symbol
        payload: Dictionary containing signal and metrics
        
    Returns:
        Natural language explanation string
        
    Raises:
        requests.exceptions.RequestException: If Ollama not running
    """
    prompt = f"""
You are a stock analysis assistant.
Do NOT invent a trading decision. The decision is already computed.
Do NOT mention profits, money, gains, or guarantees.
If asked about profit, respond: "I can’t help with profit predictions."

Explain ONLY the computed signal using the JSON.

Output format:
Signal: <repeat exactly>
Confidence: <use proba_up>
Key reasons: 3 bullets
Risks: 3 bullets
What would change the signal: 2 bullets

Ticker: {ticker}
JSON:
{json.dumps(payload, indent=2)}
"""
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def analyze_ticker(ticker: str, period: str = "5y", horizon_days: int = 5):
    """
    Complete analysis pipeline for a single ticker.
    
    Process:
    1. Download OHLCV data
    2. Clean data
    3. Engineer features
    4. Train/predict model
    5. Run walk-forward backtest
    6. Generate AI explanation
    
    Args:
        ticker: Stock ticker symbol
        period: Data period ('5y', '10y', etc.)
        horizon_days: Prediction horizon
        
    Returns:
        Prints analysis to stdout
    """
    df = fetch_stock_data(ticker, period=period)
    if df.empty:
        print(f"No data returned for {ticker}")
        return

    df = clean_dataframe(df)
    df = add_features(df)

    print(f"\nLoaded {ticker} | rows={len(df)} cols={df.shape[1]}")
    print(df[["Date", "Open", "High", "Low", "Close", "Volume"]].head(), "\n")

    # Train model and get signal
    result = train_predict_signal(df, horizon_days=horizon_days)
    if "error" in result:
        print("!!", result["error"])
        return

    print(f"Direction + Signal ({horizon_days}-day horizon)")
    print("Signal:", result["signal"])
    print("P(UP):", f"{result['proba_up']:.2f}")
    print("Metrics:", result["metrics"])
    print("Reasons:")
    for r in result["reasons"]:
        print(" -", r)

    # Walk-forward backtest
    wf = walk_forward_backtest(df, horizon_days=horizon_days, folds=5)
    print("\nWalk-forward backtest:", wf)

    # AI-powered explanation (requires Ollama)
    try:
        summary = ollama_explain(model="llama3", ticker=ticker, payload=result)
        print("\nOllama summary:")
        print(summary)
    except Exception as e:
        print(f"\nOllama explanation skipped: {e}")
        print("(Install Ollama to enable AI explanations)")


if __name__ == "__main__":
    # Entry point: CLI interface
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <TICKER1> [TICKER2 ...]")
        print("Example: python analyze.py AAPL NVDA TSLA")
        sys.exit(1)

    tickers = [t.upper() for t in sys.argv[1:]]
    for t in tickers:
        analyze_ticker(t, period="5y", horizon_days=5)
