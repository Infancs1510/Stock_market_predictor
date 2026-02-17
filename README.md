# Smart Data Analyzer - Stock Direction Prediction

A machine learning-powered stock price direction predictor that combines technical analysis, feature engineering, and gradient boosting to forecast short-term stock movement.

## 🎯 Features

- **Technical Indicators**: Comprehensive feature engineering including RSI, MACD, Bollinger Bands, ATR, EMA, and Volume-Based indicators
- **ML Model**: HistGradientBoostingClassifier with probability calibration for robust predictions
- **Risk Management**: Automatic threshold optimization, volatility detection, and risk scoring
- **Backtesting**: Walk-forward cross-validation for realistic performance evaluation
- **AI Insights**: Integration with Ollama for natural language analysis of predictions
- **Time-Series Safe**: Proper train/test split respecting temporal order (no data leakage)

## 📊 Key Metrics

The model evaluates predictions on:
- **Balanced Accuracy**: Accounts for class imbalance in market data
- **ROC-AUC**: Measures discriminative ability across thresholds
- **Precision & Recall**: Identifies reliability of each signal direction
- **Confusion Matrix**: Shows false positive/negative breakdown

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd smart-data-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Analyze single ticker
python analyze.py AAPL

# Analyze multiple tickers
python analyze.py AAPL NVDA TSLA MSFT

# Use different period (default: 5y)
python analyze.py AAPL  # Uses 5 years of data
```

### Example Output

```
Signal: BUY (consider)
P(UP): 0.65
Trend20: 0.045
Vol20: 0.28 (ok)
RSI14: 42.5

Metrics:
- Balanced Accuracy: 0.58
- ROC-AUC: 0.62
- Precision: 0.54

Reasons:
- Model edge check: ROC-AUC=0.62 (needs >=0.52 for BUY)
- Model P(up over next 5d): 0.65
- Trend20 (Close vs SMA20): 0.045
- ...
```

## 🔧 How It Works

### 1. Data Collection
- Downloads 5 years of OHLCV (Open, High, Low, Close, Volume) data via `yfinance`
- Handles multi-index flattening and data type conversions

### 2. Feature Engineering (30+ features)
- **Price Momentum**: Returns, log returns, gaps, candle patterns
- **Volatility**: Rolling standard deviation (10d, 20d), ATR, Bollinger Band width
- **Trend Indicators**: SMA (10/20/50), EMA, MACD, price-to-moving-average ratio
- **Oscillators**: RSI, Bollinger Band %B, OBV change
- **Calendar**: Day of week, month seasonality
- **Volume**: Volume change, volume Z-score

### 3. Label Creation
- Target: 1 if price closes higher in 5 days, 0 otherwise
- 5-day forward-looking prediction (customizable via `horizon_days`)

### 4. Model Training
- **Algorithm**: HistGradientBoostingClassifier (fast, handles missing values)
- **Hyperparameters**: 
  - Learning rate: 0.06
  - Max depth: 3
  - Iterations: 600
  - L2 regularization: 0.08
- **Class Imbalance**: Sample weights adjusted for class distribution
- **Probability Calibration**: Sigmoid calibration for reliable confidence scores

### 5. Threshold Optimization
- Validates on 80% of training data
- Tests thresholds from 0.35 to 0.65
- Picks threshold maximizing balanced accuracy
- Prevents overfitting to training data

### 6. Risk Assessment
- Checks model edge (ROC-AUC ≥ 0.52)
- Evaluates trend strength (SMA20 ratio)
- Detects high volatility conditions (>60%)
- Confirms RSI not overbought (RSI < 75)

### 7. Backtesting
- Walk-forward expanding-window evaluation
- 5 independent folds for robust performance
- Reports average accuracy, balanced accuracy, and ROC-AUC

## 📈 Signal Interpretation

- **BUY (consider)**: Model predicts up with edge + positive trend + manageable risk
- **SELL/AVOID**: Model predicts down or trend negative
- **WAIT**: Mixed signals or insufficient edge

### Important Disclaimers
- Historical backtests ≠ future performance
- Always perform your own due diligence
- This is educational; not financial advice
- Stock market has inherent uncertainty

## 🤖 AI-Powered Explanations

Requires Ollama (local LLM):
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
ollama serve

# In another terminal, run analysis
python analyze.py AAPL
```

The system generates natural language explanations of:
- Computed signal rationale
- Confidence levels
- Key technical factors
- Risk considerations

## 📦 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (HistGradientBoosting, calibration, metrics)
- **yfinance**: Yahoo Finance data download
- **requests**: HTTP calls to Ollama API

See [requirements.txt](requirements.txt) for exact versions.

## 📁 Project Structure

```
smart-data-analyzer/
├── analyze.py           # Main ML pipeline
├── data.csv            # Sample historical data
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore patterns
├── LICENSE            # MIT License
└── README.md          # This file
```

## 🔍 Technical Details

### Temporal Safety
- Indicators computed on past data only (no lookahead bias)
- Train: First 80% of chronological data
- Validation: Next 10% of train set
- Test: Final 20% (most recent)
- No shuffling preserves temporal relationships

### Class Imbalance Handling
Markets are often biased toward up/down. The model:
- Calculates per-class sample weights
- Upweights minority class during training
- Evaluates with balanced_accuracy_score (not just raw accuracy)

### Probability Calibration
- Base model + sigmoid calibration (3-fold CV)
- Ensures predicted probabilities reflect actual likelihood
- More stable than isotonic regression for extrapolation

## 🎓 Educational Value

Great project to showcase:
- Time-series machine learning
- Feature engineering for finance
- Scikit-learn advanced techniques
- Proper train/test methodology
- Model evaluation practices
- End-to-end ML pipeline

## ⚖️ License

MIT License - See [LICENSE](LICENSE) file for details.

## 👤 Author

Created as a portfolio project demonstrating ML applied to real-world financial data.

---

**Disclaimer**: This tool is for educational and research purposes only. It does not constitute financial advice. Always consult financial advisors and perform your own analysis before trading.
