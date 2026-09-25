# Trading Signal Machine Learning & Wall Street Simulator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Can machine learning algorithms actually learn when to buy and sell stocks using standard price behavior, without cheating by looking at the answers?

This project takes nearly five decades of real-world stock market history (over 20 million daily records from 1970 to 2018) and tests whether supervised models can spot rare trading opportunities. Along the way, it tackles the **Accuracy Paradox** (where a model looks 99.9% accurate on paper while failing completely in practice), enforces strict anti-leakage rules, and includes a built-in retro Wall Street trading simulation.

---

## System Architecture

Here is how raw equity data flows from raw numbers to trained models and our interactive trading lab:

| Stage | Name | What Happens Under the Hood |
| :--- | :--- | :--- |
| **Stage 1** | **Raw Market Data** | Ingest 48 years of daily equity history (1970–2018; 20,860,334 clean rows). Contains closing prices, volume, and daily ranges across thousands of public companies. |
| **Stage 2** | **Technical Sensor Hub** | Build our two indicators from scratch: MACD (tracks trend momentum) and RSI (tracks whether an asset is stretched too far up or down). |
| **Stage 3** | **Dual Agreement Rule** | A trade only triggers when both indicators agree simultaneously. Both say Buy = BUY. Both say Sell = SELL. Anything else = HOLD (99.91% of all days). |
| **Stage 4** | **Anti-Cheating Firewall** | Split strictly by time (learn pre-2015; test strictly on 2015–2018). Strip MACD and RSI out of the model's training notes so it cannot memorize the answers. Fit scalers strictly on training data. |
| **Stage 5** | **Three AI Architectures** | Benchmark three distinct models: Logistic Regression (boundary lines), Random Forest (100+ voting decision trees), and Linear SVM (safety margins). |
| **Stage 6** | **Tuning for Imbalance** | Penalize missed Buy and Sell signals ~2,000× heavier in Random Forest using cost-sensitive sub-sampling to wake the model up. |
| **Stage 7** | **Interactive Trading Lab** | An in-notebook, zero-dependency HTML5/SVG trading floor running on real out-of-sample test data with live position sizing and an active broker commentary desk. |

---

## What Makes This Project Stand Out

- **Massive Historical Foundation**: 20.8 million daily market records across 48 years—real historical price spikes, crashes, and sideways chop.
- **Strict Anti-Leakage Firewall**: Zero future peeking. The models never see future dates, and the mathematical formulas used to build the target are kept strictly out of the training features.
- **The Accuracy Trap in Action**: Demonstrates why a model that boasts "99.9% accuracy" can be completely useless in live trading.
- **Zero-Dependency Trading Simulator**: Runs directly inside the notebook using HTML5 and SVG. Trade in 100-share blocks against real test data, manage portfolio cash, and listen to a reactive Wall Street broker on the line.
- **Real-World Explainability**: Gini importance rankings demonstrate whether models care more about a stock's dollar price or its underlying price turbulence.

---

## How the Trading Indicators Work (In Plain English)

### 1. MACD: The Momentum Escalator
Think of a stock’s price like someone walking on an escalator. Prices wiggle every single minute, but what we really want to know is: *is the escalator moving up or down, and is it picking up speed?*

MACD does this by comparing two moving averages:
- **Fast Average (12 Days)**: Reacts quickly to short-term price moves.
- **Slow Average (26 Days)**: A smoother line showing the monthly trend.
- **MACD Line**: Fast Average minus Slow Average. When it climbs above zero, short-term buyers have taken over.
- **Signal Line**: A 9-day smoothed average of the MACD line itself, used as a trigger line.

**The Action Rules**:
- **BUY**: The MACD line crosses strictly above the Signal line (upward momentum takes off).
- **SELL**: The MACD line crosses strictly below the Signal line (momentum rolls over).
- **HOLD**: No crossover happened today.

---

### 2. RSI: The Rubber Band Indicator
If MACD tracks the direction of the trend, the **Relative Strength Index (RSI)** measures whether price has been stretched too far in one direction. Just like a rubber band, when buyers or sellers pull price too hard in two weeks, it tends to snap back.

RSI tracks average gains versus average losses over the last 14 trading days and gives the stock a score from 0 to 100:
- **Oversold (Score under 30 → BUY Zone)**: Relentless selling has stretched the band downward. The asset is at a steep discount and due for a bounce.
- **Overbought (Score over 70 → SELL Zone)**: Aggressive buying has stretched the band upward. Buyers are exhausted and a pullback is likely.
- **Neutral (Score between 30 and 70 → HOLD Zone)**: Normal day-to-day market conditions.

---

### 3. The Consensus Rule: Dual Confirmation
Individual indicators often trigger false alarms (whipsaws). To build a disciplined strategy, we demand dual confirmation: **both indicators must agree on the exact same trading day.**

| Target Class | Total Market Days | Share of Market | What Does It Mean? |
| :--- | :--- | :--- | :--- |
| **HOLD (1)** | **20,842,154** | **99.9128%** | Indicators disagree or the market is drifting normally. Sit on your hands. |
| **SELL (2)** | **10,103** | **0.0484%** | Price hit an overbought peak (>70) while momentum crossed downward on the same day. |
| **BUY (0)** | **8,077** | **0.0387%** | Price hit an oversold discount (<30) while momentum crossed upward on the same day. |

Because true market alignment is rare, fewer than 1 in 1,000 market days triggered an actionable setup.

---

## Machine Learning Results: The Accuracy Trap

We trained our models on pre-2015 historical data and evaluated them strictly on unseen test data from 2015 through 2018 (50,000 representative records):

| Model Architecture | Accuracy Score | Buy Catch Rate (Recall) | Sell Catch Rate (Recall) | Macro F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 99.9000% | 0.0000 | 0.0000 | 0.3332 |
| **Random Forest (Baseline)** | 99.9000% | 0.0000 | 0.0000 | 0.3332 |
| **Linear Support Vector Machine** | 99.9000% | 0.0000 | 0.0000 | 0.3332 |
| **Improved Random Forest (Tuned)** | 99.8980% | 0.0000 | 0.0000 | 0.3332 |

### What Happened Here? (The Accuracy Paradox)
Look at the numbers: every baseline model scored **99.90% accuracy**, yet their recall on Buys and Sells was **zero**. 

How? The algorithms discovered a clever shortcut. Because 99.9% of all trading days are Hold, a model that simply guesses "HOLD" on every single row will be right 9,990 times out of 10,000. It never gets penalized for a false alarm, but it misses 100% of actual trading opportunities. 

In financial machine learning, **accuracy without recall is a vanity metric**.

### Fixing the Bias with Cost-Sensitive Learning
To stop the model from taking the easy way out, we tuned Random Forest with balanced sub-sample class weighting. Whenever the algorithm missed a rare Buy or Sell, we hit it with a penalty roughly 2,000× heavier than missing a standard Hold. 

**The Key Takeaway**: While the penalty forced the tree splits to reconsider minority patterns, having only 55 Buys and 70 Sells out of 150,000 training rows meant individual decision trees often drew zero minority examples in their bootstrap samples. When dealing with extreme real-world imbalance (under 0.1%), algorithmic weighting must be paired with physical dataset rebalancing (undersampling the majority class) to carve out stable decision boundaries.

---

## Feature Importance: What Did the AI Care About?

Looking inside the 150 decision trees of our tuned Random Forest reveals which price features actually helped separate signals from noise:

| Feature Name | Gini Importance | Influence Share | Plain English Meaning |
| :--- | :--- | :--- | :--- |
| **volatility_20** | **0.2118** | **21.18%** | **Market turbulence**: breakouts happen when volatility expands. |
| **daily_return** | **0.1885** | **18.85%** | **Price speed**: sharp jumps up or down indicate momentum shifts. |
| **daily_range_pct** | **0.1778** | **17.78%** | **Intraday spread**: the daily battle between high and low. |
| **ma_20** | **0.1445** | **14.45%** | **Monthly baseline**: where is price compared to the 20-day trend? |
| **ma_5** | **0.1420** | **14.20%** | **Weekly baseline**: where is price compared to the 5-day trend? |
| **close_lag_1** | **0.1354** | **13.54%** | **Nominal dollar tag**: yesterday's raw price level. |

**The Big Pattern**: Dynamic features (volatility, price speed, and intraday range) accounted for nearly **60% of all decisions**. A stock's nominal dollar tag (`close_lag_1`) was the least informative feature. This matches real trading instincts: a stock trading at $40 versus $400 tells you nothing about an upcoming breakout; its velocity and volatility do.

---

## Interactive Trading Floor Simulator

The notebook includes a playable trading terminal built directly with HTML5, CSS, and SVG:

- **100-Share Block Execution**: Place orders in realistic sizes to see actual P&L swings on your $10,000 starting bankroll.
- **Moving Average Spread Tracking**: Live percentage readout tracking how far price has deviated above or below its 20-day trendline.
- **Interactive Wall Street Broker**: A reactive 1987-style advisory desk that calls you out for holding cash during rallies, warns you against catching falling knives, and celebrates locked-in gains.
- **Closing Bell Summary**: A 60-day terminal alert that recaps your final portfolio return against the broader market benchmark.

> **GitHub Display Note**: GitHub sanitizes active JavaScript in notebook views. To play with the live simulator, open `Project3_CesarJuarez.ipynb` in [Google Colab](https://colab.research.google.com/) or run it locally in Jupyter.

---

## Project Structure

```text
.
├── Project3_CesarJuarez.ipynb        # Complete Colab/Jupyter notebook with pipeline & simulator
├── Project3_Report_CesarJuarez.docx  # Formatted executive Word report
├── README.md                         # Complete project documentation
└── assets/                           # Static preview snapshots for GitHub
    ├── architecture_blueprint.png    # Pipeline flow diagram
    └── trading_simulator.png         # Full trading floor lab snapshot
```

---

## Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/trading-signal-ml-prediction.git](https://github.com/your-username/trading-signal-ml-prediction.git)
cd trading-signal-ml-prediction
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Required Libraries*:
```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
python-docx
```

### 3. Open the Notebook
```bash
jupyter notebook Project3_CesarJuarez.ipynb
```

---

## Reflections & Strategic Lessons

1. **Accuracy is Often a Trap**: A model scoring 99.9% accuracy that catches zero trades is worthless. Always evaluate precision, recall, and Macro F1 on minority events.
2. **Preventing Target Leakage**: If you use an indicator to create your labels, you must strip that indicator from your input features. Otherwise, your model is just memorizing math formulas instead of learning market behavior.
3. **Volatility Precedes Momentum**: The decision trees confirmed that volatility expansion and price speed are the strongest signals of upcoming momentum shifts.
4. **Paper Signals vs. Real Profits**: Predicting an indicator crossover does not guarantee trading profits. Live trading incurs slippage, broker fees, spread costs, and unpredictable macro shocks.

---

## Author

- **Cesar Juarez**  
  *Data Analytics and Artificial Intelligence*

---
