# 🌊 ForexFlow Analytics Dashboard

**Systematic Foreign Exchange Machine Learning Inference Terminal**

ForexFlow is an interactive quantitative analytics dashboard built with **Streamlit**. It ingests 720 days of hourly historical forex data, engineers technical and temporal features, trains a Random Forest classifier to predict next-hour directional price movement, and calculates Average True Range (ATR) volatility exit targets alongside a live macro news ticker.

---

## 🛠️ Required Libraries

To run this application, you need Python 3.8+ and the following third-party libraries (included in `requirements.txt`):

| Library | Purpose in App |
| :--- | :--- |
| **`streamlit`** | Powers the web application UI, sidebar controls, layout columns, and custom CSS cards. |
| **`numpy`** | Handles logarithmic return calculations, array operations, and vectorized signal logic. |
| **`pandas`** | Manages time-series DataFrames, rolling window indicators (RSI, ATR, Volatility), and data slicing. |
| **`yfinance`** | Downloads 720 days of 1-hour interval market data for major currency pairs (`EUR/USD`, `USD/JPY`, `GBP/USD`, `USD/CAD`, `AUD/USD`). |
| **`plotly`** | Renders interactive candlestick charts with Take Profit / Stop Loss overlays and feature importance bar charts. |
| **`scikit-learn`** | Provides the `RandomForestClassifier` used for directional price forecasting and out-of-sample backtesting. |
| **`requests`** | Fetches the live XML RSS news feed from Myfxbook for the scrolling macro impact ticker. |
| **`beautifulsoup4`** | Installed alongside `requests` for HTML/XML parsing support *(Note: core XML parsing uses Python's built-in `xml.etree.ElementTree`)*. |

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd forexflow-terminal
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install all dependencies using `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *(Replace `app.py` with the actual filename of your Python script if different.)*

---

## ✨ Key Features

* **Real-Time KPI Matrix:** Displays live closing price, net/percentage change, session high/low bounds, and 24-hour rolling return volatility.
* **Live High-Impact Macro Ticker:** Parses live financial headlines via RSS (`myfxbook.com`) into a scrolling alert banner.
* **ML Directional Forecasting:** Uses a 200-tree `RandomForestClassifier` trained on lagged log returns (`Lag_1`, `Lag_2`), 14-period RSI (`RSI_14`), and intraday hour (`Hour`) to generate `BUY (LONG)` or `SELL (SHORT)` verdicts with confidence probabilities.
* **ATR Execution Brackets:** Automatically plots volatility-adjusted **Take Profit (1.5x ATR)** and **Stop Loss (1.0x ATR)** levels directly onto the interactive Plotly candlestick chart.
* **Out-of-Sample Model Validation:** Includes an expandable academic validation panel displaying a chronological 80/20 test split confusion matrix, precision/recall/F1 metrics, simulated strategy return, and feature importance weights.

---

## ⚠️ Disclaimer

This software is intended for educational, research, and analytical purposes only. Foreign exchange trading carries a high level of risk, and machine learning predictions based on historical price action do not guarantee future performance.
