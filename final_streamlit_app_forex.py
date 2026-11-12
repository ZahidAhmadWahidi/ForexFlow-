
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET




# 1. STREAMLIT FRONTPAGE LAYOUT CONFIGURATION
st.set_page_config(
    page_title="ForexFlow Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# New Main Body Branding Title
st.title("🌊 ForexFlow Analytics Dashboard")
st.caption("Systematic Foreign Exchange Machine Learning Inference Terminal")


# STRICT STYLE INTERFACE MATCH (White content cards, crisp grays, and modern typography)
st.markdown("""
    <style>
        /* Base page canvas background setup */
        .main { background-color: #F8FAFC !important; color: #1E293B !important; }
        
        /* Modernized Sidebar panel styling */
        div[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
        div[data-testid="stSidebar"] h1 { color: #0F172A !important; font-size: 1.5rem !important; font-weight: 700 !important; }
        
        /* Premium Card Containers styling */
        .quant-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
        }
        
        /* Top Row KPI Typography rules */
        .kpi-title { font-size: 0.8rem !important; font-weight: 600 !important; color: #64748B !important; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.05em; }
        .kpi-value { font-size: 2.2rem !important; font-weight: 700 !important; color: #0284C7 !important; margin: 0; line-height: 1.1; }
        .kpi-sub { font-size: 0.85rem !important; font-weight: 600 !important; margin: 4px 0 0 0; }
        
        /* Section Titles rule */
        .section-title { font-size: 1.2rem !important; font-weight: 700 !important; color: #0F172A !important; margin-bottom: 15px !important; display: flex; align-items: center; gap: 8px; }
        
        /* Signal Dashboard alert badge */
        .signal-badge {
            padding: 15px;
            border-radius: 6px;
            font-size: 1.1rem;
            font-weight: 700;
            text-align: center;
            letter-spacing: 0.02em;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. AUTOMATED IN-MEMORY DATA INGESTION (720 DAYS LOOKBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def load_and_process_data():
    # Added "GC=F" for Gold
    currency_pairs = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCAD=X", "AUDUSD=X", "GC=F"]
    processed_chunks = []
    
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-9))))

    for pair in currency_pairs:
        df = yf.download(pair, period="720d", interval="1h", auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip() for col in df.columns]
            
        df = df.copy()
        # Renames GC=F to XAUUSD so it displays cleanly as XAU/USD
        df["Ticker"] = "XAUUSD" if pair == "GC=F" else pair.replace("=X", "")
        df.index = df.index.tz_convert('Europe/Berlin')
        
        # Pipeline Feature Transformations
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Lag_1'] = df['Log_Return'].shift(1)
        df['Lag_2'] = df['Log_Return'].shift(2)
        df['RSI_14'] = calculate_rsi(df['Close'])
        df['Hour'] = df.index.hour
        
        # Added Volatility & Moving Average Distance
        df['Volatility_14'] = df['Log_Return'].rolling(window=14).std()
        sma_24 = df['Close'].rolling(window=24).mean()
        df['SMA_24_Dist'] = (df['Close'] - sma_24) / sma_24
        
        df['Target'] = np.where(df['Log_Return'].shift(-1) > 0, 1, 0)
        processed_chunks.append(df)
        
    final_df = pd.concat(processed_chunks)
    final_df.dropna(subset=['Lag_1', 'Lag_2', 'RSI_14', 'Hour', 'Volatility_14', 'SMA_24_Dist', 'Target'], inplace=True)
    return final_df





    
    
# =====================================================================
# 2. AUTOMATED IN-MEMORY DATA INGESTION & OPEN MACRO FEED PARSER
# =====================================================================
@st.cache_data(ttl=600)
def fetch_macro_events():
    """Parses structural open-access financial wires dynamically for timeline monitoring"""
    try:
        # Utilize the open whitelist network feed node to bypass layout firewalls
        url = "https://www.myfxbook.com/rss/latest-forex-news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/xml"
        }
        res = requests.get(url, headers=headers, timeout=10)
        
        # Parse XML structure tree nodes natively
        root = ET.fromstring(res.content)
        events = list()
        
        # Ingest the latest 5 live breaking headline elements safely
        for item in root.findall('.//item')[0:5]:
            title_el = item.find('title')
            if title_el is not None and title_el.text:
                scrubbed_title = str(title_el.text).strip()
                events.append(f"⚡ LIVE FEED: {scrubbed_title}")
                
        if not events:
            return ["⚡ FXFLOW: Monitoring G10 Central Bank Monetary Gaps", "⚡ FXFLOW: Tracking Interbank Spot Liquidity Vectors"]
        return events
    except Exception:
        # Academic-grade institutional fallback frames while pipeline logs refresh
        return [
            "⚡ FXFLOW: Sourcing Live Global Market Wire...", 
            "⚡ SYSTEM: Analyzing Macro Aggregation Vectors...",
            "⚡ FEED: Tracking Central Bank Hawkish Pauses..."
        ]

# Run the updated macro scraper engine
macro_feed = fetch_macro_events()







with st.spinner("⏳ Loading ForexFlow configurations..."):
    master_df = load_and_process_data()
# =====================================================================
# 3. INTERACTIVE SIDEBAR CONTROL MODULE (HOURLY FOCUS)
# =====================================================================
st.sidebar.markdown("# ⚙️ForexFlow Control Panel")
st.sidebar.markdown("---")

# 1. Core Asset Vector
ticker_options = master_df["Ticker"].unique()
formatted_pairs = [f"{t[0:3]}/{t[3:]}" for t in ticker_options]
selected_pair_display = st.sidebar.selectbox("Select Currency Pair", formatted_pairs)
selected_ticker = selected_pair_display.replace("/", "")

# 2. Historical Horizon Segment Filter rules
st.sidebar.markdown("### 📅 Temporal Horizon Focus")

# Year Filter
available_years = sorted(list(set(master_df.index.year)), reverse=True)
selected_year = st.sidebar.selectbox("Anchor Year", available_years)

# Month Names Mapping Array (Index 0 is empty padding so index matches month number 1-12)
month_names = [
    "Padding", "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

# Extract available integer months present in dataset
available_months_int = sorted(list(set(master_df.index.month)))

# Map integers to their corresponding text names for display selection
available_months_names = [month_names[m] for m in available_months_int]

# Render the sidebar selector using clear string names
selected_month_name = st.sidebar.selectbox(
    "Anchor Month", 
    available_months_names, 
    index=len(available_months_names) - 1
)

# Convert the text string back to an integer for pandas dataframe filtering
selected_month = month_names.index(selected_month_name)
st.sidebar.markdown("### ⏱️ Bar Lookback Density")
hourly_lookback = st.sidebar.slider("Display Window (Hourly Candles)", 6, 100, 6)



# Isolate base series matrix match
df_pair = master_df[master_df["Ticker"] == selected_ticker].copy().sort_index()

# =====================================================================
# 4. SUPERVISED ML COMPUTATIONAL ROUTINES
# =====================================================================
features = ['Lag_1', 'Lag_2', 'RSI_14', 'Hour', 'Volatility_14', 'SMA_24_Dist']

X = df_pair[features]
y = df_pair['Target']

split_idx = int(len(df_pair) * 0.8)
X_train = X.iloc[0:split_idx]
y_train = y.iloc[0:split_idx]


# REPLACE IT WITH THIS (Check spelling: class_weight="balanced"):
model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=5, 
    class_weight="balanced", 
    random_state=42, 
    n_jobs=-1
)

model.fit(X_train, y_train)

latest_row = df_pair.tail(1)
probs = model.predict_proba(latest_row[features]).flatten()

prob_down = float(probs.item(0))
prob_up = float(probs.item(1))

if prob_up > prob_down:
    action_text = "BUY (LONG)"
    signal_confidence = prob_up
    alert_box_style = "background-color: #ECFDF5; border: 1px solid #10B981; color: #047857;"
else:
    action_text = "SELL (SHORT)"
    signal_confidence = prob_down
    alert_box_style = "background-color: #FEF2F2; border: 1px solid #EF4444; color: #B91C1C;"

# Raw metric calculations safely isolated using the true historical max/min bounds of the chart window
close_val = float(latest_row['Close'].to_numpy().item())
open_val = float(latest_row['Open'].to_numpy().item())
net_change = close_val - open_val
pct_change = (net_change / open_val) * 100

# FIX: Derive the session ceiling and floor from the active chart dataframe instead of just the latest row
chart_window_df = df_pair.tail(hourly_lookback)
high_val = float(chart_window_df['High'].max())
low_val = float(chart_window_df['Low'].min())

# Dynamic rolling 24-period hourly market volatility calculation
rolling_vol_series = df_pair['Log_Return'].rolling(window=24).std()
vol_val = float(rolling_vol_series.tail(1).to_numpy().item()) * 100

# =====================================================================
# 5. BUSINESS INTELLIGENCE MATRICES (KPI ROW)
# =====================================================================
# Sets 2 decimal places for Gold (XAUUSD) and 4 for Forex pairs
dec = 2 if selected_ticker == "XAUUSD" else 4

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
        <div class="quant-card">
            <p class="kpi-title">{selected_pair_display} Last Price</p>
            <p class="kpi-value" style="color: #0284C7 !important;">{close_val:.{dec}f}</p>
            <p class="kpi-sub" style="color: {'#10B981' if net_change >= 0 else '#EF4444'};">
                {'↑' if net_change >= 0 else '↓'} {net_change:+.{dec}f} ({pct_change:+.2f}%)
            </p>
        </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
        <div class="quant-card">
            <p class="kpi-title">Session High</p>
            <p class="kpi-value" style="color: #0F172A !important;">{high_val:.{dec}f}</p>
            <p class="kpi-sub" style="color: #64748B;">Intraday Ceiling</p>
        </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
        <div class="quant-card">
            <p class="kpi-title">Session Low</p>
            <p class="kpi-value" style="color: #0F172A !important;">{low_val:.{dec}f}</p>
            <p class="kpi-sub" style="color: #64748B;">Intraday Floor</p>
        </div>
    """, unsafe_allow_html=True)

with k4:
    # Pull the 14-hour rolling volatility and convert to a clean percentage
    vol_pct = float(latest_row['Volatility_14'].iloc[0]) * 100
    st.markdown(f"""
        <div class="quant-card">
            <p class="kpi-title">14H Volatility</p>
            <p class="kpi-value" style="color: #10B981 !important;">{vol_pct:.2f}%</p>
            <p class="kpi-sub" style="color: #64748B;">Rolling Hourly Risk</p>
        </div>
    """, unsafe_allow_html=True)






# =====================================================================
# SYSTEM FLOATING MACRO FEED TICKER
# =====================================================================
macro_text_string = " &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;||&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ".join(macro_feed)

st.markdown(f"""
    <div style="background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%); padding: 10px 20px; border-radius: 8px; margin-bottom: 25px; overflow: hidden; display: flex; align-items: center; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);">
        <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; margin-right: 15px; letter-spacing: 0.05em; display: inline-block; white-space: nowrap; box-shadow: 0 0 10px rgba(239,68,68,0.5);">HIGH IMPACT</span>
        <marquee scrollamount="4" style="color: #E2E8F0; font-size: 0.85rem; font-weight: 600; font-family: monospace;">
            {macro_text_string}
        </marquee>
    </div>
""", unsafe_allow_html=True)


# =====================================================================
# 6. LOWER PANEL: MAXIMIZED CHART CORE & TARGET RISK BRACKETS
# =====================================================================

# --- STEP A: COMPUTE ATR VOLATILITY EXIT TARGETS FIRST ---
high_low = df_pair['High'] - df_pair['Low']
high_close_prev = (df_pair['High'] - df_pair['Close'].shift(1)).abs()
low_close_prev = (df_pair['Low'] - df_pair['Close'].shift(1)).abs()

true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
atr_baseline = float(true_range.rolling(window=14).mean().tail(1).to_numpy().item())

if prob_up > prob_down:
    # BUY SETUP: Take Profit sets above, Stop Loss sets below entry
    target_tp = close_val + (atr_baseline * 1.5)
    target_sl = close_val - (atr_baseline * 1.0)
    tp_label, sl_label = "TAKE PROFIT (TARGET HIGH)", "STOP LOSS (RISK CEILING)"
    tp_color, sl_color = "#10B981", "#EF4444"
else:
    # SELL SETUP: Take Profit sets below, Stop Loss sets above entry
    target_tp = close_val - (atr_baseline * 1.5)
    target_sl = close_val + (atr_baseline * 1.0)
    tp_label, sl_label = "TAKE PROFIT (TARGET LOW)", "STOP LOSS (RISK FLOOR)"
    tp_color, sl_color = "#10B981", "#EF4444"

# --- STEP B: WIDE CANDLESTICK CHART WITH VISUAL EXIT OVERLAYS ---
st.markdown('<p class="section-title">📈 Real-Time Price Matrix Chart & Execution Brackets</p>', unsafe_allow_html=True)

filtered_chart_df = df_pair[
    (df_pair.index.year == selected_year) & 
    (df_pair.index.month <= selected_month)
]

if len(filtered_chart_df) == 0:
    filtered_chart_df = df_pair.copy()
    
chart_df = filtered_chart_df.tail(hourly_lookback)

fig = go.Figure(data=list([go.Candlestick(
    x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
    low=chart_df['Low'], close=chart_df['Close'],
    increasing_line_color='#10B981', decreasing_line_color='#EF4444',
    increasing_fillcolor='#10B981', decreasing_fillcolor='#EF4444'
)]))

# Draw Take-Profit dashed green line across the chart
fig.add_hline(
    y=target_tp,
    line_dash="dash",
    line_color="#10B981",
    line_width=2,
    annotation_text=f"🎯 TP: {target_tp:.4f}",
    annotation_position="top left",
    annotation_font=dict(color="#047857", size=12, weight="bold")
)

# Draw Stop-Loss dashed red line across the chart
fig.add_hline(
    y=target_sl,
    line_dash="dash",
    line_color="#EF4444",
    line_width=2,
    annotation_text=f"🛡️ SL: {target_sl:.4f}",
    annotation_position="bottom left",
    annotation_font=dict(color="#B91C1C", size=12, weight="bold")
)

# Expand Y-axis bounds slightly so TP/SL lines are never cut off the screen
chart_min = min(float(chart_df['Low'].min()), target_sl) * 0.9995
chart_max = max(float(chart_df['High'].max()), target_tp) * 1.0005

fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_rangeslider_visible=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#FFFFFF',
    height=460,
    yaxis=dict(
        gridcolor='#F1F5F9', 
        tickformat='.4f', 
        color='#64748B', 
        side='right',
        range=[chart_min, chart_max]
    ),
    xaxis=dict(gridcolor='#F1F5F9', color='#64748B')
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# --- STEP C: PREDICTIVE SIGNAL CORE & TARGET EXECUTION CARDS ---
st.markdown('<p class="section-title">🤖 Machine Learning Analytics & Target Risk Core</p>', unsafe_allow_html=True)

c_sig, c_proj, c_exit = st.columns([1, 1, 1.5])

with c_sig:
    st.markdown(f"""
        <div class="signal-badge" style="{alert_box_style} padding: 25px; border-radius: 12px; height: 160px;">
            <span style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; opacity: 0.8;">Direction Verdict</span>
            <div style="font-size: 1.45rem; margin-top: 4px; font-weight: 800;">{action_text}</div>
            <div style="font-size: 0.9rem; font-weight: 600; margin-top: 6px; opacity: 0.9;">Confidence: {signal_confidence * 100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with c_proj:
    st.markdown(f"""
        <div class="quant-card" style="border-radius: 12px; padding: 22px; height: 160px;">
            <p style="margin: 0 0 10px 0; font-size: 0.75rem; color: #64748B; font-weight: 700; letter-spacing: 0.05em;">PROJECTIONS</p>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 6px;">
                    <span style="color: #10B981; font-weight: 700; font-size: 0.85rem;">🟢 Bullish (Up)</span>
                    <span style="font-weight: 800; color: #0F172A;">{prob_up * 100:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-top: 4px;">
                    <span style="color: #EF4444; font-weight: 700; font-size: 0.85rem;">🔴 Bearish (Down)</span>
                    <span style="font-weight: 800; color: #0F172A;">{prob_down * 100:.1f}%</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with c_exit:
    st.markdown(f"""
        <div class="quant-card" style="border-radius: 12px; padding: 20px; height: 160px;">
            <p style="margin: 0 0 12px 0; font-size: 0.75rem; color: #64748B; font-weight: 700; letter-spacing: 0.05em;">VOLATILITY-ADJUSTED EXIT TARGETS</p>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F5F9; padding-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 0.85rem; color: {tp_color};">🎯 {tp_label}</span>
                    <span style="font-weight: 800; font-family: monospace; font-size: 1.05rem; color: #0F172A;">{target_tp:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 2px;">
                    <span style="font-weight: 700; font-size: 0.85rem; color: {sl_color};">🛡️ {sl_label}</span>
                    <span style="font-weight: 800; font-family: monospace; font-size: 1.05rem; color: #0F172A;">{target_sl:.4f}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)




# =====================================================================
# 7. ACADEMIC THESIS DEFENSE: MODEL VALIDATION & INTERPRETABILITY
# =====================================================================
with st.expander("View Complete Available Market Processes (Raw Data Stream)"):
    st.dataframe(df_pair[['Open', 'High', 'Low', 'Close', 'RSI_14', 'Volatility_14', 'SMA_24_Dist', 'Hour']].tail(10), use_container_width=True)

    
    # 1. Isolate chronological 20% out-of-sample test set
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    test_preds = model.predict(X_test)
    
    # 2. Compute core classification metrics safely without external dependencies
    tp = int(((test_preds == 1) & (y_test == 1)).sum())
    tn = int(((test_preds == 0) & (y_test == 0)).sum())
    fp = int(((test_preds == 1) & (y_test == 0)).sum())
    fn = int(((test_preds == 0) & (y_test == 1)).sum())
    
    total_samples = tp + tn + fp + fn
    acc_score = ((tp + tn) / max(total_samples, 1)) * 100
    precision_score = (tp / max((tp + fp), 1)) * 100
    recall_score = (tp / max((tp + fn), 1)) * 100
    f1_score = (2 * precision_score * recall_score) / max((precision_score + recall_score), 1e-9)
    
    # 3. Calculate simulated out-of-sample strategy return vs Buy & Hold
    test_returns = df_pair['Log_Return'].iloc[split_idx:].shift(-1).fillna(0)
    signals = np.where(test_preds == 1, 1, -1)
    strat_cum_return = (np.exp((signals * test_returns).sum()) - 1) * 100
    
    # Top Metric Row inside Expander
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Out-of-Sample Accuracy", f"{acc_score:.2f}%", f"{total_samples:,} Test Bars")
    v2.metric("Bullish Precision", f"{precision_score:.2f}%")
    v3.metric("Model F1-Score", f"{f1_score:.2f}%")
    v4.metric("Simulated Strategy Return", f"{strat_cum_return:+.2f}%", "20% Test Split")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Lower Split: Left = Confusion Matrix, Right = Feature Importance Chart
    col_cm, col_fi = st.columns(2)
    
    with col_cm:
        st.markdown('<p class="section-title">🧮 Out-of-Sample Confusion Matrix</p>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="quant-card" style="padding: 20px; border-radius: 12px;">
                <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 0.9rem;">
                    <tr style="border-bottom: 2px solid #E2E8F0; color: #64748B; font-weight: 700;">
                        <td style="padding: 10px; text-align: left;">ACTUAL \\ PREDICTED</td>
                        <td style="padding: 10px; color: #10B981;">PREDICTED UP (1)</td>
                        <td style="padding: 10px; color: #EF4444;">PREDICTED DOWN (0)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 10px; text-align: left; font-weight: 700; color: #0F172A;">ACTUAL UP (1)</td>
                        <td style="padding: 14px; background-color: #ECFDF5; color: #047857; font-weight: 800; border-radius: 6px;">{tp} (True Pos)</td>
                        <td style="padding: 14px; color: #64748B; font-weight: 600;">{fn} (False Neg)</td>
                    </tr>
                    <tr>
                        <td style="padding: 14px 10px; text-align: left; font-weight: 700; color: #0F172A;">ACTUAL DOWN (0)</td>
                        <td style="padding: 14px; color: #64748B; font-weight: 600;">{fp} (False Pos)</td>
                        <td style="padding: 14px; background-color: #FEF2F2; color: #B91C1C; font-weight: 800; border-radius: 6px;">{tn} (True Neg)</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        
    with col_fi:
        st.markdown('<p class="section-title">📊 Random Forest Feature Weights</p>', unsafe_allow_html=True)
        
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances * 100
        }).sort_values(by="Importance", ascending=True)
        
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation='h',
            marker=dict(color='#0284C7', line=dict(color='#0369A1', width=1)),
            text=[f"{val:.1f}%" for val in fi_df["Importance"]],
            textposition='auto'
        ))
        
        fig_fi.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=215,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FFFFFF',
            xaxis=dict(title="Relative Weight (%)", gridcolor='#F1F5F9', color='#64748B'),
            yaxis=dict(color='#0F172A', tickfont=dict(weight='bold'))
        )
        
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})


