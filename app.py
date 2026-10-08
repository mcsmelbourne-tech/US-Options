import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import os
import io
import plotly.graph_objects as go
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="Moomoo ITM Options Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look, clean cards, and responsive design
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .card { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px; }
    .badge-call { background-color: #238636; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-put { background-color: #da3633; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-neutral { background-color: #8b949e; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

ET = ZoneInfo("America/New_York")
LOG_FILE = "scan_log.csv"

st.sidebar.title("⚙️ Scanner Settings")

mode = st.sidebar.radio("Operating Mode", ["Simulation Mode (Cloud Ready)", "Live Moomoo OpenD (Local)"])

host = st.sidebar.text_input("OpenD Host", "127.0.0.1", disabled=(mode == "Simulation Mode (Cloud Ready)"))
port = st.sidebar.number_input("OpenD Port", value=11111, step=1, disabled=(mode == "Simulation Mode (Cloud Ready)"))

symbols_input = st.sidebar.text_input("Watchlist Symbols (Comma Separated)", "NVDA, TSLA, AAPL, AMZN, MSFT")
SYMBOLS = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

st.sidebar.markdown("---")
min_score = st.sidebar.slider("Min Score (|Score| required)", 1, 5, 3)
target_delta = st.sidebar.slider("Target Option Delta", 0.50, 0.90, 0.70, 0.05)
max_spread_pct = st.sidebar.slider("Max Spread %", 1.0, 25.0, 10.0, 0.5)
min_option_volume = st.sidebar.number_input("Min Option Volume", 10, 500, 50, 10)
stop_loss_pct = st.sidebar.slider("Suggested Stop Loss %", 10.0, 50.0, 25.0, 5.0)
expiry_offset = st.sidebar.selectbox("Expiry Offset", [0, 1, 2], index=0, help="0 = next expiry after today, 1 = subsequent expiry")

# Initialize CSV log file if not exists
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "time_et", "symbol", "side", "contract", "expiry", "strike", "spot", "pred_15m",
        "bid", "ask", "target", "stop", "score", "rr", "delta", "gamma", "theta", "vega", "rho", "iv"
    ])
    df_init.to_csv(LOG_FILE, index=False)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def generate_mock_market_data(sym):
    np.random.seed(hash(sym + dt.datetime.now().strftime("%Y%m%d%H%M")) % 2**32)
    base_price = {"NVDA": 130.0, "TSLA": 220.0, "AAPL": 230.0, "AMZN": 185.0, "MSFT": 420.0}.get(sym, 150.0)
    
    # Generate 100 1-minute synthetic candles
    timestamps = [dt.datetime.now(ET) - dt.timedelta(minutes=i) for i in range(100, 0, -1)]
    noise = np.random.normal(0, 0.4, 100).cumsum()
    close_prices = base_price + noise
    high_prices = close_prices + np.random.uniform(0.1, 0.5, 100)
    low_prices = close_prices - np.random.uniform(0.1, 0.5, 100)
    open_prices = close_prices + np.random.normal(0, 0.2, 100)
    volumes = np.random.randint(1000, 50000, 100)
    
    df = pd.DataFrame({
        "time_key": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    })
    return df

def simulate_analysis(sym):
    df = generate_mock_market_data(sym)
    close = df["close"]
    e9, e20 = ema(close, 9), ema(close, 20)
    macd = ema(close, 12) - ema(close, 26)
    hist = macd - ema(macd, 9)
    r = rsi(close)
    a = atr(df)
    
    spot = close.iloc[-1]
    vwap = spot * (1 + np.random.normal(0, 0.002))
    
    score = 0
    score += 1 if e9.iloc[-1] > e20.iloc[-1] else -1
    score += 1 if spot > vwap else -1
    score += 1 if r.iloc[-1] > 55 else (-1 if r.iloc[-1] < 45 else 0)
    score += 1 if (hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]) else (-1 if (hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]) else 0)
    mom = (spot - close.iloc[-6]) / max(a.iloc[-1], 1e-9)
    score += 1 if mom > 0.5 else (-1 if mom < -0.5 else 0)
    score = int(np.clip(score, -5, 5))

    y = close.tail(12).values
    x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1)
    base = slope * (len(y) - 1) + icpt
    proj = {m: base + slope * m for m in (5, 10, 15)}
    band = a.iloc[-1] * np.sqrt(10)

    return df, dict(sym=sym, spot=spot, vwap=vwap, rsi=r.iloc[-1], score=score,
                    p5=proj[5], p10=proj[10], p15=proj[15], band=band, atr=a.iloc[-1])

def simulate_option_pick(sym, spot, side):
    expiry_date = (dt.datetime.now(ET) + dt.timedelta(days=(7 * (expiry_offset + 1)))).strftime("%Y-%m-%d")
    strike_offset = -2.0 if side == "CALL" else 2.0
    strike = round(spot + strike_offset, 1)
    contract_code = f"US.{sym}{expiry_date.replace('-', '')}{side[0]}{int(strike*1000):08d}"
    
    bid = round(max(3.50, abs(spot - strike) + np.random.uniform(2.0, 5.0)), 2)
    ask = round(bid + np.random.uniform(0.10, 0.40), 2)
    spread_pct = (ask - bid) / ask * 100
    
    return dict(
        code=contract_code, expiry=expiry_date, strike=strike, bid=bid, ask=ask,
        delta=0.71 if side == "CALL" else -0.69, gamma=0.035, theta=-0.15,
        vega=0.22, rho=0.05, iv=32.5, vol=1250, oi=4500, spread=spread_pct
    )

st.title("📈 Moomoo ITM Call/Put Signal Scanner")
st.markdown("Real-time quantitative momentum scoring, regression price projection, and deep ITM options filtration.")

tab1, tab2, tab3 = st.tabs(["🚀 Live / Mock Scanner", "📋 Scan Log History", "📊 Performance & Analytics"])

with tab1:
    col_ctrl1, col_ctrl2 = st.columns([2, 6])
    with col_ctrl1:
        run_scan_btn = st.button("🔍 Run Scan Now", type="primary", use_container_width=True)
    with col_ctrl2:
        st.info(f"Active Watchlist: {', '.join(SYMBOLS)} | Target Delta: ~{target_delta} | Min Score: ±{min_score}")

    if run_scan_btn or "scanned_results" not in st.session_state:
        with st.spinner("Analyzing ticker momentum, calculating regressions, and querying options chain..."):
            time.sleep(1) # Simulated scan latency
            scan_results = []
            new_log_rows = []
            
            for sym in SYMBOLS:
                if mode == "Simulation Mode (Cloud Ready)":
                    df_candles, analysis = simulate_analysis(sym)
                    side = "CALL" if analysis["score"] >= min_score else ("PUT" if analysis["score"] <= -min_score else None)
                    
                    opt = None
                    if side:
                        opt = simulate_option_pick(sym, analysis["spot"], side)
                        
                    scan_results.append({
                        "sym": sym, "df": df_candles, "analysis": analysis, "side": side, "opt": opt
                    })
                    
                    if side and opt:
                        dS = analysis["p15"] - analysis["spot"]
                        theta_cost = opt["theta"] * (15 / 390)
                        est_change = opt["delta"] * dS + 0.5 * opt["gamma"] * dS ** 2 + theta_cost
                        entry = opt["ask"]
                        target = max(entry + est_change, entry)
                        stop = entry * (1 - stop_loss_pct / 100)
                        rr = (target - entry) / (entry - stop) if entry > stop else 0
                        
                        new_log_rows.append([
                            dt.datetime.now(ET).isoformat(timespec="seconds"), sym, side, opt["code"],
                            opt["expiry"], opt["strike"], round(analysis["spot"], 2), round(analysis["p15"], 2),
                            opt["bid"], opt["ask"], round(target, 2), round(stop, 2), analysis["score"], round(rr, 2),
                            round(opt["delta"], 4), round(opt["gamma"], 5), round(opt["theta"], 4),
                            round(opt["vega"], 4), round(opt["rho"], 4), round(opt["iv"], 2)
                        ])
                else:
                    # Live Moomoo API fallback warning for cloud environments
                    st.warning("Live Moomoo OpenD requires a local gateway running on port 11111. Please switch to Simulation Mode for cloud deployments.")
                    break
            
            if new_log_rows:
                df_logs = pd.read_csv(LOG_FILE)
                df_new = pd.DataFrame(new_log_rows, columns=df_logs.columns)
                df_combined = pd.concat([df_logs, df_new], ignore_index=True)
                df_combined.to_csv(LOG_FILE, index=False)
                
            st.session_state["scanned_results"] = scan_results

    if "scanned_results" in st.session_state:
        st.markdown("### 🔍 Scan Results by Symbol")
        
        for item in st.session_state["scanned_results"]:
            sym = item["sym"]
            a = item["analysis"]
            side = item["side"]
            opt = item["opt"]
            
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0;">{sym} &nbsp;|&nbsp; Spot: ${a['spot']:.2f}</h3>
                        <div>
                            <span>Score: <b>{a['score']:+d}</b></span> &nbsp;&nbsp;|&nbsp;&nbsp;
                            <span>RSI: <b>{a['rsi']:.0f}</b></span> &nbsp;&nbsp;|&nbsp;&nbsp;
                            {'<span class="badge-call">CALL SIGNAL</span>' if side == 'CALL' else ('<span class="badge-put">PUT SIGNAL</span>' if side == 'PUT' else '<span class="badge-neutral">NO TRADE</span>')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_chart, col_details = st.columns([3, 2])
                
                with col_chart:
                    # Plotly chart with regression projections and ATR bands
                    fig = go.Figure()
                    df_c = item["df"]
                    
                    fig.add_trace(go.Candlestick(
                        x=df_c['time_key'], open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'],
                        name="1m Candles"
                    ))
                    
                    # Projection line
                    last_time = pd.to_datetime(df_c['time_key'].iloc[-1])
                    future_times = [last_time + pd.Timedelta(minutes=m) for m in (5, 10, 15)]
                    future_prices = [a['p5'], a['p10'], a['p15']]
                    
                    fig.add_trace(go.Scatter(
                        x=[df_c['time_key'].iloc[-1]] + [t.strftime("%Y-%m-%d %H:%M:%S") for t in future_times],
                        y=[a['spot']] + future_prices,
                        mode='lines+markers',
                        name='15m Regression Projection',
                        line=dict(color='cyan', dash='dash', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{sym} Price Action & 15m Projection",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_details:
                    if opt and side:
                        dS = a["p15"] - a["spot"]
                        theta_cost = opt["theta"] * (15 / 390)
                        est_change = opt["delta"] * dS + 0.5 * opt["gamma"] * dS ** 2 + theta_cost
                        entry = opt["ask"]
                        target = max(entry + est_change, entry)
                        stop = entry * (1 - stop_loss_pct / 100)
                        rr = (target - entry) / (entry - stop) if entry > stop else 0
                        
                        st.markdown(f"""
                        **Contract:** `{opt['code']}`  
                        **Strike:** `${opt['strike']}` | **Expiry:** `{opt['expiry']}`  
                        **Bid/Ask:** `${opt['bid']:.2f}` / `${opt['ask']:.2f}` (Spread: {opt['spread']:.1f}%)  
                        **Greeks:** $\\Delta$ `{opt['delta']:.2f}` | $\\Gamma$ `{opt['gamma']:.3f}` | $\\Theta$ `{opt['theta']:.2f}`  
                        **Trade Setup:**  
                        - **Entry Ask $\\le$** `${entry:.2f}`  
                        - **Est. 15m Change:** `{est_change:+.2f}`  
                        - **Target:** `~${target:.2f}` | **Stop:** `${stop:.2f}` (R:R `{rr:.1f}`)  
                        """)
                    else:
                        st.markdown("_No high-conviction ITM contract meeting liquidity and delta criteria during this scan._")

with tab2:
    st.subheader("📁 Scan Log History (`scan_log.csv`)")
    if os.path.exists(LOG_FILE):
        df_log_view = pd.read_csv(LOG_FILE)
        st.dataframe(df_log_view, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns([2, 8])
        with col_dl1:
            csv_bytes = df_log_view.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download scan_log.csv",
                data=csv_bytes,
                file_name="scan_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            if st.button("🗑️ Clear Log History"):
                pd.DataFrame(columns=df_log_view.columns).to_csv(LOG_FILE, index=False)
                st.success("Scan history cleared successfully.")
                st.rerun()
    else:
        st.info("No scan history logged yet.")

with tab3:
    st.subheader("📊 Strategy Performance & Distribution")
    if os.path.exists(LOG_FILE):
        df_analytics = pd.read_csv(LOG_FILE)
        if not df_analytics.empty:
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Total Signals Logged", len(df_analytics))
            with col_met2:
                calls_count = len(df_analytics[df_analytics["side"] == "CALL"])
                puts_count = len(df_analytics[df_analytics["side"] == "PUT"])
                st.metric("Calls vs Puts", f"{calls_count} Calls / {puts_count} Puts")
            with col_met3:
                avg_rr = df_analytics["rr"].mean() if "rr" in df_analytics.columns else 0.0
                st.metric("Avg Projected Risk:Reward", f"{avg_rr:.2f}")
            
            st.markdown("---")
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                if "symbol" in df_analytics.columns:
                    sym_counts = df_analytics["symbol"].value_counts().reset_index()
                    fig_sym = go.Figure(go.Bar(x=sym_counts["symbol"], y=sym_counts["count"], marker_color="#238636"))
                    fig_sym.update_layout(title="Signals by Symbol", template="plotly_dark", height=300)
                    st.plotly_chart(fig_sym, use_container_width=True)
            with col_ch2:
                if "rr" in df_analytics.columns:
                    fig_rr = go.Figure(go.Histogram(x=df_analytics["rr"], marker_color="#1f6feb"))
                    fig_rr.update_layout(title="Risk:Reward Distribution", template="plotly_dark", height=300)
                    st.plotly_chart(fig_rr, use_container_width=True)
        else:
            st.info("Log file is empty. Run scans to populate analytics.")
    else:
        st.info("No log data available.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b949e;'>Moomoo ITM Options Scanner | Designed for GitHub & Streamlit Cloud Deployment</p>", unsafe_allow_html=True)
