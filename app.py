"""
US Top-100 Call/Put Scanner - scheduled scans every US trading day (yfinance + Greeks + price action)
                              + SELF-IMPROVING: every candidate is tracked and its result fetched after the trade
                                day and after expiry; the scoring weights learn from those results.

  Scan 1  OPEN 09:45 ET   - first scan of the day (premarket range/gap/volume + the first 15 min of trading)
  Scan 2+ HOURLY          10:45, 11:45, 12:45, 13:45, 14:45 ET - fresh picks through the day (volume is judged
                          against the time of day; 14:45 is the last, trades exit by 15:45 ET)

Page layout: best CALL and best PUT (tabs) on TOP, then two list boxes - US INDICES and COMMODITIES -
             then the TOP 10 US SHARES.

Per stock: EMA 9 / EMA 15, RSI, MACD, volume (relative to normal), ATR, VWAP, gap
           + PRICE ACTION: swing structure (HH/HL, LH/LL), break of structure (BOS), candle patterns
             (engulfing, hammer, shooting star, strong bar), previous-day high/low breaks, and a
             "blocked" check when the next support/resistance is too close for the expected move.
             The stop loss is placed beyond the last swing low (call) / swing high (put).
           + ACCURACY FILTERS: the SPY (market) direction must not fight the trade, minimum relative volume,
             no earnings before the option expires, tighter option liquidity (spread / open interest).
Per option: Black-Scholes Greeks (delta, gamma, theta, vega, rho) from yfinance implied vol,
            next expiry AFTER today, ranked by Greeks + spread + liquidity + IV vs historical vol.
Output:     best CALL and best PUT in a box (+ one TOP PICK), predicted day price (close/high/low),
            entry, stop loss, take profit, profit for 1 lot. Compact Telegram alert for each scan.

SELF-IMPROVEMENT (the "Learning" tab):
  1. Every candidate contract of every live scan (not only the 2 best) is saved to tracked_trades.json.
  2. After the trade day ends (16:10 ET) the 5-minute bars are fetched and the plan is replayed bar by bar
     (stop loss / take profit / 15:45 exit) -> "plan result".
  3. After the option EXPIRES, the stock's closing price on the expiry date is fetched and the exact expiry value
     is computed (intrinsic value) -> "expiry result" (what holding to expiry would have paid).
  4. After each settlement the weights of the individual signals (EMA, VWAP, RSI, MACD, BOS, candles ...) are
     re-learned from the plan results (learned.json). A signal that keeps losing gets a lower weight, one that
     keeps winning a higher weight. Needs 25+ settled trades per signal before it changes anything.
  Honest limits: yfinance keeps no option quote history, so the plan result is the option MODELLED with Black-Scholes
  from the logged entry price and IV (the expiry result is exact). The sample is small at first, the weights move
  slowly on purpose, and no scanner can guarantee profit.

BACKTEST:   "Backtest" tab (or command line) replays the same scans on the last N trading days of 5-minute
            data, prices the option with Black-Scholes, walks forward bar by bar to the SL / TP / 15:45 exit,
            and compares results WITH and WITHOUT price action. (The backtest never uses the learned weights.)

Run:   streamlit run us_itm_scanner.py        (leave the page open - it fires the scans on time)
       python us_itm_scanner.py --settle      (fetch results of finished trades + re-learn, no page needed)
       python us_itm_scanner.py --backtest 20 (backtest from the command line, 20 trading days)
       python us_itm_scanner.py --sweep 40    (settings sweep, train on early days / test on later days)
Needs: pip install streamlit yfinance pandas numpy requests
Educational use only - estimates are heuristics, not advice. No scanner can guarantee profit.
"""

import csv
import itertools
import os
import sys
import json
import math
import time
import datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

# ----------------------------- SETTINGS -----------------------------
ET = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("Australia/Sydney")      # only used to show your local time

LOG_FILE = "scan_log.csv"
CONFIG_FILE = "config.json"
STATE_FILE = "scan_state.json"
BT_FILE = "backtest_trades.csv"
SWEEP_FILE = "backtest_sweep.csv"
TRACK_FILE = "tracked_trades.json"           # every live candidate + its settled results
LEARN_FILE = "learned.json"                  # learned signal weights + statistics

SCAN_PRE_TIME = dt.time(9, 0)                # only used by the manual "Premarket scan now" button
SCAN_OPEN_TIME = dt.time(9, 45)              # first scheduled scan (ET)
SCAN_OPEN_LATEST = dt.time(10, 30)           # if the app starts late, still run the 09:45 scan until here
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
EXIT_TIME = dt.time(15, 45)                  # trades are closed by this time
HOURLY_TIMES = [dt.time(h, 45) for h in range(10, 15)]   # 10:45 ... 14:45 ET, every hour after the first scan
HOURLY_GRACE_MIN = 20                                    # a late start can still run an hourly scan this long after its time


def _plus(t, minutes):
    return (dt.datetime.combine(dt.date(2000, 1, 1), t) + dt.timedelta(minutes=minutes)).time()


# (key, mode, start time ET, latest start ET, label) - in time order
SCHEDULE = (
    [("open", "open", SCAN_OPEN_TIME, SCAN_OPEN_LATEST, "Open 09:45")]
    + [(f"h{t:%H%M}", "hourly", t, _plus(t, HOURLY_GRACE_MIN), f"Hourly {t:%H:%M}") for t in HOURLY_TIMES]
)

RISK_FREE = 0.04
CONTRACT_SIZE = 100                          # 1 lot = 1 contract = 100 shares

MIN_SCORE = 4                                # minimum |score| to be considered (was 3 - stricter for accuracy)
STRONG_SCORE = 5                             # |score| for a "strong" signal
CANDIDATES = 16                              # stocks (best scores) whose option chains are checked

# accuracy filters
USE_MARKET_FILTER = True                     # SPY direction adds / removes a point (trade WITH the market)
MARKET_SYM = "SPY"
MIN_RVOL = 0.8                               # skip stocks trading on clearly below-normal volume (not applied premarket)
SKIP_EARNINGS = True                         # live scans: skip when earnings fall before the option expires (IV crush)

# option filters / ranking
TARGET_DELTA = 0.65
DELTA_MIN, DELTA_MAX = 0.55, 0.80
MAX_SPREAD_PCT = 8.0                         # (was 10)
MIN_OI = 200                                 # (was 100)

# trade plan (stock based, converted to option prices with the Greeks / Black-Scholes)
HOLD_HOURS = 6.0                             # time decay assumed before exit
TP_FRAC = 1.0                                # stock target = this x the expected move (smaller = closer target, higher win rate)
SL_MOVE_FRAC = 0.5                           # stock stop = 50% of the expected move against the trade
MIN_LOSS, MAX_LOSS = 0.10, 0.30              # option stop loss clamped to 10-30% of premium
MIN_RR = 1.3                                 # skip trades with reward:risk below this
REQUIRE_PA = False                           # True = only trade when price action points the same way as the trade
# TP_FRAC / SL_MOVE_FRAC / MAX_LOSS / MIN_SCORE / MIN_RR / REQUIRE_PA are what the Backtest tab's "Sweep" tunes -
# copy the values it finds here (live scans and the backtest both read these).

# price action
SWING_K = 3                                  # a swing high/low = extreme of 3 bars either side (5-min bars)
BLOCKED_FRAC = 0.35                          # next S/R closer than 35% of the expected move = trade is "blocked"

# backtest assumptions (yfinance has no history of option quotes, so the option is MODELLED)
BT_IV_MULT = 1.10                            # assumed option IV = 20-day historical vol x this
BT_SLIPPAGE = 0.02                           # 2% of premium lost on entry and on stop / time exits
BT_COMMISSION = 0.65                         # $ per contract per side

# self-improvement
LEARN_ON = True                              # the sidebar toggle overrides this
MIN_TAG_N = 25                               # a signal needs this many settled trades before its weight changes
LEARN_STRENGTH = 0.25                        # weight = 1 + strength x t-stat (t = +-2 -> 1.5 / 0.5), clipped below
W_MIN, W_MAX = 0.0, 1.5                      # a signal can be switched off (0) or boosted by at most 50%
SETTLE_AFTER = dt.time(16, 10)               # plan result is fetched after this time on the trade day (ET)
EXPIRY_SETTLE_AFTER = dt.time(16, 30)        # expiry result is fetched after this time on the expiry day (ET)
MAX_BAR_AGE_DAYS = 58                        # yfinance keeps ~60 days of 5-minute bars
LEARNED_W = {}                               # {signal name: weight}, filled by refresh_weights()
SETTLE_EVERY_SEC = 1800                      # while the page is open, look for finished trades this often

WORKERS = 8                                  # parallel downloads

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "PG", "JPM", "HD", "MA",
    "XOM", "BAC", "PFE", "KO", "PEP",
    "CSCO", "ABBV", "ADBE", "NFLX", "CRM",
    "LLY", "AVGO", "COST", "WMT", "ORCL",
    "MRK", "TMO", "ACN", "MCD", "LIN",
    "ABT", "WFC", "DIS", "DHR", "TXN",
    "QCOM", "AMD", "INTC", "IBM", "GS",
    "CAT", "AMGN", "HON", "UNP", "LOW",
    # ---- 50 more liquid, optionable US stocks ----
    "PLTR", "UBER", "NOW", "INTU", "ISRG",
    "NEE", "SPGI", "BLK", "AXP", "MS",
    "C", "T", "VZ", "CMCSA", "NKE",
    "SBUX", "PM", "MO", "CVX", "COP",
    "SLB", "GE", "RTX", "LMT", "BA",
    "DE", "MMM", "UPS", "FDX", "TGT",
    "MU", "AMAT", "LRCX", "PANW", "CRWD",
    "PYPL", "COIN", "ABNB", "MRVL", "ANET",
    "GILD", "BMY", "CVS", "MDT", "SYK",
    "VRTX", "REGN", "PGR", "CB", "KLAC",
]

# ----- market panels -----
INDICES = [
    ("S&P 500", "^GSPC"), ("Dow Jones", "^DJI"), ("Nasdaq Composite", "^IXIC"), ("Nasdaq 100", "^NDX"),
    ("Russell 2000", "^RUT"), ("S&P MidCap 400", "^MID"), ("NYSE Composite", "^NYA"), ("Semiconductors SOX", "^SOX"),
    ("VIX (fear index)", "^VIX"), ("SPY - S&P 500 ETF", "SPY"), ("QQQ - Nasdaq 100 ETF", "QQQ"),
    ("DIA - Dow ETF", "DIA"), ("IWM - Russell 2000 ETF", "IWM"),
]
COMMODITIES = [
    ("Gold", "GC=F"), ("Silver", "SI=F"), ("Crude Oil WTI", "CL=F"), ("Brent Crude", "BZ=F"),
    ("Natural Gas", "NG=F"), ("Copper", "HG=F"), ("Platinum", "PL=F"), ("Palladium", "PA=F"),
]
TOP10 = [("Apple", "AAPL"), ("Microsoft", "MSFT"), ("Nvidia", "NVDA"), ("Alphabet", "GOOGL"), ("Amazon", "AMZN"),
         ("Meta", "META"), ("Broadcom", "AVGO"), ("Tesla", "TSLA"), ("Berkshire B", "BRK-B"), ("Eli Lilly", "LLY")]
MARKET_REFRESH_SEC = 30

# ----------------------------- CONFIG / STATE / TELEGRAM -----------------------------

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    return str(o)


def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_config():
    return _load_json(CONFIG_FILE)


def save_config(cfg):
    _save_json(CONFIG_FILE, cfg)


def load_state():
    return _load_json(STATE_FILE)


def save_state(state):
    _save_json(STATE_FILE, state)


def send_telegram(cfg, msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": msg}, timeout=15)
        return r.ok
    except Exception:
        return False

# ----------------------------- INDICATORS -----------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

# ----------------------------- GREEKS (Black-Scholes) -----------------------------

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1d2(S, K, T, sigma, r):
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    return d1, d1 - sq


def bs_price(S, K, T, sigma, side, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if side == "CALL" else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, sigma, r)
    if side == "CALL":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def greeks(S, K, T, sigma, side, r=RISK_FREE):
    """delta, gamma, theta (per day), vega (per 1 vol point), rho (per 1% rate) - all per share."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    d1, d2 = _d1d2(S, K, T, sigma, r)
    sq = sigma * math.sqrt(T)
    gamma = _npdf(d1) / (S * sq)
    vega = S * _npdf(d1) * math.sqrt(T) / 100.0
    decay = -S * _npdf(d1) * sigma / (2 * math.sqrt(T))
    if side == "CALL":
        delta = _ncdf(d1)
        theta = (decay - r * K * math.exp(-r * T) * _ncdf(d2)) / 365.0
        rho = K * T * math.exp(-r * T) * _ncdf(d2) / 100.0
    else:
        delta = _ncdf(d1) - 1.0
        theta = (decay + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365.0
        rho = -K * T * math.exp(-r * T) * _ncdf(-d2) / 100.0
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

# ----------------------------- DATA FETCH -----------------------------

def _to_et(df):
    df = df.copy()
    idx = df.index
    df.index = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize(ET)
    return df


def _clean_intraday(df):
    if df is None or df.empty:
        return None
    df = _to_et(df).rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    keep = np.array([dt.time(4, 0) <= x < MARKET_CLOSE for x in df.index.time])
    return df[keep]


def fetch_intraday(sym):
    """5-minute bars incl. premarket (04:00-16:00 ET), last 5 days."""
    return _clean_intraday(yf.Ticker(sym).history(period="5d", interval="5m", prepost=True))


def fetch_daily(sym, period="3mo"):
    d = yf.Ticker(sym).history(period=period, interval="1d")
    if d is None or d.empty:
        return None
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    return d

# ----------------------------- PRICE ACTION -----------------------------

def _swings(df, k=SWING_K):
    """Confirmed swing highs / lows (prices, oldest -> newest). A swing needs k bars on each side, so it never
    uses future data."""
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    sh, sl = [], []
    for i in range(k, len(df) - k):
        w_hi, w_lo = hi[i - k:i + k + 1], lo[i - k:i + k + 1]
        if hi[i] == w_hi.max() and (w_hi == hi[i]).sum() == 1:
            sh.append(float(hi[i]))
        if lo[i] == w_lo.min() and (w_lo == lo[i]).sum() == 1:
            sl.append(float(lo[i]))
    return sh, sl


def price_action(df, hist_d, spot, atr5, pm_high, pm_low, mode):
    """Returns (points, tags, info). Points are added to the indicator score.
       +1/-1 each: swing structure, break of structure, candle pattern, previous-day high/low break.
       info also holds the nearest support/resistance ('room') and the last swing low/high (stop reference)."""
    pts, tags = 0, []

    def add(v, tag):
        nonlocal pts
        pts += v
        if v:
            tags.append(("+" if v > 0 else "-") + "PA:" + tag)

    reg = df[np.array([x >= MARKET_OPEN for x in df.index.time])].tail(120)
    sh, sl = _swings(reg) if len(reg) >= 12 else ([], [])
    pdh, pdl = float(hist_d["high"].iloc[-1]), float(hist_d["low"].iloc[-1])
    info = dict(struct="none", candle="none", pdh=pdh, pdl=pdl, sw_hi=None, sw_lo=None,
                room_up=None, room_dn=None)

    # 1) swing structure + break of structure
    if len(sh) >= 2 and len(sl) >= 2:
        if sh[-1] > sh[-2] and sl[-1] > sl[-2]:
            add(1, "HH/HL")
            info["struct"] = "HH/HL (uptrend)"
        elif sh[-1] < sh[-2] and sl[-1] < sl[-2]:
            add(-1, "LH/LL")
            info["struct"] = "LH/LL (downtrend)"
        else:
            info["struct"] = "range"
    if sh and spot > sh[-1]:
        add(1, "BOS-up")
    elif sl and spot < sl[-1]:
        add(-1, "BOS-down")

    # 2) candle pattern on the last bar, only where it matters (reversal at an extreme, or a strong bar)
    if mode != "pre" and len(reg) >= 14:
        b0, b1 = reg.iloc[-1], reg.iloc[-2]
        o0, c0, h0, l0 = float(b0["open"]), float(b0["close"]), float(b0["high"]), float(b0["low"])
        o1, c1 = float(b1["open"]), float(b1["close"])
        rng0, body0, body1 = h0 - l0, abs(c0 - o0), abs(c1 - o1)
        if rng0 > 0:
            up_w, lo_w = h0 - max(o0, c0), min(o0, c0) - l0
            w = reg.tail(12)
            lo12, hi12 = float(w["low"].min()), float(w["high"].max())
            r12 = max(hi12 - lo12, 1e-9)
            at_low, at_high = (l0 - lo12) / r12 <= 0.25, (hi12 - h0) / r12 <= 0.25
            bull_eng = c1 < o1 and c0 > o0 and c0 >= o1 and o0 <= c1 and body0 > body1
            bear_eng = c1 > o1 and c0 < o0 and c0 <= o1 and o0 >= c1 and body0 > body1
            hammer = lo_w >= 2 * body0 and lo_w >= 0.6 * rng0 and up_w <= 0.25 * rng0
            star = up_w >= 2 * body0 and up_w >= 0.6 * rng0 and lo_w <= 0.25 * rng0
            if (bull_eng or hammer) and at_low:
                name = "Bull engulf" if bull_eng else "Hammer"
                add(1, name)
                info["candle"] = name
            elif (bear_eng or star) and at_high:
                name = "Bear engulf" if bear_eng else "Shooting star"
                add(-1, name)
                info["candle"] = name
            elif c0 > o0 and body0 >= 0.7 * rng0 and body0 >= atr5:
                add(1, "Strong bar")
                info["candle"] = "Strong green bar"
            elif c0 < o0 and body0 >= 0.7 * rng0 and body0 >= atr5:
                add(-1, "Strong bar")
                info["candle"] = "Strong red bar"

    # 3) previous-day high / low
    if spot > pdh:
        add(1, "PDH-break")
    elif spot < pdl:
        add(-1, "PDL-break")

    # 4) nearest support / resistance ("room") and the stop reference swings
    levels = [pdh, pdl] + [x for x in (pm_high, pm_low) if x is not None] + sh[-4:] + sl[-4:]
    gap = 0.1 * atr5
    above = [x - spot for x in levels if x > spot + gap]
    below = [spot - x for x in levels if x < spot - gap]
    info["room_up"] = min(above) if above else None
    info["room_dn"] = min(below) if below else None
    info["sw_lo"] = next((x for x in reversed(sl) if x < spot - gap), None)
    info["sw_hi"] = next((x for x in reversed(sh) if x > spot + gap), None)
    return pts, tags, info

# ----------------------------- STOCK ANALYSIS + DAY PREDICTION -----------------------------

def analyse_df(sym, df, dd, mode):
    """Indicators + price action from the data given (the last bar of df = 'now'). Works for live data and for
    the backtest (df cut at the scan time). mode: 'pre', 'open' or 'hourly'. Returns the raw pieces; finalize()
    turns them into the score and the day prediction."""
    try:
        if df is None or dd is None or len(df) < 40:
            return None

        sess = df.index[-1].date()
        hist_d = dd[np.array([x < sess for x in dd.index.date])]
        if len(hist_d) < 20:
            return None
        prev_close = float(hist_d["close"].iloc[-1])
        day_atr = float(atr(hist_d).iloc[-1])                               # daily ATR(14)
        avg_vol = float(hist_d["volume"].tail(20).mean())
        hv = float(np.log(hist_d["close"]).diff().dropna().tail(20).std() * math.sqrt(252))

        close = df["close"]
        e9, e15 = ema(close, 9), ema(close, 15)
        macd = ema(close, 12) - ema(close, 26)
        hist = macd - ema(macd, 9)
        r = rsi(close)
        a5 = atr(df)

        today = df[np.array([x == sess for x in df.index.date])]
        tt = today.index.time
        pm = today[np.array([x < MARKET_OPEN for x in tt])]
        reg = today[np.array([x >= MARKET_OPEN for x in tt])]

        tp = (today["high"] + today["low"] + today["close"]) / 3
        vsum = float(today["volume"].sum())
        vwap = float((tp * today["volume"]).sum() / vsum) if vsum > 0 else float(tp.iloc[-1])

        spot = float(close.iloc[-1])
        rsi_last = float(r.iloc[-1])
        h_last, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
        atr5 = max(float(a5.iloc[-1]), 1e-9)

        pm_vol, reg_vol = float(pm["volume"].sum()), float(reg["volume"].sum())
        pm_high = float(pm["high"].max()) if len(pm) else None
        pm_low = float(pm["low"].min()) if len(pm) else None
        day_open = float(reg["open"].iloc[0]) if len(reg) else (float(pm["close"].iloc[-1]) if len(pm) else spot)
        gap = (day_open - prev_close) / prev_close * 100

        if mode != "pre" and len(reg) > 0 and avg_vol > 0:
            last_bar = df.index[-1]
            elapsed = (last_bar.hour * 60 + last_bar.minute + 5) - (9 * 60 + 30)       # minutes since the open
            # typical share of the day's volume traded by then (U-shaped intraday curve)
            frac = float(np.interp(elapsed, [0, 10, 30, 60, 120, 180, 240, 300, 360, 390],
                                   [0, 0.07, 0.16, 0.26, 0.40, 0.50, 0.60, 0.72, 0.85, 1.0]))
            rvol = reg_vol / (avg_vol * max(frac, 0.02))
        elif avg_vol > 0:
            rvol = pm_vol / (avg_vol * 0.03)        # premarket normally ~3% of the day's volume
        else:
            rvol = 0.0

        # ---- indicator score (direction) ----
        s, tags = 0, []

        def add(v, tag):
            nonlocal s
            s += v
            if v:
                tags.append(("+" if v > 0 else "-") + tag)

        add(1 if e9.iloc[-1] > e15.iloc[-1] else -1, "EMA9/15")
        add(1 if spot > vwap else -1, "VWAP")
        add(1 if rsi_last > 55 else (-1 if rsi_last < 45 else 0), "RSI")
        add(1 if (h_last > 0 and h_last > h_prev) else (-1 if (h_last < 0 and h_last < h_prev) else 0), "MACD")
        add(int(np.sign(spot - prev_close)) if rvol >= 1.2 else 0, "Vol")
        add(int(np.sign(gap)) if abs(gap) >= 0.5 else 0, "Gap")
        mom = (spot - float(close.iloc[-7])) / atr5
        add(1 if mom > 0.7 else (-1 if mom < -0.7 else 0), "Mom")
        if mode != "pre" and len(reg) > 0 and pm_high is not None:
            add(1 if spot > pm_high else (-1 if spot < pm_low else 0), "PM-break")
        if mode != "pre" and len(reg) > 0 and abs(spot - day_open) > 0.25 * atr5:
            add(1 if spot > day_open else -1, "vsOpen")
        ob = (-1, "overbought") if rsi_last > 75 else ((1, "oversold") if rsi_last < 25 else (0, ""))

        # ---- price action ----
        pa_s, pa_tags, pa = price_action(df, hist_d, spot, atr5, pm_high, pm_low, mode)

        return dict(sym=sym, mode=mode, spot=spot, vwap=vwap, rsi=rsi_last, gap=gap, rvol=float(rvol),
                    day_atr=day_atr, atr5=atr5, hv=hv, prev_close=prev_close, pm_high=pm_high, pm_low=pm_low,
                    s_core=int(s), tags_core=tags, ob=ob, pa_s=int(pa_s), pa_tags=pa_tags, pa=pa)
    except Exception:
        return None


def mkt_dir(a0):
    """Market (SPY) bias from its indicator score: +1 bullish, -1 bearish, 0 neutral / unknown."""
    if not a0:
        return 0
    s = a0["s_core"]
    return 1 if s >= 2 else (-1 if s <= -2 else 0)


def _tag_parts(t):
    """'+EMA9/15' -> (1, 'EMA9/15'); '-PA:BOS-down' -> (-1, 'PA:BOS-down'); 'PA:blocked' -> (0, 'PA:blocked')."""
    if t and t[0] in "+-":
        return (1 if t[0] == "+" else -1), t[1:]
    return 0, t


def finalize(a, use_pa=True, pre_prev=None, mkt=0, learn=False):
    """Final score, day prediction and stop reference. use_pa=False gives the original indicator-only score
    (used by the backtest to compare). mkt = market bias from mkt_dir(). learn=True multiplies every signal by
    its learned weight (live scans only - the backtest keeps all weights at 1)."""
    a = dict(a)
    sym = a["sym"]
    W = LEARNED_W if learn else {}
    tags = list(a["tags_core"])

    def tval(t):
        sg, name = _tag_parts(t)
        return sg * W.get(name, 1.0)

    s = sum(tval(t) for t in tags) if W else a["s_core"]

    def add(v, tag):
        nonlocal s
        s += v * W.get(tag, 1.0)
        if v:
            tags.append(("+" if v > 0 else "-") + tag)

    if a["mode"] != "pre" and pre_prev and pre_prev.get(sym) and s != 0:
        # premarket scan agreed with today's direction -> strengthen it; disagreed -> weaken it
        agree = (pre_prev[sym] > 0) == (s > 0)
        add((1 if s > 0 else -1) * (1 if agree else -1), "PM-scan")
    if USE_MARKET_FILTER and mkt and s != 0:
        # trade with the market: SPY agrees -> +1, SPY fights the trade -> -1
        add((1 if s > 0 else -1) * (1 if (mkt > 0) == (s > 0) else -1), "Mkt")
    add(*a["ob"])
    if use_pa:
        s += sum(tval(t) for t in a["pa_tags"]) if W else a["pa_s"]
        tags += a["pa_tags"]

    def predict(score):
        conf = float(np.clip(score / 7.0, -1, 1))
        return (1 if score > 0 else -1), a["day_atr"] * (0.25 + 0.20 * abs(conf))

    sign, move = predict(s)
    if use_pa and s != 0:
        room = a["pa"]["room_up"] if s > 0 else a["pa"]["room_dn"]
        if room is not None and room < BLOCKED_FRAC * move:      # next support/resistance is right in the way
            s -= sign
            tags.append("PA:blocked")
            sign, move = predict(s)

    spot = a["spot"]
    pred_close = spot + sign * move
    a.update(score=(round(float(s), 2) if W else int(s)), tags=tags, move=float(move),
             pred_close=float(pred_close),
             pred_high=float(max(spot, pred_close) + 0.25 * a["day_atr"]),
             pred_low=float(min(spot, pred_close) - 0.25 * a["day_atr"]),
             pre_score=(pre_prev or {}).get(sym),
             sl_ref=(a["pa"]["sw_lo"] if s > 0 else a["pa"]["sw_hi"]) if use_pa else None,
             pa_on=use_pa)
    return a


def pa_agrees(a):
    """True when the price-action points add up in the same direction as the final score."""
    return a["pa_s"] != 0 and (a["pa_s"] > 0) == (a["score"] > 0)


def analyse(sym, mode, pre_prev, mkt=0):
    """Live analysis (current data) with price action on and the learned weights applied."""
    try:
        base = analyse_df(sym, fetch_intraday(sym), fetch_daily(sym), mode)
        return finalize(base, True, pre_prev, mkt, learn=True) if base else None
    except Exception:
        return None


def market_bias(mode):
    """Live SPY bias (+1 / 0 / -1)."""
    if not USE_MARKET_FILTER:
        return 0
    try:
        return mkt_dir(analyse_df(MARKET_SYM, fetch_intraday(MARKET_SYM), fetch_daily(MARKET_SYM), mode))
    except Exception:
        return 0


def earnings_before(sym, expiry):
    """True when an earnings date falls between today and the option expiry (IV crush / gap risk)."""
    try:
        cal = yf.Ticker(sym).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return False
        exp = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
        today = dt.datetime.now(ET).date()
        for d in dates:
            d = d.date() if isinstance(d, dt.datetime) else d
            if today <= d <= exp:
                return True
        return False
    except Exception:
        return False

# ----------------------------- OPTION PICKER (all Greeks) -----------------------------

def pick_option(sym, spot, side, hv):
    """Next expiry AFTER today -> best contract by delta / gamma / theta / vega / spread / liquidity / IV-vs-HV."""
    try:
        t = yf.Ticker(sym)
        now = dt.datetime.now(ET)
        today = now.date()

        future = sorted(e for e in (t.options or []) if dt.datetime.strptime(e, "%Y-%m-%d").date() > today)
        if not future:
            return None
        expiry = future[0]

        exp_dt = dt.datetime.strptime(expiry, "%Y-%m-%d").replace(hour=16, tzinfo=ET)
        T = (exp_dt - now).total_seconds() / (365 * 86400)
        dte = (exp_dt.date() - today).days

        chain = t.option_chain(expiry)
        df = (chain.calls if side == "CALL" else chain.puts).copy()
        df = df[(df["strike"] > spot * 0.90) & (df["strike"] < spot * 1.10)]
        if df.empty:
            return None

        rows = []
        for _, o in df.iterrows():
            bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
            last = float(o.get("lastPrice") or 0)
            iv = float(o.get("impliedVolatility") or 0)
            oi = 0 if pd.isna(o.get("openInterest")) else int(o.get("openInterest") or 0)
            vol = 0 if pd.isna(o.get("volume")) else int(o.get("volume") or 0)

            market_open = bid > 0 and ask > 0
            if market_open:
                mid, spread, entry = (bid + ask) / 2, (ask - bid) / ask * 100, ask
            elif last > 0:                              # no live quote: use last traded price
                mid, spread, entry = last, 0.0, last
            else:
                continue

            if not (0.05 <= iv <= 3.0) or oi < MIN_OI or spread > MAX_SPREAD_PCT:
                continue
            g = greeks(spot, float(o["strike"]), T, iv, side)
            if not g or not (DELTA_MIN <= abs(g["delta"]) <= DELTA_MAX):
                continue

            d = abs(g["delta"])
            theta_pct = abs(g["theta"]) / max(mid, 0.01) * 100        # daily decay, % of premium
            gamma_pct = g["gamma"] * spot / max(mid, 0.01) * 100      # responsiveness
            vega_pct = g["vega"] / max(mid, 0.01) * 100               # premium change per vol point
            iv_ratio = iv / hv if hv > 0 else 1.0
            quality = (
                100
                - abs(d - TARGET_DELTA) * 100          # delta near the sweet spot
                - spread * 2                            # tight spread
                - theta_pct * 4                         # low time decay
                + gamma_pct * 2                         # good gamma
                - vega_pct * 1.0                        # not too IV-sensitive
                - max(0.0, iv_ratio - 1.2) * 10         # don't overpay for rich IV
                + math.log10(oi + vol + 1) * 3          # liquidity
            )
            rows.append(dict(symbol=sym, side=side, expiry=expiry, dte=dte, T=T,
                             strike=float(o["strike"]), bid=bid, ask=ask, last=last, entry=entry,
                             iv=iv, iv_ratio=iv_ratio, vol=vol, oi=oi, spread=spread, market_open=market_open,
                             delta=g["delta"], gamma=g["gamma"], theta=g["theta"], vega=g["vega"], rho=g["rho"],
                             quality=quality))
        return max(rows, key=lambda x: x["quality"]) if rows else None
    except Exception:
        return None


def trade_plan(opt, a, tp_frac=None, sl_frac=None, min_loss=None, max_loss=None):
    """Stock targets from the day prediction, converted to option prices with Black-Scholes.
       With price action on, the stock stop goes just beyond the last swing low (call) / swing high (put).
       tp_frac / sl_frac / min_loss / max_loss default to the settings at the top of the file."""
    tp_frac = TP_FRAC if tp_frac is None else tp_frac
    sl_frac = SL_MOVE_FRAC if sl_frac is None else sl_frac
    min_loss = MIN_LOSS if min_loss is None else min_loss
    max_loss = MAX_LOSS if max_loss is None else max_loss
    side = opt["side"]
    sign = 1 if side == "CALL" else -1
    S, K, T, iv = a["spot"], opt["strike"], opt["T"], opt["iv"]
    T_exit = max(T - HOLD_HOURS / (24 * 365), 1.0 / (24 * 365))

    now_px = bs_price(S, K, T, iv, side)
    stock_tp = S + sign * a["move"] * tp_frac
    stock_sl = S - sign * a["move"] * sl_frac
    ref = a.get("sl_ref")
    if ref is not None:                                   # structural stop, kept within a sensible distance
        dist = abs(S - ref) + 0.1 * a["atr5"]
        if 0.25 * a["move"] <= dist <= max(1.0, sl_frac) * a["move"]:
            stock_sl = S - sign * dist
    gain = bs_price(stock_tp, K, T_exit, iv, side) - now_px
    loss = now_px - bs_price(stock_sl, K, T_exit, iv, side)

    entry = opt["entry"]
    if gain <= 0:
        return None
    loss = min(max(loss, entry * min_loss), entry * max_loss)

    entry_r = round(entry, 2)
    tp = round(entry + gain, 2)
    sl = round(entry - loss, 2)
    return dict(entry=entry_r, sl=sl, tp=tp,
                profit_1lot=round((tp - entry_r) * CONTRACT_SIZE, 2),
                loss_1lot=round((entry_r - sl) * CONTRACT_SIZE, 2),
                cost_1lot=round(entry_r * CONTRACT_SIZE, 2),
                rr=round((tp - entry_r) / max(entry_r - sl, 1e-9), 2),
                stock_tp=round(stock_tp, 2), stock_sl=round(stock_sl, 2))

# ----------------------------- LOGGING -----------------------------

def log_picks(res):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_et", "scan", "symbol", "side", "expiry", "strike", "spot", "pred_close", "pred_high",
                        "pred_low", "entry", "stop_loss", "take_profit", "profit_1lot", "delta", "gamma", "theta",
                        "vega", "rho", "iv", "score", "top", "price_action"])
        for side in ("CALL", "PUT"):
            it = res.get(side)
            if not it:
                continue
            o, p, a = it["opt"], it["plan"], it["a"]
            w.writerow([res["time"], res["label"], o["symbol"], side, o["expiry"], o["strike"], round(a["spot"], 2),
                        round(a["pred_close"], 2), round(a["pred_high"], 2), round(a["pred_low"], 2),
                        p["entry"], p["sl"], p["tp"], p["profit_1lot"], round(o["delta"], 3), round(o["gamma"], 4),
                        round(o["theta"], 3), round(o["vega"], 3), round(o["rho"], 3), round(o["iv"], 3),
                        a["score"], res.get("top") == side, " ".join(a["pa_tags"])])

# ----------------------------- SELF-IMPROVEMENT: TRACK -> SETTLE -> LEARN -----------------------------

def load_tracker():
    d = _load_json(TRACK_FILE)
    return d if isinstance(d, list) else []


def save_tracker(rows):
    _save_json(TRACK_FILE, rows)


def refresh_weights():
    """Load the learned signal weights (empty = all 1.0 when self-learning is off or nothing learned yet)."""
    global LEARNED_W
    d = _load_json(LEARN_FILE)
    LEARNED_W = dict(d.get("weights", {})) if (LEARN_ON and isinstance(d, dict)) else {}
    return LEARNED_W


def record_trades(built, res, now):
    """Save every candidate contract of a live scan (not just the 2 best) so its result can be fetched later."""
    if res["mode"] == "pre" or now.weekday() >= 5 or not (MARKET_OPEN <= now.time() < EXIT_TIME):
        return 0
    rows = load_tracker()
    have = {r["id"] for r in rows}
    n = 0
    for b in built:
        o, p, a = b["opt"], b["plan"], b["a"]
        rid = f"{res['time']}|{o['symbol']}|{b['side']}"
        if rid in have:
            continue
        picked = res.get(b["side"]) is b
        rows.append(dict(
            id=rid, time=res["time"], date=str(now.date()), scan=res["label"], mode=res["mode"],
            sym=o["symbol"], side=b["side"], picked=bool(picked), top=bool(picked and res.get("top") == b["side"]),
            strike=float(o["strike"]), expiry=o["expiry"], T=float(o["T"]), iv=float(o["iv"]),
            entry=float(p["entry"]), sl=float(p["sl"]), tp=float(p["tp"]), spot=float(a["spot"]),
            score=float(a["score"]), tags=list(a["tags"]), pa_s=int(a["pa_s"]), rr=float(p["rr"]),
            delta=float(o["delta"]),
            plan_pnl=None, plan_reason=None, plan_exit=None, plan_time=None,
            expiry_close=None, expiry_value=None, expiry_pnl=None))
        n += 1
    if n:
        save_tracker(rows)
    return n


def _prefetch(symbols, fn):
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fn, s): s for s in symbols}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = None
    return out


def _fetch_bars_60d(sym):
    return _clean_intraday(yf.Ticker(sym).history(period="60d", interval="5m", prepost=True))


def settle_pending(cfg=None):
    """Fetch the results of finished trades.
       Stage 1 (after the trade day): replay the plan on the 5-minute bars -> plan_pnl (stop / target / 15:45 exit).
       Stage 2 (after expiry): the stock's close on the expiry date -> exact intrinsic value -> expiry_pnl.
       Then the signal weights are re-learned. Returns the number of trades that got a new result."""
    rows = load_tracker()
    if not rows:
        return 0
    now = dt.datetime.now(ET)

    def plan_due(r):
        return (r.get("plan_pnl") is None and not r.get("plan_skip")
                and now >= dt.datetime.combine(dt.date.fromisoformat(r["date"]), SETTLE_AFTER, tzinfo=ET))

    def exp_due(r):
        if r.get("expiry_pnl") is not None or r.get("exp_skip"):
            return False
        e = dt.date.fromisoformat(r["expiry"])
        return e < now.date() or (e == now.date() and now.time() >= EXPIRY_SETTLE_AFTER)

    need_bars = sorted({r["sym"] for r in rows if plan_due(r)})
    need_daily = sorted({r["sym"] for r in rows if exp_due(r)})
    if not need_bars and not need_daily:
        return 0
    bars = _prefetch(need_bars, _fetch_bars_60d)
    daily = _prefetch(need_daily, lambda s: fetch_daily(s, "6mo"))

    new_plan, new_exp = [], []
    for r in rows:
        day = dt.date.fromisoformat(r["date"])
        exp = dt.date.fromisoformat(r["expiry"])
        if plan_due(r):
            df = bars.get(r["sym"])
            age = (now.date() - day).days
            res_bars = None
            if df is not None:
                t0 = pd.Timestamp(r["time"]).tz_convert(ET)
                exit_ts = pd.Timestamp(dt.datetime.combine(day, EXIT_TIME), tz=ET)
                j0, j1 = df.index.searchsorted(t0), df.index.searchsorted(exit_ts)
                res_bars = df.iloc[j0:j1]
            if res_bars is not None and len(res_bars) >= 3:
                opt = dict(side=r["side"], strike=r["strike"], iv=r["iv"],
                           exp_dt=dt.datetime.combine(exp, MARKET_CLOSE, tzinfo=ET))
                # the logged entry is the real ask; shift the Black-Scholes curve so it starts exactly there
                offset = r["entry"] - bs_price(r["spot"], r["strike"], r["T"], r["iv"], r["side"])
                sim = simulate_trade(opt, dict(tp=r["tp"], sl=r["sl"]), res_bars, BT_SLIPPAGE, offset)
                if sim:
                    px, reason, ts = sim
                    r.update(plan_pnl=round((px - r["entry"]) * CONTRACT_SIZE - 2 * BT_COMMISSION, 2),
                             plan_reason=reason, plan_exit=round(float(px), 2), plan_time=ts.isoformat())
                    new_plan.append(r)
            elif age > 5 or (df is not None and age > MAX_BAR_AGE_DAYS):
                r["plan_skip"] = True                          # no bars available any more
        if exp_due(r):
            dd = daily.get(r["sym"])
            if dd is not None:
                m = dd[np.array([x == exp for x in dd.index.date])]
                if len(m):
                    close = float(m["close"].iloc[0])
                    val = max(close - r["strike"], 0.0) if r["side"] == "CALL" else max(r["strike"] - close, 0.0)
                    r.update(expiry_close=round(close, 2), expiry_value=round(val, 2),
                             expiry_pnl=round((val - r["entry"]) * CONTRACT_SIZE - BT_COMMISSION, 2))
                    new_exp.append(r)
                elif (now.date() - exp).days > 6:
                    r["exp_skip"] = True
            elif (now.date() - exp).days > 6:
                r["exp_skip"] = True

    save_tracker(rows)
    compute_learning(rows)
    refresh_weights()
    if cfg is not None and (new_plan or new_exp):
        send_telegram(cfg, settle_text(new_plan, new_exp))
    return len(new_plan) + len(new_exp)


def settle_text(new_plan, new_exp):
    lines = ["📒 RESULTS UPDATE"]
    if new_plan:
        w = sum(1 for r in new_plan if r["plan_pnl"] > 0)
        lines.append(f"Trade day: {len(new_plan)} settled, {w} won, P&L ${sum(r['plan_pnl'] for r in new_plan):+,.0f}/lot")
    if new_exp:
        w = sum(1 for r in new_exp if r["expiry_pnl"] > 0)
        lines.append(f"Expired: {len(new_exp)} settled, {w} in profit at expiry, "
                     f"${sum(r['expiry_pnl'] for r in new_exp):+,.0f}/lot")
    return "\n".join(lines)


def compute_learning(rows):
    """Re-learn the weight of every signal from the plan results. One trade per (day, stock, side) is used so
    the same setup repeated in several hourly scans is not counted many times. A signal's weight is
    1 + LEARN_STRENGTH x t-stat of (its aligned trades' P&L vs all other trades), clipped to W_MIN..W_MAX, and only
    changes after MIN_TAG_N settled trades - small samples leave the weight at 1.0."""
    out = dict(updated=dt.datetime.now(ET).isoformat(timespec="seconds"), trades=0, weights={}, tags=[],
               by_side={}, by_scan={})
    done = [r for r in rows if r.get("plan_pnl") is not None]
    if done:
        df = (pd.DataFrame(done).sort_values("time").drop_duplicates(["date", "sym", "side"], keep="first")
              .reset_index(drop=True))
        pnl = df["plan_pnl"].astype(float).to_numpy()
        sg = df["side"].map({"CALL": 1, "PUT": -1}).to_numpy()
        aligned = []
        for tags, s in zip(df["tags"], sg):
            al = set()
            for t in tags:
                sign, name = _tag_parts(t)
                if sign and sign == s:
                    al.add(name)
            aligned.append(al)
        names = sorted(set().union(*aligned)) if aligned else []
        for name in names:
            m = np.array([name in al for al in aligned])
            n_in, n_out = int(m.sum()), int((~m).sum())
            p_in, p_out = pnl[m], pnl[~m]
            t = 0.0
            if n_in >= 2 and n_out >= 2:
                se = math.sqrt(p_in.var(ddof=1) / n_in + p_out.var(ddof=1) / n_out)
                t = float((p_in.mean() - p_out.mean()) / se) if se > 0 else 0.0
            active = n_in >= MIN_TAG_N and n_out >= 10
            w = float(np.clip(1 + LEARN_STRENGTH * t, W_MIN, W_MAX)) if active else 1.0
            if active:
                out["weights"][name] = round(w, 2)
            out["tags"].append(dict(signal=name, trades=n_in, win_pct=round(float((p_in > 0).mean() * 100), 1),
                                    avg_pnl=round(float(p_in.mean()), 1), t_stat=round(t, 2),
                                    weight=round(w, 2), active=active))
        out["trades"] = len(df)
        out["win_pct"] = round(float((pnl > 0).mean() * 100), 1)
        out["avg_pnl"] = round(float(pnl.mean()), 1)
        for col, key in (("side", "by_side"), ("scan", "by_scan")):
            g = df.groupby(col)["plan_pnl"]
            out[key] = {str(k): dict(trades=int(v.size), win_pct=round(float((v > 0).mean() * 100), 1),
                                     avg_pnl=round(float(v.mean()), 1)) for k, v in g}
    _save_json(LEARN_FILE, out)
    return out

# ----------------------------- SCAN -----------------------------

def build_top10(analysed):
    """Top 10 US shares: for each one the predicted direction (CALL if the day estimate is above the price, else PUT),
       the best next-expiry contract (same Greeks ranking as the main scan) and its trade plan (entry, stop loss, take
       profit). Every share gets a lean; weak signals, low reward:risk and earnings before expiry are flagged
       instead of hidden."""
    by = {a["sym"]: a for a in analysed}

    def one(item):
        name, sym = item
        a = by.get(sym)
        if not a:
            return dict(name=name, sym=sym, error="no price data")
        side = "CALL" if a["pred_close"] > a["spot"] else "PUT"
        base = dict(name=name, sym=sym, side=side, score=a["score"])
        opt = pick_option(sym, a["spot"], side, a["hv"])
        if not opt:
            return dict(base, error="no liquid contract on the next expiry")
        plan = trade_plan(opt, a)
        if not plan:
            return dict(base, error="no positive reward on this contract")
        return dict(base, strong=abs(a["score"]) >= STRONG_SCORE, weak=abs(a["score"]) < MIN_SCORE,
                    low_rr=plan["rr"] < MIN_RR, earnings=earnings_before(sym, opt["expiry"]), opt=opt, plan=plan,
                    a=dict(spot=a["spot"], pred_close=a["pred_close"], pred_high=a["pred_high"],
                           pred_low=a["pred_low"], rsi=a["rsi"], rvol=a["rvol"], gap=a["gap"], tags=list(a["tags"])))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(one, TOP10))


def scan_once(mode, pre_prev=None, label=None):
    now = dt.datetime.now(ET)
    refresh_weights()
    mkt = market_bias(mode)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        analysed = [a for a in ex.map(lambda s: analyse(s, mode, pre_prev, mkt), SYMBOLS) if a]

    table = [dict(symbol=a["sym"], price=round(a["spot"], 2), gap_pct=round(a["gap"], 2), rsi=round(a["rsi"], 1),
                  rvol=round(a["rvol"], 2), score=a["score"], price_action=a["pa_s"], pre_score=a["pre_score"],
                  structure=a["pa"]["struct"], day_est=round(a["pred_close"], 2)) for a in analysed]

    cands = sorted([a for a in analysed if abs(a["score"]) >= MIN_SCORE and (not REQUIRE_PA or pa_agrees(a))
                    and (mode == "pre" or a["rvol"] >= MIN_RVOL)],
                   key=lambda a: -abs(a["score"]))[:CANDIDATES]

    def build(a):
        side = "CALL" if a["score"] > 0 else "PUT"
        opt = pick_option(a["sym"], a["spot"], side, a["hv"])
        if not opt:
            return None
        if SKIP_EARNINGS and earnings_before(a["sym"], opt["expiry"]):
            return None
        plan = trade_plan(opt, a)
        if not plan or plan["rr"] < MIN_RR:
            return None
        rank = abs(a["score"]) * 10 + opt["quality"] * 0.25 + min(plan["rr"], 3.0) * 5
        return dict(side=side, a=a, opt=opt, plan=plan, rank=rank, strong=abs(a["score"]) >= STRONG_SCORE)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        built = [b for b in ex.map(build, cands) if b]

    res = {"mode": mode, "label": label or {"pre": "Premarket", "open": "Open 09:45", "hourly": "Hourly"}[mode],
           "time": now.isoformat(timespec="seconds"), "CALL": None, "PUT": None, "top": None, "market": mkt,
           "table": table, "scores": {a["sym"]: a["score"] for a in analysed}, "scanned": len(analysed),
           "learned": len(LEARNED_W)}
    for b in built:
        if res[b["side"]] is None or b["rank"] > res[b["side"]]["rank"]:
            res[b["side"]] = b
    try:
        res["top10"] = build_top10(analysed)              # CALL / PUT + Moomoo values for each of the top 10 shares
    except Exception:
        res["top10"] = []
    picks = [s for s in ("CALL", "PUT") if res[s]]
    if picks:
        res["top"] = max(picks, key=lambda s: res[s]["rank"])
    log_picks(res)
    try:
        record_trades(built, res, now)                    # every candidate is tracked, results fetched later
    except Exception:
        pass
    return res

# ----------------------------- SCHEDULE -----------------------------

def _done(state, key, today):
    return state.get("date") == today and state.get(key) is not None


def due_scan(now, state):
    """The SCHEDULE entry that should start now (and has not run yet today), else None."""
    if now.weekday() >= 5:
        return None
    t, today = now.time(), str(now.date())
    for entry in SCHEDULE:
        key, _, start, latest, _ = entry
        if start <= t < latest and not _done(state, key, today):
            return entry
    return None


def next_scan(now):
    for i in range(8):
        day = now.date() + dt.timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for entry in SCHEDULE:
            when = dt.datetime.combine(day, entry[2], tzinfo=ET)
            if when > now:
                return entry, when
    return None, None


def pre_scores_today(state):
    today = str(dt.datetime.now(ET).date())
    pre = state.get("pre")
    if state.get("date") == today and pre and pre.get("scores"):
        return pre["scores"]
    return None


def run_scheduled(cfg):
    """Runs a due scan once (state file stops repeats, also across several browser tabs)."""
    now = dt.datetime.now(ET)
    state = load_state()
    entry = due_scan(now, state)
    if not entry:
        return None
    key, mode, _, _, label = entry
    today = str(now.date())
    if state.get("date") != today:
        state = {"date": today}
    state[key] = {"running": True, "time": now.isoformat(timespec="seconds"), "label": label}   # claim it
    save_state(state)
    try:
        res = scan_once(mode, pre_scores_today(state) if mode != "pre" else None, label)
        state[key] = res
        save_state(state)
        res["telegram"] = send_telegram(cfg, telegram_text(res))
        try:                                                    # second message: BUY / SELL for indices, commodities, top 10
            send_telegram(cfg, market_signals_text())
        except Exception:
            pass
        for sd in ("CALL", "PUT"):                              # Moomoo ticket picture with the exact numbers
            it = res.get(sd)
            if it:
                try:
                    send_telegram_photo(cfg, ticket_png(it), f"{sd} {it['opt']['symbol']} - values to enter in Moomoo")
                except Exception:
                    pass
        state[key] = res
        save_state(state)
    except Exception as e:
        state[key] = {"error": str(e), "time": now.isoformat(timespec="seconds"), "mode": mode, "label": label}
        save_state(state)
    return key

# ----------------------------- BACKTEST -----------------------------
# yfinance only keeps 60 days of 5-minute bars and NO history of option quotes, so the backtest:
#   1. replays every scan of the SCHEDULE on past days with the data available at that moment (no look-ahead),
#   2. models the option: strike nearest delta 0.65, next Friday expiry, IV = 20-day HV x BT_IV_MULT, Black-Scholes,
#   3. enters at the open of the bar after the scan, plus slippage,
#   4. walks forward bar by bar: stop loss checked first (worst case) then take profit, otherwise exit 15:45 ET,
#   5. runs the same scans twice - WITH price action and WITHOUT - so you can see what it adds.
# The SPY market filter and volume filter are replayed too. The earnings filter is live-only (no history).
# The backtest never uses the learned weights (they come from the live results, so using them here would be circular).

def fetch_history(symbols, progress=None):
    """{symbol: (5-min bars incl. premarket, last 60 days ; daily bars, 6 months)}"""
    out = {}

    def one(sym):
        # worker thread: download only - no Streamlit calls here (they fail outside the script thread)
        try:
            df = _clean_intraday(yf.Ticker(sym).history(period="60d", interval="5m", prepost=True))
            dd = fetch_daily(sym, period="6mo")
            if df is not None and dd is not None and len(df) > 200:
                return sym, (df, dd)
        except Exception:
            pass
        return sym, None

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, s) for s in symbols]
        for n, fut in enumerate(as_completed(futures), 1):      # progress is reported from the main thread
            sym, val = fut.result()
            if val is not None:
                out[sym] = val
            if progress:
                progress(n / len(symbols), f"Downloading history {n}/{len(symbols)}")
    return out


def _next_friday(day):
    d = day + dt.timedelta(days=1)
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    return d


def _strike_step(S):
    return 0.5 if S < 25 else (1.0 if S < 100 else (2.5 if S < 200 else 5.0))


def synth_option(sym, S, side, hv, now, iv_mult, slip):
    """Modelled option: strike with delta closest to TARGET_DELTA, next Friday expiry (never same day)."""
    exp_day = _next_friday(now.date())
    exp_dt = dt.datetime.combine(exp_day, MARKET_CLOSE, tzinfo=ET)
    T = (exp_dt - now).total_seconds() / (365 * 86400)
    iv = float(np.clip(hv * iv_mult, 0.10, 2.0))
    step = _strike_step(S)
    base = round(S / step) * step
    best = None
    for k in range(-30, 31):
        K = base + k * step
        if K <= 0:
            continue
        g = greeks(S, K, T, iv, side)
        if g and DELTA_MIN <= abs(g["delta"]) <= DELTA_MAX:
            dist = abs(abs(g["delta"]) - TARGET_DELTA)
            if best is None or dist < best[0]:
                best = (dist, K, g)
    if best is None:
        return None
    _, K, g = best
    px = bs_price(S, K, T, iv, side)
    if px < 0.20:
        return None
    return dict(symbol=sym, side=side, strike=float(K), expiry=str(exp_day), exp_dt=exp_dt, T=T, iv=iv,
                entry=px * (1 + slip), delta=g["delta"])


def simulate_trade(opt, plan, bars, slip, offset=0.0):
    """Walk the 5-min bars forward. Returns (exit_option_price, reason, exit_time) or None.
       offset = dollars added to the Black-Scholes value (used when settling live trades, so the modelled price
       starts exactly at the real entry price; 0 in the backtest)."""
    if bars is None or bars.empty:
        return None
    side, K, iv, exp_dt = opt["side"], opt["strike"], opt["iv"], opt["exp_dt"]
    tp, sl = plan["tp"], plan["sl"]

    def val(S, t_end):
        T = max((exp_dt - t_end).total_seconds() / (365 * 86400), 1.0 / (365 * 24))
        return max(bs_price(S, K, T, iv, side) + offset, 0.0)

    last_ts = None
    for ts, b in zip(bars.index, bars.itertuples()):
        t_end = ts + pd.Timedelta(minutes=5)
        v_open = val(b.open, ts)
        if v_open <= sl:                                       # gapped through the stop
            return v_open * (1 - slip), "SL", ts
        if v_open >= tp:                                       # gapped through the target
            return v_open, "TP", ts
        worst, best = (b.low, b.high) if side == "CALL" else (b.high, b.low)
        if val(worst, t_end) <= sl:                            # stop first when both are inside one bar
            return sl * (1 - slip), "SL", ts
        if val(best, t_end) >= tp:
            return tp, "TP", ts
        last_ts, last_close, last_end = ts, b.close, t_end
    return val(last_close, last_end) * (1 - slip), "TIME", last_ts


def prepare_backtest(data, days=20, progress=None):
    """The slow part, done once: every stock's indicator + price-action read at every scan time of the last
    `days` trading days (cut at the scan time, so no look-ahead). run_backtest() and the sweep reuse it."""
    if not data:
        return []
    today_et = dt.datetime.now(ET).date()
    ref = max(data.values(), key=lambda x: len(x[0]))[0]
    regr = ref[np.array([x >= MARKET_OPEN for x in ref.index.time])]
    full = [d for d, g in regr.groupby(regr.index.date)
            if d < today_et and len(g) >= 70 and g.index[-1].time() >= dt.time(15, 30)]
    test_days = full[-days:]
    prep = []
    for di, day in enumerate(test_days):
        exit_ts = pd.Timestamp(dt.datetime.combine(day, EXIT_TIME), tz=ET)
        for key, mode, start, _, label in SCHEDULE:
            when = pd.Timestamp(dt.datetime.combine(day, start), tz=ET)
            entry_ts = max(when, pd.Timestamp(dt.datetime.combine(day, MARKET_OPEN), tz=ET))
            base = []
            for sym, (df, dd) in data.items():
                if sym == MARKET_SYM and MARKET_SYM not in SYMBOLS:      # SPY is only the market filter
                    continue
                i = df.index.searchsorted(when)
                a0 = analyse_df(sym, df.iloc[max(0, i - 720):i], dd, mode)
                if a0:
                    base.append(a0)
            mkt = 0
            if USE_MARKET_FILTER and MARKET_SYM in data:
                df_m, dd_m = data[MARKET_SYM]
                i = df_m.index.searchsorted(when)
                mkt = mkt_dir(analyse_df(MARKET_SYM, df_m.iloc[max(0, i - 720):i], dd_m, mode))
            prep.append(dict(day=day, mode=mode, label=label, entry_ts=entry_ts, exit_ts=exit_ts, base=base, mkt=mkt))
        if progress:
            progress((di + 1) / len(test_days), f"Reading signals, day {di + 1}/{len(test_days)} ({day})")
    return prep


def run_backtest(data, days=20, iv_mult=BT_IV_MULT, slip=BT_SLIPPAGE, commission=BT_COMMISSION, progress=None,
                 prep=None, params=None, variants=None):
    """One row per trade. params (all optional) override tp_frac, sl_frac, min_loss, max_loss, min_score, min_rr,
    require_pa, min_rvol. variants = {name: use_price_action}; default compares with / without price action."""
    p = dict(tp_frac=TP_FRAC, sl_frac=SL_MOVE_FRAC, min_loss=MIN_LOSS, max_loss=MAX_LOSS,
             min_score=MIN_SCORE, min_rr=MIN_RR, require_pa=REQUIRE_PA, min_rvol=MIN_RVOL, random_dir=None)
    p.update(params or {})
    variants = variants or {"Price action": True, "Indicators only": False}
    if prep is None:
        prep = prepare_backtest(data, days, progress)
    rows, cur_day, pre_scores, taken = [], None, {}, set()
    rnd = np.random.default_rng(p["random_dir"]) if p["random_dir"] is not None else None   # control run: coin-flip direction

    for sc in prep:
        day, mode, label, entry_ts, exit_ts = sc["day"], sc["mode"], sc["label"], sc["entry_ts"], sc["exit_ts"]
        if day != cur_day:
            cur_day, pre_scores, taken = day, {v: {} for v in variants}, set()
        for vname, use_pa in variants.items():
            fin = [finalize(a0, use_pa, pre_scores[vname] if mode != "pre" else None, sc.get("mkt", 0))
                   for a0 in sc["base"]]
            if mode == "pre":
                pre_scores[vname] = {a["sym"]: a["score"] for a in fin}
            cands = sorted([a for a in fin if abs(a["score"]) >= p["min_score"]
                            and not (p["require_pa"] and use_pa and not pa_agrees(a))
                            and (mode == "pre" or a["rvol"] >= p["min_rvol"])],
                           key=lambda a: -abs(a["score"]))[:CANDIDATES]
            best = {"CALL": None, "PUT": None}
            for a in cands:
                side = "CALL" if a["score"] > 0 else "PUT"
                if rnd is not None:                                  # same stocks and exits, random direction
                    flip = "CALL" if rnd.random() < 0.5 else "PUT"
                    if flip != side:
                        a = dict(a, sl_ref=None)
                    side = flip
                if (vname, a["sym"], side, day) in taken:
                    continue
                df = data[a["sym"]][0]
                j0, j1 = df.index.searchsorted(entry_ts), df.index.searchsorted(exit_ts)
                bars = df.iloc[j0:j1]
                if len(bars) < 3:
                    continue
                S_in = float(bars.iloc[0]["open"])
                opt = synth_option(a["sym"], S_in, side, a["hv"], entry_ts.to_pydatetime(), iv_mult, slip)
                if not opt:
                    continue
                plan = trade_plan(opt, dict(a, spot=S_in), p["tp_frac"], p["sl_frac"], p["min_loss"], p["max_loss"])
                if not plan or plan["rr"] < p["min_rr"]:
                    continue
                rank = abs(a["score"]) * 10 + min(plan["rr"], 3.0) * 5
                if best[side] is None or rank > best[side]["rank"]:
                    best[side] = dict(a=a, opt=opt, plan=plan, bars=bars, rank=rank, side=side)

            for side, b in best.items():
                if not b:
                    continue
                a, opt, plan = b["a"], b["opt"], b["plan"]
                sim = simulate_trade(opt, plan, b["bars"], slip)
                if not sim:
                    continue
                taken.add((vname, a["sym"], side, day))
                px_out, reason, t_out = sim
                pnl = (px_out - plan["entry"]) * CONTRACT_SIZE - 2 * commission
                pa_dir = 0 if a["pa_s"] == 0 else (1 if a["pa_s"] > 0 else -1)
                rows.append(dict(variant=vname, date=str(day), scan=label, symbol=a["sym"], side=side,
                                 score=a["score"], strong=abs(a["score"]) >= STRONG_SCORE,
                                 pa_points=a["pa_s"], pa_agrees=pa_dir == (1 if side == "CALL" else -1),
                                 tags=" ".join(a["tags"]), strike=opt["strike"], expiry=opt["expiry"],
                                 entry=plan["entry"], tp=plan["tp"], sl=plan["sl"], exit=round(px_out, 2),
                                 reason=reason, exit_time=t_out.isoformat(), pnl=round(pnl, 2),
                                 pnl_pct=round((px_out / plan["entry"] - 1) * 100, 1)))
    return pd.DataFrame(rows)


# ----- sweep: try many exit / filter settings, tune on the EARLY days, check them on the LATER days -----
SWEEP_GRID = dict(tp_frac=[0.4, 0.7, 1.0],            # target distance (x expected move)
                  sl_frac=[0.5, 1.0, 1.5],            # stop distance (x expected move)
                  max_loss=[0.30, 0.50],              # option stop never wider than this share of the premium
                  min_score=[3, 5],                   # signal strength needed
                  mode=["Indicators only", "Price action", "PA must agree"])


def run_sweep(data, prep, iv_mult=BT_IV_MULT, slip=BT_SLIPPAGE, commission=BT_COMMISSION, train_frac=0.6,
              progress=None):
    """Chronological split: the first `train_frac` of the days is for choosing, the rest only for checking.
    A setting that wins on 'train' but loses on 'test' was just fitted to noise."""
    days = sorted({str(s["day"]) for s in prep})
    if len(days) < 6:
        return pd.DataFrame()
    split = days[max(1, int(len(days) * train_frac))]            # first test day
    combos = list(itertools.product(SWEEP_GRID["tp_frac"], SWEEP_GRID["sl_frac"], SWEEP_GRID["max_loss"],
                                    SWEEP_GRID["min_score"], SWEEP_GRID["mode"]))
    out = []
    for n, (tp, sl, ml, ms, mode) in enumerate(combos, 1):
        params = dict(tp_frac=tp, sl_frac=sl, max_loss=ml, min_score=ms, min_rr=0.3,
                      require_pa=(mode == "PA must agree"))
        tr = run_backtest(data, iv_mult=iv_mult, slip=slip, commission=commission, prep=prep, params=params,
                          variants={"v": mode != "Indicators only"})
        if not tr.empty:
            row = dict(mode=mode, tp_frac=tp, sl_frac=sl, max_loss=ml, min_score=ms)
            # control: identical stocks / exits but random call-or-put (2 seeds) - the signals must beat this
            ctrl = pd.concat([run_backtest(data, iv_mult=iv_mult, slip=slip, commission=commission, prep=prep,
                                           params=dict(params, random_dir=sd), variants={"v": mode != "Indicators only"})
                              for sd in (1, 2)])
            ct = ctrl[ctrl["date"] >= split] if not ctrl.empty else ctrl
            row["random_test_pnl$"] = round(ct["pnl"].sum() / 2, 0) if not ct.empty else 0.0
            row["random_test_win%"] = round((ct["pnl"] > 0).mean() * 100, 1) if not ct.empty else 0.0
            for name, part in (("train", tr[tr["date"] < split]), ("test", tr[tr["date"] >= split])):
                s = summarize(part)
                row[f"{name}_trades"] = s.get("Trades", 0)
                row[f"{name}_win%"] = s.get("Win rate %", 0.0)
                row[f"{name}_pf"] = s.get("Profit factor", 0.0)
                row[f"{name}_pnl$"] = s.get("Total P&L $", 0.0)
            out.append(row)
        if progress:
            progress(n / len(combos), f"Sweep {n}/{len(combos)}")
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # strict rule (calibrated on random, zero-edge data where it passed 0-4 of 108 settings, vs ~40 when a real
    # edge exists): profitable with PF >= 1.5 on BOTH halves, enough trades, and the test half beats the same exits
    # with a random call/put by a wide margin (otherwise the exits - not the signals - made the money)
    df["beats_random"] = df["test_pnl$"] > df["random_test_pnl$"] + 1.0 * df["test_pnl$"].abs()
    df["holds_up"] = ((df["train_pnl$"] > 0) & (df["test_pnl$"] > 0) & (df["train_pf"] >= 1.5) & (df["test_pf"] >= 1.5)
                      & (df["train_trades"] >= 30) & (df["test_trades"] >= 30) & df["beats_random"])
    df["split_date"] = split
    return df.sort_values("train_pf", ascending=False).reset_index(drop=True)


def summarize(tr):
    """Headline numbers for a set of trades (pnl in $ per 1 lot, after slippage and commission)."""
    if tr is None or tr.empty:
        return {}
    tr = tr.sort_values("exit_time")
    win, loss = tr[tr["pnl"] > 0]["pnl"], tr[tr["pnl"] <= 0]["pnl"]
    eq = pd.concat([pd.Series([0.0]), tr["pnl"].cumsum().reset_index(drop=True)])
    gl = abs(loss.sum())
    return {"Trades": len(tr),
            "Win rate %": round(len(win) / len(tr) * 100, 1),
            "Total P&L $": round(tr["pnl"].sum(), 0),
            "Avg P&L $": round(tr["pnl"].mean(), 1),
            "Avg win $": round(win.mean(), 1) if len(win) else 0.0,
            "Avg loss $": round(loss.mean(), 1) if len(loss) else 0.0,
            "Profit factor": round(win.sum() / gl, 2) if gl > 0 else float("inf"),
            "Max drawdown $": round((eq - eq.cummax()).min(), 0),
            "TP hits": int((tr["reason"] == "TP").sum()),
            "SL hits": int((tr["reason"] == "SL").sum()),
            "Time exits": int((tr["reason"] == "TIME").sum())}


def _breakdown(tr, col):
    g = tr.groupby(col)["pnl"]
    out = pd.DataFrame({"trades": g.size(), "win %": g.apply(lambda x: round((x > 0).mean() * 100, 1)),
                        "total $": g.sum().round(0), "avg $": g.mean().round(1)})
    return out

# ----------------------------- TELEGRAM TEXT (compact) -----------------------------

def _bias_txt(m):
    return "bullish" if m > 0 else ("bearish" if m < 0 else "neutral")


def telegram_text(res):
    when = dt.datetime.fromisoformat(res["time"])
    title = "PREMARKET (indicative)" if res["mode"] == "pre" else res.get("label", "SCAN").upper()
    lines = [f"📊 {title} · {when.strftime('%H:%M')} ET · market {_bias_txt(res.get('market', 0))}"]
    shown = False
    for side in ("CALL", "PUT"):
        it = res.get(side)
        if not it:
            continue
        shown = True
        o, p, a = it["opt"], it["plan"], it["a"]
        star = "⭐ " if res.get("top") == side else ""
        lines += [
            f"{star}{'🟢' if side == 'CALL' else '🔴'} {side} {o['symbol']} {o['strike']:g} exp {o['expiry']}",
            f"Entry {p['entry']} | SL {p['sl']} | TP {p['tp']} (+${p['profit_1lot']:.0f}/lot)",
            f"Δ{o['delta']:.2f} Γ{o['gamma']:.3f} Θ{o['theta']:.2f} V{o['vega']:.2f} ρ{o['rho']:.2f} IV{o['iv'] * 100:.0f}%",
            f"Stock {a['spot']:.2f} → day est {a['pred_close']:.2f} (H {a['pred_high']:.2f} / L {a['pred_low']:.2f})",
        ]
        lines.append(f"Chart: {tv_url(o['symbol'])}")
        pa_txt = " ".join(t.replace("PA:", "") for t in a["pa_tags"] + (["PA:blocked"] if "PA:blocked" in a["tags"] else []))
        if pa_txt:
            lines.append(f"Price action: {a['pa']['struct']} | {pa_txt}")
    if not shown:
        lines.append("No qualifying setup this scan.")
    return "\n".join(lines)

# ----------------------------- DISPLAY -----------------------------

def fmt_local(when):
    return when.astimezone(LOCAL_TZ).strftime("%a %d %b %H:%M")


def tv_url(sym, interval="5"):
    """TradingView chart for the stock (5-minute by default). Yahoo's BRK-B style becomes BRK.B."""
    return f"https://www.tradingview.com/chart/?symbol={sym.replace('-', '.')}&interval={interval}"


# ----- market panels: indices list + commodities list + top 10 shares -----
def get_quote(sym):
    """{price, chg, pct} vs the previous close, or None."""
    px = prev = None
    try:
        fi = yf.Ticker(sym).fast_info
        px, prev = float(fi["last_price"]), float(fi["previous_close"])
    except Exception:
        px = prev = None
    if not px or not prev or px != px or prev != prev:
        try:
            d = yf.Ticker(sym).history(period="5d", interval="1d")["Close"].dropna()
            if len(d) < 2:
                return None
            px, prev = float(d.iloc[-1]), float(d.iloc[-2])
        except Exception:
            return None
    return dict(price=px, chg=px - prev, pct=(px / prev - 1) * 100)


@st.cache_data(ttl=MARKET_REFRESH_SEC, show_spinner=False)
def market_quotes(symbols):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return dict(zip(symbols, ex.map(get_quote, symbols)))


def _metric_grid(items, quotes, ncol=5):
    for i in range(0, len(items), ncol):
        cols = st.columns(ncol)
        for c, (name, sym) in zip(cols, items[i:i + ncol]):
            d = quotes.get(sym)
            with c:
                if not d:
                    st.metric(name, "n/a")
                    continue
                st.metric(name, f"{d['price']:,.2f}", f"{d['chg']:+,.2f} ({d['pct']:+.2f}%)",
                          delta_color="inverse" if sym == "^VIX" else "normal")


def _list_html(items, quotes):
    """Compact list view: name | price | change | % (green up / red down; the VIX is reversed)."""
    td = "padding:9px 8px;border-bottom:1px solid rgba(128,128,128,0.25)"
    rows = []
    for name, sym in items:
        d = quotes.get(sym)
        if not d:
            rows.append(f'<tr><td style="{td}">{name}</td>'
                        f'<td colspan="3" style="{td};text-align:right;opacity:0.6">n/a</td></tr>')
            continue
        good = d["chg"] >= 0
        if sym == "^VIX":
            good = not good
        col = "#2fbf71" if good else "#ef4f5f"
        rows.append(f'<tr><td style="{td}">{name}</td>'
                    f'<td style="{td};text-align:right">{d["price"]:,.2f}</td>'
                    f'<td style="{td};text-align:right;color:{col}">{d["chg"]:+,.2f}</td>'
                    f'<td style="{td};text-align:right;color:{col};font-weight:600">{d["pct"]:+.2f}%</td></tr>')
    head = ('<tr style="opacity:0.6;font-size:14px"><td style="padding:2px 8px">Name</td>'
            '<td style="padding:2px 8px;text-align:right">Price</td>'
            '<td style="padding:2px 8px;text-align:right">Change</td>'
            '<td style="padding:2px 8px;text-align:right">%</td></tr>')
    return f'<table style="width:100%;border-collapse:collapse;font-size:16px">{head}{"".join(rows)}</table>'


# ----- BUY / SELL signals for the market panels (US indices, commodities, top-10 shares) -----
# Supertrend + EMA 9/15 + BOS / CHoCH market structure + supply & demand zones, read from 1-hour bars.
# "Next 1h" is an estimate (last price +/- a share of the 1-hour ATR), not a promise. Educational only.
SIG_PERIOD, SIG_INTERVAL = "3mo", "1h"
ST_PERIOD, ST_MULT = 10, 3.0                 # Supertrend settings
SIG_SWING_K = 3                              # a swing high / low = extreme of 3 bars either side
SIG_FRESH_BARS = 12                          # a CHoCH this recent counts as a fresh reversal (+1 extra point)
SIG_ZONE_PAD = 0.3                           # a zone counts when price is inside it or within 0.3 x ATR of it
SIG_BUY, SIG_SELL = 3, -3                    # score needed for BUY / SELL (score runs -5 ... +5), otherwise WAIT
SIG_REFRESH_SEC = 300


def supertrend(df, period=ST_PERIOD, mult=ST_MULT):
    """Supertrend line and direction (+1 up / -1 down) for every bar."""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    a = atr(df, period).to_numpy()
    ub, lb = hl2 + mult * a, hl2 - mult * a
    c = df["close"].to_numpy()
    n = len(c)
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(n, dtype=int)
    line = np.empty(n)
    d[0] = 1 if c[0] >= hl2[0] else -1
    line[0] = lb[0] if d[0] == 1 else ub[0]
    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        d[i] = d[i - 1]
        if d[i - 1] == 1 and c[i] < fl[i]:
            d[i] = -1
        elif d[i - 1] == -1 and c[i] > fu[i]:
            d[i] = 1
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return d, line


def structure_events(df, k=SIG_SWING_K):
    """Walks the bars once (no look-ahead: a swing is only known k bars after it forms).
       A close beyond the last swing high / low in the direction of the trend = BOS (break of structure).
       A close beyond the last swing AGAINST the trend = CHoCH (change of character) and the trend flips.
       Each break leaves a zone: bullish break -> DEMAND zone at the last swing low (wick to body top),
       bearish break -> SUPPLY zone at the last swing high (body bottom to wick). A zone is dropped once price
       closes through it. Returns trend, last event (kind, direction, bar index) and the open zones."""
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    op, cl = df["open"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    trend, ev = 0, None
    last_sh = last_sl = None
    sh_done = sl_done = False
    demand, supply = [], []
    for i in range(n):
        j = i - k
        if j >= k:
            w_hi, w_lo = hi[j - k:j + k + 1], lo[j - k:j + k + 1]
            if hi[j] == w_hi.max() and (w_hi == hi[j]).sum() == 1:
                last_sh, sh_done = (j, float(hi[j])), False
            if lo[j] == w_lo.min() and (w_lo == lo[j]).sum() == 1:
                last_sl, sl_done = (j, float(lo[j])), False
        if last_sh and not sh_done and cl[i] > last_sh[1]:
            ev = ("BOS" if trend >= 0 else "CHoCH", 1, i)
            trend, sh_done = 1, True
            if last_sl:
                m = last_sl[0]
                demand.append((float(lo[m]), float(max(op[m], cl[m])), m))
        elif last_sl and not sl_done and cl[i] < last_sl[1]:
            ev = ("BOS" if trend <= 0 else "CHoCH", -1, i)
            trend, sl_done = -1, True
            if last_sh:
                m = last_sh[0]
                supply.append((float(min(op[m], cl[m])), float(hi[m]), m))
        demand = [z for z in demand if cl[i] >= z[0]][-5:]
        supply = [z for z in supply if cl[i] <= z[1]][-5:]
    return dict(trend=trend, ev=ev, bars_ago=(n - 1 - ev[2]) if ev else None, demand=demand, supply=supply)


def signal_for(sym):
    """BUY / SELL / WAIT for one symbol from its 1-hour bars (None when there is no data).
       Score: Supertrend +-1, EMA9 vs EMA15 +-1, structure trend +-1 (+-1 more for a fresh CHoCH), price inside /
       next to a demand zone +1 or a supply zone -1.  BUY >= +3, SELL <= -3, else WAIT; |score| >= 4 = strong."""
    try:
        df = yf.Ticker(sym).history(period=SIG_PERIOD, interval=SIG_INTERVAL)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        if len(df) < 60:
            return None
        spot = float(df["close"].iloc[-1])
        a1 = max(float(atr(df).iloc[-1]), 1e-9)
        d, line = supertrend(df)
        e9, e15 = float(ema(df["close"], 9).iloc[-1]), float(ema(df["close"], 15).iloc[-1])
        se = structure_events(df)

        s, tags = 0, []
        s += int(d[-1])
        tags.append("ST" + ("↑" if d[-1] > 0 else "↓"))
        s += 1 if e9 > e15 else -1
        tags.append("EMA9" + (">" if e9 > e15 else "<") + "15")
        ev = se["ev"]
        if ev:
            s += se["trend"]
            if ev[0] == "CHoCH" and se["bars_ago"] <= SIG_FRESH_BARS:
                s += ev[1]
            tags.append(ev[0] + ("↑" if ev[1] > 0 else "↓"))
        pad = SIG_ZONE_PAD * a1
        in_dem = any(z[0] - pad <= spot <= z[1] + pad for z in se["demand"])
        in_sup = any(z[0] - pad <= spot <= z[1] + pad for z in se["supply"])
        if in_dem and not in_sup:
            s += 1
            tags.append("Demand")
        elif in_sup and not in_dem:
            s -= 1
            tags.append("Supply")

        label = "BUY" if s >= SIG_BUY else ("SELL" if s <= SIG_SELL else "WAIT")
        est = spot + (1 if s > 0 else -1) * a1 * (0.3 + 0.1 * abs(s)) if label != "WAIT" else spot
        return dict(sig=label, score=int(s), strong=abs(s) >= 4, tags=tags, spot=spot, est=float(est),
                    st_line=float(line[-1]))
    except Exception:
        return None


@st.cache_data(ttl=SIG_REFRESH_SEC, show_spinner=False)
def market_signals(symbols):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return dict(zip(symbols, ex.map(signal_for, symbols)))


def market_signals_text():
    """Compact BUY / SELL list for the US indices, commodities and top-10 shares (second Telegram message)."""
    groups = (("US indices", INDICES), ("Commodities", COMMODITIES), ("Top 10 US shares", [(sy, sy) for _, sy in TOP10]))
    syms = tuple(dict.fromkeys(s for _, items in groups for _, s in items))
    sg = market_signals(syms)
    lines = ["📈 MARKET SIGNALS (1h: Supertrend + EMA 9/15 + BOS/CHoCH + supply/demand)"]
    for title, items in groups:
        lines.append(f"\n{title}")
        for name, sym in items:
            s = sg.get(sym)
            if not s:
                continue
            vix = sym == "^VIX"
            if s["sig"] == "WAIT":
                lines.append(f"⚪ {name}: WAIT {s['spot']:,.2f}")
                continue
            up = s["sig"] == "BUY"
            icon = "🟢" if (up != vix) else "🔴"
            lab = ("RISK-OFF" if up else "RISK-ON") if vix else s["sig"] + ("+" if s["strong"] else "")
            lines.append(f"{icon} {name}: {lab} {s['spot']:,.2f} → {s['est']:,.2f} ({' '.join(s['tags'])})")
    lines.append("\nEstimates for the next hour, not advice. + = strong signal.")
    return "\n".join(lines)[:4000]


def _sig_badge(s, sym):
    """Coloured BUY / SELL / WAIT label. For the VIX, 'BUY' means fear is rising, so it reads RISK-OFF (red)."""
    if not s:
        return '<span style="opacity:0.5">n/a</span>'
    lab, vix = s["sig"], sym == "^VIX"
    if lab == "WAIT":
        bg, txt = "#6b7280", "WAIT"
    else:
        up = lab == "BUY"
        bg = "#2fbf71" if (up != vix) else "#ef4f5f"
        txt = ("RISK-OFF" if up else "RISK-ON") if vix else lab + ("+" if s["strong"] else "")
    return (f'<span style="background:{bg};color:#fff;border-radius:4px;padding:2px 7px;font-size:12px;'
            f'font-weight:700;white-space:nowrap">{txt}</span>')


def _signal_list_html(items, quotes, sigs):
    """List view: name | price | % | signal (+ the reasons in small print) | estimate for the next hour."""
    td = "padding:7px 6px;border-bottom:1px solid rgba(128,128,128,0.25);vertical-align:top"
    rows = []
    for name, sym in items:
        d, s = quotes.get(sym), sigs.get(sym)
        if not d:
            rows.append(f'<tr><td style="{td}">{name}</td>'
                        f'<td colspan="4" style="{td};text-align:right;opacity:0.6">n/a</td></tr>')
            continue
        good = d["chg"] >= 0
        if sym == "^VIX":
            good = not good
        col = "#2fbf71" if good else "#ef4f5f"
        if s:
            tip = f'score {s["score"]:+d} of 5 | Supertrend line {s["st_line"]:,.2f}'
            sig_cell = (f'<div title="{tip}">{_sig_badge(s, sym)}</div>'
                        f'<div style="font-size:11px;opacity:0.65;margin-top:2px">{" · ".join(s["tags"])}</div>')
            est = d["price"] + (s["est"] - s["spot"])
            pct = (est / d["price"] - 1) * 100 if d["price"] else 0.0
            ecol = "#8e939c" if s["sig"] == "WAIT" else ("#2fbf71" if pct >= 0 else "#ef4f5f")
            est_cell = f'{est:,.2f}<div style="font-size:11px;color:{ecol}">{pct:+.2f}%</div>'
        else:
            sig_cell, est_cell = '<span style="opacity:0.5">n/a</span>', ""
        rows.append(f'<tr><td style="{td}">{name}</td>'
                    f'<td style="{td};text-align:right">{d["price"]:,.2f}</td>'
                    f'<td style="{td};text-align:right;color:{col};font-weight:600">{d["pct"]:+.2f}%</td>'
                    f'<td style="{td}">{sig_cell}</td>'
                    f'<td style="{td};text-align:right">{est_cell}</td></tr>')
    head = ('<tr style="opacity:0.6;font-size:12px"><td style="padding:2px 6px">Name</td>'
            '<td style="padding:2px 6px;text-align:right">Price</td>'
            '<td style="padding:2px 6px;text-align:right">%</td>'
            '<td style="padding:2px 6px">Signal (1h)</td>'
            '<td style="padding:2px 6px;text-align:right">Next 1h</td></tr>')
    return f'<table style="width:100%;border-collapse:collapse;font-size:14px">{head}{"".join(rows)}</table>'


def render_market(idx_box, com_box, sh_box):
    """US indices and commodities stacked in column 2 (commodities below), top-10 shares at the bottom of column 1."""
    top10 = [(f"{n} ({sy})", sy) for n, sy in TOP10]
    syms = tuple(dict.fromkeys([s for _, s in INDICES] + [s for _, s in COMMODITIES] + [s for _, s in TOP10]))
    q = market_quotes(syms)
    try:
        sg = market_signals(syms)
    except Exception:
        sg = {}
    cap = "BUY / SELL from Supertrend + EMA 9/15 + BOS / CHoCH + supply & demand (1-hour bars)"
    panels = ((idx_box, "US indices", "price and change on the day - " + cap, INDICES),
              (com_box, "Commodities", "futures, price and change on the day - " + cap, COMMODITIES),
              (sh_box, "Top 10 US shares", "price and change on the day - " + cap, top10))
    for box, title, caption, items in panels:
        with box.container():
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.caption(caption)
                st.markdown(_signal_list_html(items, q, sg), unsafe_allow_html=True)
                if title == "US indices":
                    st.caption("Score runs -5 to +5: BUY at +3 or more, SELL at -3 or less, otherwise WAIT; "
                               "+ = strong (4-5). Next 1h = estimate, not advice. VIX: RISK-OFF = fear rising.")


# ----- live news (Yahoo Finance via yfinance) -----
NEWS_INDEX = [("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"), ("Dow", "^DJI"), ("SPY", "SPY"), ("QQQ", "QQQ")]
NEWS_REFRESH_SEC = 300


def _news_item(n, sym):
    try:
        c = n.get("content") if isinstance(n.get("content"), dict) else None
        if c:
            title = c.get("title")
            url = ((c.get("canonicalUrl") or c.get("clickThroughUrl") or {}) or {}).get("url")
            src = (c.get("provider") or {}).get("displayName")
            raw = c.get("pubDate") or c.get("displayTime")
            when = dt.datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
        else:
            title, url, src = n.get("title"), n.get("link"), n.get("publisher")
            ts = n.get("providerPublishTime")
            when = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc) if ts else None
        if not title or not url:
            return None
        return dict(title=title, url=url, src=src or "", when=when, sym=sym)
    except Exception:
        return None


def _news_one(sym):
    try:
        return [i for i in (_news_item(n, sym) for n in (yf.Ticker(sym).news or [])[:10]) if i]
    except Exception:
        return []


@st.cache_data(ttl=NEWS_REFRESH_SEC, show_spinner=False)
def news_for(symbols):
    """Newest headlines for a group of symbols, de-duplicated, newest first."""
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        lists = list(ex.map(_news_one, symbols))
    seen, out = {}, []
    for items in lists:
        for it in items:
            if it["title"] in seen:
                seen[it["title"]]["syms"].add(it["sym"])
                continue
            it["syms"] = {it["sym"]}
            seen[it["title"]] = it
            out.append(it)
    floor = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    out.sort(key=lambda i: i["when"] or floor, reverse=True)
    return out[:12]


def _ago(when):
    if not when:
        return ""
    m = int((dt.datetime.now(dt.timezone.utc) - when).total_seconds() // 60)
    return f"{m}m ago" if m < 60 else (f"{m // 60}h ago" if m < 1440 else f"{m // 1440}d ago")


NEWS_COMMOD = [("Gold", "GC=F"), ("Silver", "SI=F"), ("Crude Oil", "CL=F"), ("Natural Gas", "NG=F"),
               ("Copper", "HG=F")]


def render_news(box, pick_syms):
    """Live news in the right-hand column - one tab per group: call/put picks, indices, top-10 shares, commodities."""
    groups = [("Picks", tuple(pick_syms)), ("Indices", tuple(s for _, s in NEWS_INDEX)),
              ("Shares", tuple(s for _, s in TOP10)), ("Commodities", tuple(s for _, s in NEWS_COMMOD))]
    with box.container():
        st.markdown("### Live news")
        tabs = st.tabs([g[0] for g in groups])
        for tab, (title, syms) in zip(tabs, groups):
            with tab:
                if not syms:
                    st.caption("No call/put pick yet - news appears after the first scan.")
                    continue
                items = news_for(syms)[:8]
                if not items:
                    st.caption("No headlines available right now.")
                for it in items:
                    t = it["title"].replace("[", "(").replace("]", ")")
                    tag = ", ".join(sorted(x.replace("^", "") for x in it["syms"]))[:40]
                    st.markdown(f"[{t}]({it['url']})  \n<small>{it['src']} · {_ago(it['when'])} · {tag}</small>",
                                unsafe_allow_html=True)


# ----- Moomoo-style order ticket (dark panel, inline styles only) -----
_BG, _FIELD, _LINE = "#1b1d22", "#2a2d34", "#3b3f48"
_TXT, _MUT = "#e8eaed", "#8e939c"
_GRN, _RED, _ORG, _BLU = "#2fbf71", "#ef4f5f", "#f26b21", "#2f6fed"


def _lab(t):
    return f'<div style="color:{_MUT};font-size:12px;margin:12px 0 4px 0">{t}</div>'


def _box(inner, color=_TXT, align="left", extra=""):
    return (f'<div style="background:{_FIELD};border:1px solid {_LINE};border-radius:4px;padding:8px 10px;'
            f'color:{color};font-size:14px;text-align:{align};{extra}">{inner}</div>')


def _stepper(value):
    return (f'<div style="display:flex;align-items:center;background:{_FIELD};border:1px solid {_LINE};'
            f'border-radius:4px;color:{_TXT};font-size:15px">'
            f'<div style="padding:8px 12px;color:{_MUT}">&minus;</div>'
            f'<div style="flex:1;text-align:center">{value}</div>'
            f'<div style="padding:8px 12px;color:{_MUT}">+</div></div>')


def _check(text):
    return (f'<span style="display:inline-block;width:14px;height:14px;background:{_BLU};border-radius:2px;'
            f'color:#fff;font-size:11px;line-height:14px;text-align:center;margin-right:6px;'
            f'vertical-align:middle">&#10003;</span><span style="vertical-align:middle">{text}</span>')


def _quote_bar(o, entry):
    bid, ask = o["bid"], o["ask"]
    live = bool(o["market_open"]) and ask > bid
    pos = min(max((entry - bid) / (ask - bid), 0.0), 1.0) * 100 if live else 100.0
    mid = (bid + ask) / 2 if live else o["last"]
    f = (lambda v: f"{v:.2f}") if live else (lambda v: "--")
    fill_left, fill_w = min(pos, 50.0), abs(pos - 50.0)
    bar = (f'<div style="position:relative;height:14px;margin:8px 5px 4px 5px">'
           f'<div style="position:absolute;top:6px;left:0;right:0;height:2px;background:{_LINE}"></div>'
           f'<div style="position:absolute;top:6px;left:{fill_left}%;width:{fill_w}%;height:2px;background:{_RED}"></div>'
           f'<div style="position:absolute;top:1px;left:calc({pos}% - 6px);width:8px;height:8px;'
           f'border:2px solid {_RED};border-radius:50%;background:{_BG}"></div></div>')
    labels = (f'<div style="display:flex;justify-content:space-between;color:{_MUT};font-size:12px">'
              f'<span>Bid</span><span>Mid</span><span>Ask</span></div>'
              f'<div style="display:flex;justify-content:space-between;color:{_TXT};font-size:13px;margin-top:2px">'
              f'<span>{f(bid)}</span><span>{f(mid)}</span><span>{f(ask)}</span></div>')
    return (f'<div style="background:{_FIELD};border:1px solid {_LINE};border-radius:4px;padding:8px 10px;margin-top:10px">'
            f'{bar}{labels}</div>')


def order_ticket_html(item):
    """Looks like the Moomoo order ticket, filled with the scanner's values."""
    o, p = item["opt"], item["plan"]
    code = (o["expiry"][2:4] + o["expiry"][5:7] + o["expiry"][8:10] + f" {o['strike']:g}"
            + ("C" if o["side"] == "CALL" else "P"))
    entry_s, tp_s, sl_s = f"{p['entry']:.2f}", f"{p['tp']:.2f}", f"{p['sl']:.2f}"
    gain_pct = (p["tp"] / p["entry"] - 1) * 100
    loss_pct = (1 - p["sl"] / p["entry"]) * 100

    tabs = (f'<div style="display:flex;gap:2px;margin-top:12px">'
            f'<div style="flex:1;text-align:center;padding:7px;background:#3a3f4b;border-radius:4px 0 0 4px;'
            f'color:{_TXT};font-size:14px">Trade</div>'
            f'<div style="flex:1;text-align:center;padding:7px;background:{_FIELD};border-radius:0 4px 4px 0;'
            f'color:{_MUT};font-size:14px">Ladder</div></div>')
    contract_side = (f'<div style="display:flex;gap:10px;align-items:flex-end">'
                     f'<div style="flex:3">{_lab("Contract")}{_box(code + "<span style=float:right>&#8964;</span>")}</div>'
                     f'<div style="flex:1.4">{_lab("Side")}'
                     f'{_box("Buy<span style=float:right>&#8644;</span>", color=_GRN, extra="background:#1d3a2b;")}</div></div>')
    note_tp = f'<div style="font-size:12px;color:{_GRN};margin-top:4px">+${p["profit_1lot"]:,.0f} per lot (+{gain_pct:.0f}%)</div>'
    note_sl = f'<div style="font-size:12px;color:{_RED};margin-top:4px">-${p["loss_1lot"]:,.0f} per lot (-{loss_pct:.0f}%)</div>'
    amount = (f'<div style="display:flex;justify-content:space-between;margin-top:14px;font-size:14px">'
              f'<span style="color:{_MUT}">Amount</span>'
              f'<span style="color:{_TXT};font-weight:600">{p["cost_1lot"]:,.2f} USD(Debit)</span></div>')
    button = (f'<div style="margin-top:12px;background:{_ORG};color:#fff;text-align:center;padding:10px;'
              f'border-radius:4px;font-weight:600;font-size:14px">Enter these values in Moomoo</div>')

    html = (
        f'<div style="background:{_BG};border-radius:8px;padding:14px 16px;max-width:340px;'
        f'font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{_TXT}">'
        f'<div style="font-size:16px;font-weight:600">Trade</div>'
        f'<div style="font-size:12px;margin-top:2px"><a href="{tv_url(o["symbol"])}" target="_blank" rel="noopener" '
        f'style="color:{_BLU};text-decoration:none">{o["symbol"]} - {o["side"]} - open chart in TradingView &#8599;</a></div>'
        f'{tabs}{contract_side}'
        f'{_lab("Session")}{_box("RTH only", color=_MUT)}'
        f'{_lab("Order Type")}{_box("Limit<span style=float:right>&#8964;</span>")}'
        f'{_lab("Price")}{_stepper(entry_s)}'
        f'{_quote_bar(o, p["entry"])}'
        f'{_lab("Contractsx100 shares")}{_stepper("1")}'
        f'{_lab("TIF")}{_box("Day<span style=float:right>&#8964;</span>")}'
        f'<div style="margin-top:14px;font-size:14px">{_check("Take Profit")}&nbsp;&nbsp;&nbsp;{_check("Stop Loss")}</div>'
        f'{_lab("TP Price")}{_stepper(tp_s)}{note_tp}'
        f'{_lab("SL Price")}{_stepper(sl_s)}{note_sl}'
        f'{_lab("TIF")}{_box("GTC<span style=float:right>&#8964;</span>")}'
        f'{amount}{button}</div>'
    )
    return html


def _pil_font(size, bold=False):
    from PIL import ImageFont
    names = (["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def ticket_png(item):
    """PNG picture of the Moomoo order ticket with the exact values to type in (a replica, not a real screenshot)."""
    from PIL import Image, ImageDraw
    import io
    o, p = item["opt"], item["plan"]
    code = (o["expiry"][2:4] + o["expiry"][5:7] + o["expiry"][8:10] + f" {o['strike']:g}"
            + ("C" if o["side"] == "CALL" else "P"))
    W, pad = 720, 36
    img = Image.new("RGB", (W, 1900), _BG)
    d = ImageDraw.Draw(img)
    f_t, f_n, f_s, f_b = _pil_font(32, True), _pil_font(27), _pil_font(21), _pil_font(27, True)
    st_ = dict(y=pad)

    def txt(x, y, s_, font, fill, anchor="lm"):
        try:
            d.text((x, y), s_, font=font, fill=fill, anchor=anchor)
        except Exception:
            d.text((x, y - 14), s_, font=font, fill=fill)

    def label(s_):
        st_["y"] += 10
        txt(pad, st_["y"] + 12, s_, f_s, _MUT)
        st_["y"] += 30

    def box(s_, color=_TXT, fill=_FIELD, x0=pad, x1=W - pad, arrow=False, h=60, font=None):
        y = st_["y"]
        d.rounded_rectangle([x0, y, x1, y + h], 8, fill=fill, outline=_LINE)
        txt(x0 + 18, y + h / 2, s_, font or f_n, color)
        if arrow:
            txt(x1 - 22, y + h / 2, "v", f_s, _MUT, "mm")
        return y

    def stepper(v):
        y = st_["y"]
        d.rounded_rectangle([pad, y, W - pad, y + 64], 8, fill=_FIELD, outline=_LINE)
        txt(pad + 28, y + 32, "-", f_b, _MUT, "mm")
        txt(W - pad - 28, y + 32, "+", f_b, _MUT, "mm")
        txt(W / 2, y + 32, v, _pil_font(32, True), _TXT, "mm")
        st_["y"] += 64

    def check(x, y, s_):
        d.rounded_rectangle([x, y, x + 28, y + 28], 4, fill=_BLU)
        d.line([x + 7, y + 15, x + 12, y + 21, x + 22, y + 8], fill="white", width=3)
        txt(x + 42, y + 14, s_, f_n, _TXT)

    txt(pad, st_["y"] + 18, "Trade", f_t, _TXT)
    st_["y"] += 44
    txt(pad, st_["y"] + 10, f"{o['symbol']} - {o['side']} - values to enter", f_s, _BLU)
    st_["y"] += 36
    y = st_["y"]
    d.rounded_rectangle([pad, y, W / 2 - 2, y + 48], 8, fill="#3a3f4b")
    d.rounded_rectangle([W / 2 + 2, y, W - pad, y + 48], 8, fill=_FIELD)
    txt(pad + (W / 2 - pad) / 2, y + 24, "Trade", f_n, _TXT, "mm")
    txt(W / 2 + (W / 2 - pad) / 2, y + 24, "Ladder", f_n, _MUT, "mm")
    st_["y"] += 58

    st_["y"] += 10
    txt(pad, st_["y"] + 12, "Contract", f_s, _MUT)
    txt(490, st_["y"] + 12, "Side", f_s, _MUT)
    st_["y"] += 30
    y = st_["y"]
    d.rounded_rectangle([pad, y, 470, y + 60], 8, fill=_FIELD, outline=_LINE)
    txt(pad + 18, y + 30, code, f_b, _TXT)
    d.rounded_rectangle([490, y, W - pad, y + 60], 8, fill="#1d3a2b", outline=_LINE)
    txt(508, y + 30, "Buy", f_b, _GRN)
    st_["y"] += 60

    label("Session")
    box("RTH only", _MUT)
    st_["y"] += 60
    label("Order Type")
    box("Limit", arrow=True)
    st_["y"] += 60
    label("Price")
    stepper(f"{p['entry']:.2f}")
    st_["y"] += 12
    y = st_["y"]
    d.rounded_rectangle([pad, y, W - pad, y + 74], 8, fill=_FIELD, outline=_LINE)
    live = bool(o["market_open"]) and o["ask"] > o["bid"]
    mid = (o["bid"] + o["ask"]) / 2 if live else o["last"]
    fm = (lambda v: f"{v:.2f}") if live else (lambda v: "--")
    for x, nm, v in ((pad + 20, "Bid", o["bid"]), (W / 2 - 40, "Mid", mid), (W - pad - 110, "Ask", o["ask"])):
        txt(x, y + 22, nm, f_s, _MUT)
        txt(x, y + 52, fm(v), f_n, _TXT)
    st_["y"] += 74
    label("Contracts x100 shares")
    stepper("1")
    label("TIF")
    box("Day", arrow=True)
    st_["y"] += 60
    st_["y"] += 22
    check(pad, st_["y"], "Take Profit")
    check(W / 2, st_["y"], "Stop Loss")
    st_["y"] += 40
    gain = (p["tp"] / p["entry"] - 1) * 100
    loss = (1 - p["sl"] / p["entry"]) * 100
    label("TP Price")
    stepper(f"{p['tp']:.2f}")
    st_["y"] += 8
    txt(pad, st_["y"] + 12, f"+${p['profit_1lot']:,.0f} per lot (+{gain:.0f}%)", f_s, _GRN)
    st_["y"] += 24
    label("SL Price")
    stepper(f"{p['sl']:.2f}")
    st_["y"] += 8
    txt(pad, st_["y"] + 12, f"-${p['loss_1lot']:,.0f} per lot (-{loss:.0f}%)", f_s, _RED)
    st_["y"] += 24
    label("TIF")
    box("GTC", arrow=True)
    st_["y"] += 60
    st_["y"] += 26
    txt(pad, st_["y"] + 14, "Amount", f_n, _MUT)
    txt(W - pad, st_["y"] + 14, f"{p['cost_1lot']:,.2f} USD (Debit)", f_b, _TXT, "rm")
    st_["y"] += 44
    y = st_["y"]
    d.rounded_rectangle([pad, y, W - pad, y + 64], 8, fill=_ORG)
    txt(W / 2, y + 32, "Enter these values in Moomoo", f_b, "white", "mm")
    st_["y"] += 64 + pad
    img = img.crop((0, 0, W, st_["y"]))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def send_telegram_photo(cfg, png, caption):
    token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                          data={"chat_id": chat_id, "caption": caption[:1000]},
                          files={"photo": ("moomoo_ticket.png", png)}, timeout=30)
        return r.ok
    except Exception:
        return False


def draw_box(item, side, is_top, kp="x"):
    icon = "🟢" if side == "CALL" else "🔴"
    if not item:
        with st.container(border=True):
            st.subheader(f"{icon} {side}")
            st.info(f"No qualifying contract (needs |score| >= {MIN_SCORE}, volume >= {MIN_RVOL}x normal, "
                    f"no earnings before expiry, next expiry, delta {DELTA_MIN}-{DELTA_MAX}, spread <= {MAX_SPREAD_PCT:g}%, "
                    f"OI >= {MIN_OI}, reward:risk >= {MIN_RR}). Waiting for a cleaner setup is intended.")
        return
    o, p, a = item["opt"], item["plan"], item["a"]
    with st.container(border=True):
        st.subheader(f"{icon} {side} - {o['symbol']}" + ("   ⭐ TOP PICK" if is_top else ""))
        link = tv_url(o["symbol"])
        try:
            st.link_button(f"📈 Open {o['symbol']} chart in TradingView", link)
        except AttributeError:                                  # older Streamlit
            st.markdown(f"[📈 Open {o['symbol']} chart in TradingView]({link})")
        pre = f" | premarket scan score {a['pre_score']}" if a.get("pre_score") is not None else ""
        st.caption(f"{'STRONG' if item['strong'] else 'Moderate'} signal | score {a['score']}{pre} | "
                   f"{' '.join(a['tags'])}")
        if not o["market_open"]:
            st.caption("Option quotes not live (market closed / pre-open) - last price used. Confirm bid/ask before entry.")

        left, right = st.container(), st.container()
        with left:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Contract**")
                st.write(f"Strike: **{o['strike']:g}**")
                st.write(f"Expiry: **{o['expiry']}** ({o['dte']}d)")
                st.write(f"Bid / Ask: {o['bid']:.2f} / {o['ask']:.2f}")
                st.write(f"Spread: {o['spread']:.1f}%")
                st.write(f"Vol / OI: {o['vol']} / {o['oi']}")
            with c2:
                st.markdown("**Greeks**")
                st.write(f"Delta: **{o['delta']:.3f}**")
                st.write(f"Gamma: {o['gamma']:.4f}")
                st.write(f"Theta: {o['theta']:.3f} /day")
                st.write(f"Vega: {o['vega']:.3f}")
                st.write(f"Rho: {o['rho']:.3f}")
                st.write(f"IV: {o['iv'] * 100:.1f}% (IV/HV {o['iv_ratio']:.2f})")
            with c3:
                st.markdown("**Stock & day prediction**")
                st.write(f"Now: **{a['spot']:.2f}** (gap {a['gap']:+.2f}%)")
                st.write(f"Day close est: **{a['pred_close']:.2f}**")
                st.write(f"High / low est: {a['pred_high']:.2f} / {a['pred_low']:.2f}")
                st.write(f"RSI {a['rsi']:.0f} | RVOL {a['rvol']:.1f}x")
                st.write(f"Daily ATR: {a['day_atr']:.2f}")
            pa = a["pa"]
            room = pa["room_up"] if side == "CALL" else pa["room_dn"]
            room_txt = f"{room:.2f} to the next {'resistance' if side == 'CALL' else 'support'}" if room is not None else "open space ahead"
            st.write(f"**Price action:** {pa['struct']} | candle: {pa['candle']} | "
                     f"PDH {pa['pdh']:.2f} / PDL {pa['pdl']:.2f} | {room_txt}"
                     f"{' | BLOCKED (level too close)' if 'PA:blocked' in a['tags'] else ''}")
            st.write(f"**Reward:Risk {p['rr']}** | stock stop {p['stock_sl']} / stock target {p['stock_tp']}")
            st.write(f"**Take profit ${p['tp']:.2f} -> +${p['profit_1lot']:,.0f} per lot** | "
                     f"stop loss ${p['sl']:.2f} -> -${p['loss_1lot']:,.0f} per lot")
            st.caption("Exit: at TP or SL, early if a 5-min bar closes back through EMA15 against you, "
                       "otherwise by 15:45 ET. The result of this trade is fetched automatically after the trade day "
                       "and after expiry (Learning tab).")
        with right:
            code = (o["expiry"][2:4] + o["expiry"][5:7] + o["expiry"][8:10] + f" {o['strike']:g}"
                    + ("C" if o["side"] == "CALL" else "P"))
            st.markdown(f"**Enter in Moomoo:** `{code}` · Buy · Limit **{p['entry']:.2f}** · 1 contract · Day · "
                        f"TP **{p['tp']:.2f}** · SL **{p['sl']:.2f}** (GTC)")
            try:
                png = ticket_png(item)
                with st.expander("Moomoo ticket (reference)", expanded=False):
                    st.image(png, width=260)
                    st.download_button("⬇ Download ticket (PNG)", png,
                                       file_name=f"moomoo_{o['symbol']}_{side}.png", mime="image/png",
                                       key=f"dl_{kp}_{side}_{o['symbol']}_{o['strike']}_{p['entry']}")
                    st.caption("Replica of the Moomoo order screen - not a real screenshot of your account.")
            except Exception:
                pass


def render_scan(res, empty_msg, kp="x"):
    if not res:
        st.info(empty_msg)
        return
    if res.get("running"):
        st.info("Scan is running...")
        return
    if res.get("error"):
        st.error(f"Scan failed: {res['error']}")
        return
    when = dt.datetime.fromisoformat(res["time"])
    tg = " | Telegram sent ✅" if res.get("telegram") else ""
    lr = f" | learned weights: {res['learned']} signals" if res.get("learned") else ""
    st.write(f"Scan time: **{when.strftime('%a %d %b %H:%M')} ET** ({fmt_local(when)} your time) | "
             f"{res.get('scanned', '?')}/{len(SYMBOLS)} stocks read | "
             f"{MARKET_SYM} market bias: **{_bias_txt(res.get('market', 0))}**{tg}{lr}")
    draw_box(res.get("CALL"), "CALL", res.get("top") == "CALL", kp)
    draw_box(res.get("PUT"), "PUT", res.get("top") == "PUT", kp)
    with st.expander("All stocks - direction score & day estimate"):
        if res.get("table"):
            st.dataframe(pd.DataFrame(res["table"]).sort_values("score", key=abs, ascending=False))


# ----- learning tab -----
def _result_table(df, col):
    g = df[df[col].notna()]
    if g.empty:
        return pd.DataFrame()
    grp = g.groupby("side")[col]
    return pd.DataFrame({"trades": grp.size(), "win %": grp.apply(lambda x: round((x > 0).mean() * 100, 1)),
                         "total $/lot": grp.sum().round(0), "avg $/lot": grp.mean().round(1)})


def _moomoo_code(o):
    """Moomoo option code as shown on its ticket, e.g. 261030 250C."""
    return (o["expiry"][2:4] + o["expiry"][5:7] + o["expiry"][8:10] + f" {o['strike']:g}"
            + ("C" if o["side"] == "CALL" else "P"))


def _top10_html(rows):
    """Compact list: share + CALL/PUT + Moomoo contract | buy limit | stop loss | take profit (with $ per lot)."""
    td = "padding:7px 6px;border-bottom:1px solid rgba(128,128,128,0.25);vertical-align:top"
    out = []
    for r in rows:
        if r.get("error"):
            out.append(f'<tr><td style="{td}"><b>{r["sym"]}</b></td>'
                       f'<td colspan="3" style="{td};opacity:0.6;font-size:12px">{r["error"]}</td></tr>')
            continue
        o, p = r["opt"], r["plan"]
        bg = "#2fbf71" if r["side"] == "CALL" else "#ef4f5f"
        flags = [f for f, on in (("weak", r["weak"]), ("low R:R", r["low_rr"]), ("earnings", r["earnings"])) if on]
        flag_txt = f'<div style="font-size:11px;color:#f2a33a">{", ".join(flags)}</div>' if flags else ""
        out.append(
            f'<tr><td style="{td}"><b>{r["sym"]}</b> <span style="background:{bg};color:#fff;border-radius:4px;'
            f'padding:1px 6px;font-size:11px;font-weight:700">{r["side"]}</span>'
            f'<div style="font-size:11px;opacity:0.7;margin-top:2px">{_moomoo_code(o)}</div>{flag_txt}</td>'
            f'<td style="{td};text-align:right"><b>{p["entry"]:.2f}</b>'
            f'<div style="font-size:11px;opacity:0.65">cost ${p["cost_1lot"]:,.0f}</div></td>'
            f'<td style="{td};text-align:right;color:#ef4f5f"><b>{p["sl"]:.2f}</b>'
            f'<div style="font-size:11px">-${p["loss_1lot"]:,.0f}</div></td>'
            f'<td style="{td};text-align:right;color:#2fbf71"><b>{p["tp"]:.2f}</b>'
            f'<div style="font-size:11px">+${p["profit_1lot"]:,.0f}</div></td></tr>')
    head = ('<tr style="opacity:0.6;font-size:12px"><td style="padding:2px 6px">Share / Moomoo contract</td>'
            '<td style="padding:2px 6px;text-align:right">Buy limit</td>'
            '<td style="padding:2px 6px;text-align:right">Stop loss</td>'
            '<td style="padding:2px 6px;text-align:right">Take profit</td></tr>')
    return f'<table style="width:100%;border-collapse:collapse;font-size:14px">{head}{"".join(out)}</table>'


def render_top10_box(box, res):
    """Box under the US indices: the next-expiry CALL / PUT for each top-10 share with the values the Moomoo ticket
       needs, the stop loss and the profit (dollars per lot at stop / target)."""
    with box.container():
        with st.container(border=True):
            st.markdown("### Top 10 options - next expiry")
            ok = isinstance(res, dict) and res.get("time") and not res.get("running") and not res.get("error")
            rows = res.get("top10") if ok else None
            if not rows:
                st.caption("Appears after the next scan - press the Scan button or wait for the 09:45 ET scan.")
                return
            when = dt.datetime.fromisoformat(res["time"])
            st.caption(f"Scan {when.strftime('%a %H:%M')} ET - next expiry (not today's), predicted direction for each share. "
                       "Moomoo: Buy - Limit at the price shown - 1 contract - Day - tick Take Profit and Stop Loss (GTC). "
                       "$ = per lot at stop / target.")
            st.markdown(_top10_html(rows), unsafe_allow_html=True)
            st.caption("'weak' = low score, 'low R:R' = reward:risk under the minimum, 'earnings' = results before expiry. "
                       "Ticket pictures and Greeks: Top 10 options tab. Estimates, not advice.")


def render_top10_options(res):
    """Top 10 US shares: predicted CALL / PUT on the next expiry, the values to enter in Moomoo, stop and profit."""
    st.caption("For each of the top 10 US shares: the predicted direction for the day (CALL = up, PUT = down), the "
               "best next-expiry contract by Greeks / spread / liquidity, and the numbers to type into the Moomoo "
               "ticket. Every share gets a pick, so check the flags: 'weak' = score below the main scan's minimum, "
               "'low R:R' = reward:risk below the minimum, 'earnings' = results due before expiry. Estimates, not advice.")
    if not res or res.get("running") or res.get("error"):
        st.info("No scan yet - press the Scan button above (or wait for the 09:45 ET scan).")
        return
    rows = res.get("top10")
    if not rows:
        st.info("The latest scan has no top-10 data - run a new scan with the button above.")
        return
    when = dt.datetime.fromisoformat(res["time"])
    st.write(f"From the scan at **{when.strftime('%a %d %b %H:%M')} ET** ({fmt_local(when)} your time)")
    table = []
    for r in rows:
        if r.get("error"):
            table.append({"Share": f"{r['name']} ({r['sym']})", "Pick": r.get("side", "-"), "Moomoo contract": r["error"]})
            continue
        o, p = r["opt"], r["plan"]
        flags = [f for f, on in (("weak", r["weak"]), ("low R:R", r["low_rr"]), ("earnings", r["earnings"])) if on]
        table.append({"Share": f"{r['name']} ({r['sym']})", "Pick": r["side"], "Moomoo contract": _moomoo_code(o),
                      "Buy limit": p["entry"], "Stop loss": p["sl"], "Take profit": p["tp"],
                      "Profit/lot $": p["profit_1lot"], "Loss/lot $": p["loss_1lot"], "R:R": p["rr"],
                      "Score": r["score"], "Flags": ", ".join(flags)})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    for r in rows:
        if r.get("error"):
            st.warning(f"{r['name']} ({r['sym']}): {r['error']}")
            continue
        o, p, a = r["opt"], r["plan"], r["a"]
        icon = "🟢" if r["side"] == "CALL" else "🔴"
        flags = [f for f, on in (("weak signal", r["weak"]), ("low R:R", r["low_rr"]),
                                 ("earnings before expiry", r["earnings"])) if on]
        title = f"{icon} {r['sym']} - {r['side']} {_moomoo_code(o)} | buy {p['entry']:.2f} | SL {p['sl']:.2f} | TP {p['tp']:.2f}"
        with st.expander(title):
            if flags:
                st.warning("Flags: " + ", ".join(flags))
            st.markdown(f"**Enter in Moomoo:** `{_moomoo_code(o)}` · Buy · Limit **{p['entry']:.2f}** · 1 contract · Day · "
                        f"TP **{p['tp']:.2f}** · SL **{p['sl']:.2f}** (GTC)")
            st.write(f"**Take profit ${p['tp']:.2f} -> +${p['profit_1lot']:,.0f} per lot** | "
                     f"**stop loss ${p['sl']:.2f} -> -${p['loss_1lot']:,.0f} per lot** | cost ${p['cost_1lot']:,.0f} | "
                     f"reward:risk {p['rr']}")
            st.write(f"Expiry **{o['expiry']}** ({o['dte']}d) | delta {o['delta']:.2f} | IV {o['iv'] * 100:.0f}% | "
                     f"bid / ask {o['bid']:.2f} / {o['ask']:.2f} | vol / OI {o['vol']} / {o['oi']}")
            st.write(f"Stock {a['spot']:.2f} -> day estimate {a['pred_close']:.2f} (H {a['pred_high']:.2f} / "
                     f"L {a['pred_low']:.2f}) | stock stop {p['stock_sl']} / target {p['stock_tp']} | "
                     f"score {r['score']} | {' '.join(a['tags'])}")
            if not o["market_open"]:
                st.caption("Option quotes not live (market closed) - last price used. Confirm bid/ask before entry.")
            try:
                st.image(ticket_png(r), width=260)
                st.caption("Replica of the Moomoo order screen with the values to enter - not a real screenshot.")
            except Exception:
                pass


def render_learning_tab(cfg):
    st.caption("Every candidate of every live scan is saved. After the trade day its plan is replayed on the real "
               "5-minute bars (stop / target / 15:45 exit, option MODELLED from the logged entry and IV). After "
               "expiry the stock's closing price on the expiry date gives the exact expiry value. The signal weights "
               "are re-learned from the plan results; a signal changes weight only after 25+ settled trades. "
               "Expiry result = holding the option to expiry, shown for information.")
    if st.button("Fetch results now"):
        with st.spinner("Fetching results..."):
            n = settle_pending(cfg)
        st.success(f"{n} new results" if n else "Nothing new to settle yet.")
    rows = load_tracker()
    if not rows:
        st.info("No tracked trades yet - they are saved automatically from the next live scan (09:45-14:45 ET).")
        return
    df = pd.DataFrame(rows)
    only_picks = st.checkbox("Only the best CALL / PUT shown on screen (untick = all tracked candidates)", value=False)
    if only_picks:
        df = df[df["picked"]]
    n_plan, n_exp = int(df["plan_pnl"].notna().sum()), int(df["expiry_pnl"].notna().sum())
    m = st.columns(4)
    m[0].metric("Tracked trades", len(df))
    m[1].metric("Trade-day results in", n_plan)
    m[2].metric("Expiry results in", n_exp)
    m[3].metric("Signals re-weighted", len(_load_json(LEARN_FILE).get("weights", {})))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Plan result (stop / target / 15:45 exit)**")
        t = _result_table(df, "plan_pnl")
        st.dataframe(t) if not t.empty else st.info("Waiting for the first trade day to end (16:10 ET).")
    with c2:
        st.markdown("**Expiry result (held to expiry)**")
        t = _result_table(df, "expiry_pnl")
        st.dataframe(t) if not t.empty else st.info("Waiting for the first option to expire.")
    learned = _load_json(LEARN_FILE)
    st.markdown("**What it has learned** (weight 1.0 = unchanged; changes only after "
                f"{MIN_TAG_N}+ settled trades per signal)")
    if learned.get("tags"):
        st.dataframe(pd.DataFrame(learned["tags"]).sort_values("trades", ascending=False))
        st.caption(f"Learned from {learned.get('trades', 0)} settled trades (one per day / stock / side), "
                   f"win rate {learned.get('win_pct', 0)}%, avg ${learned.get('avg_pnl', 0)}/lot. Updated {learned.get('updated', '')}.")
    else:
        st.info("Nothing learned yet - the first results arrive after the first trade day.")
    with st.expander("All tracked trades"):
        cols = ["time", "scan", "sym", "side", "picked", "strike", "expiry", "entry", "tp", "sl", "score",
                "plan_reason", "plan_pnl", "expiry_close", "expiry_value", "expiry_pnl"]
        st.dataframe(df[[c for c in cols if c in df.columns]].sort_values("time", ascending=False))


# ----- backtest tab -----
def show_backtest(tr):
    if tr is None or tr.empty:
        st.info("No trades found in that period.")
        return
    summ = {v: summarize(g) for v, g in tr.groupby("variant")}
    order = [v for v in ("Price action", "Indicators only") if v in summ]
    cols = st.columns(len(order))
    for c, v in zip(cols, order):
        s = summ[v]
        with c:
            st.markdown(f"**{v}**")
            st.metric("Total P&L (1 lot)", f"${s['Total P&L $']:,.0f}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Trades", s["Trades"])
            m2.metric("Win rate", f"{s['Win rate %']}%")
            m3.metric("Profit factor", s["Profit factor"])
    st.dataframe(pd.DataFrame(summ).T.loc[order])

    eq = {}
    for v in order:
        g = tr[tr["variant"] == v].sort_values("exit_time")
        eq[v] = g["pnl"].cumsum().reset_index(drop=True)
    st.markdown("**Equity curve ($ per 1 lot, trade by trade)**")
    st.line_chart(pd.DataFrame(eq))

    main_v = order[0]
    g = tr[tr["variant"] == main_v]
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**{main_v} - by scan**")
        st.dataframe(_breakdown(g, "scan"))
        st.markdown(f"**{main_v} - by side**")
        st.dataframe(_breakdown(g, "side"))
    with b2:
        st.markdown(f"**{main_v} - by signal strength (strong = |score| >= {STRONG_SCORE})**")
        st.dataframe(_breakdown(g, "strong"))
        st.markdown(f"**{main_v} - price action agreed with the trade direction**")
        st.dataframe(_breakdown(g, "pa_agrees"))
    with st.expander("All trades"):
        st.dataframe(tr.sort_values(["date", "exit_time"]))


def render_backtest_tab():
    st.caption("Replays the scheduled scans on past days (no look-ahead) with the same scoring, trade plan and exits. "
               "yfinance keeps only 60 days of 5-minute data and no option history, so the option is MODELLED with "
               "Black-Scholes (strike nearest delta 0.65, next Friday expiry, IV = 20-day HV x multiplier, constant IV). "
               "Stop loss is checked before take profit inside the same bar. Results are estimates, not a promise. "
               "With 100 stocks the download and signal reading take several minutes; the live auto-scan loop is "
               "paused while it runs.")
    c1, c2, c3, c4 = st.columns(4)
    days = c1.slider("Trading days", 5, 40, 20)
    iv_mult = c2.slider("Option IV vs 20-day HV (x)", 0.8, 1.6, BT_IV_MULT, 0.05)
    slip = c3.slider("Slippage per side (%)", 0.0, 5.0, BT_SLIPPAGE * 100, 0.5) / 100
    comm = c4.number_input("Commission $/contract/side", 0.0, 5.0, BT_COMMISSION, 0.05)
    nsym = st.slider("Stocks", 10, len(SYMBOLS), len(SYMBOLS), 5)

    def get_prep(cb):
        """Download + read signals once; kept for the sweep so it does not repeat the slow part."""
        key = (days, nsym)
        if st.session_state.get("bt_key") != key:
            syms = list(SYMBOLS[:nsym]) + ([MARKET_SYM] if MARKET_SYM not in SYMBOLS[:nsym] else [])
            data = fetch_history(syms, cb)
            if not data:
                return None, None
            st.session_state.update(bt_data=data, bt_prep=prepare_backtest(data, days, cb), bt_key=key)
        return st.session_state["bt_data"], st.session_state["bt_prep"]

    b1, b2 = st.columns(2)
    if b1.button("Run backtest", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        cb = lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t)
        data, prep = get_prep(cb)
        if not data:
            bar.empty()
            st.error("Could not download history from yfinance.")
        else:
            tr = run_backtest(data, days, iv_mult, slip, comm, prep=prep)
            bar.empty()
            if not tr.empty:
                tr.to_csv(BT_FILE, index=False)
            st.session_state["bt"] = tr
    if b2.button("Sweep settings (train on early days, test on later days)"):
        bar = st.progress(0.0, text="Starting...")
        cb = lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t)
        data, prep = get_prep(cb)
        if not data:
            bar.empty()
            st.error("Could not download history from yfinance.")
        else:
            sw = run_sweep(data, prep, iv_mult, slip, comm, progress=cb)
            bar.empty()
            if not sw.empty:
                sw.to_csv(SWEEP_FILE, index=False)
            st.session_state["sweep"] = sw
    tr = st.session_state.get("bt")
    if tr is None and os.path.exists(BT_FILE):
        tr = pd.read_csv(BT_FILE)
        st.caption(f"Showing the last saved backtest ({BT_FILE}).")
    if tr is not None:
        show_backtest(tr)
    sw = st.session_state.get("sweep")
    if sw is None and os.path.exists(SWEEP_FILE):
        sw = pd.read_csv(SWEEP_FILE)
    if sw is not None:
        show_sweep(sw)


def show_sweep(sw):
    st.subheader("Sweep results")
    if sw.empty:
        st.info("Not enough days or trades for a sweep (needs 6+ trading days).")
        return
    st.caption(f"Settings were ranked on the days BEFORE {sw['split_date'].iloc[0]} (train) and then checked on the "
               "days from then on (test) - the test columns are the honest ones. 'holds_up' = profit factor 1.5+ in both "
               "halves, 30+ trades each, and far better than the same exits with a random call/put. On random data this "
               "rule passes 0-4 of 108 settings, so only a clearly larger count means something. A high train score that fails on test is overfitting; do not use it.")
    good = sw[sw["holds_up"]]
    hi = good[good["test_win%"] >= 70]
    c1, c2, c3 = st.columns(3)
    c1.metric("Settings tried", len(sw))
    c2.metric("Hold up on both halves", len(good))
    c3.metric("...and 70%+ win rate on test", len(hi))
    if good.empty:
        st.warning("No setting was profitable on both the early and the later days. That is a real result: with "
                   "this data, the signals show no edge to tune. Do not trade these scans on that basis.")
    st.markdown("**Settings that hold up (best test profit factor first)**")
    st.dataframe(good.sort_values("test_pf", ascending=False) if not good.empty else good)
    with st.expander("All settings tried"):
        st.dataframe(sw)
    st.caption("To use a row, copy its values into TP_FRAC, SL_MOVE_FRAC, MAX_LOSS, MIN_SCORE (and REQUIRE_PA = True "
               "for 'PA must agree'; MIN_RR to 0.3 as the sweep used) at the top of the file. Re-test on fresh days "
               "later - 60 days of data is a small sample.")


def cli_sweep(days):
    syms = list(SYMBOLS) + ([MARKET_SYM] if MARKET_SYM not in SYMBOLS else [])
    print(f"Downloading history for {len(syms)} symbols ...")
    data = fetch_history(syms, lambda f, t: None)
    print(f"{len(data)} symbols. Reading signals for {days} days (slow, once) ...")
    prep = prepare_backtest(data, days)
    sw = run_sweep(data, prep)
    if sw.empty:
        print("Not enough data for a sweep.")
        return
    sw.to_csv(SWEEP_FILE, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    good = sw[sw["holds_up"]]
    print(f"{len(sw)} settings tried, {len(good)} profitable on BOTH early (train) and later (test) days, "
          f"{len(good[good['test_win%'] >= 70])} of those with 70%+ win rate on test.")
    print((good if not good.empty else sw.head(10)).sort_values("test_pf", ascending=False).head(15).to_string())
    print(f"\nAll results saved to {SWEEP_FILE}")


def cli_backtest(days):
    syms = list(SYMBOLS) + ([MARKET_SYM] if MARKET_SYM not in SYMBOLS else [])
    print(f"Downloading history for {len(syms)} symbols ...")
    data = fetch_history(syms, lambda f, t: None)
    print(f"{len(data)} symbols with data. Running {days}-day backtest ...")
    tr = run_backtest(data, days)
    if tr.empty:
        print("No trades.")
        return
    tr.to_csv(BT_FILE, index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame({v: summarize(g) for v, g in tr.groupby("variant")}).T.to_string())
    for v, g in tr.groupby("variant"):
        print(f"\n--- {v}: by scan ---")
        print(_breakdown(g, "scan").to_string())
    print(f"\nTrades saved to {BT_FILE}")


def cli_settle():
    n = settle_pending(load_config())
    learned = _load_json(LEARN_FILE)
    print(f"{n} new results. Learned from {learned.get('trades', 0)} settled trades; "
          f"{len(learned.get('weights', {}))} signals re-weighted.")

# ----------------------------- STREAMLIT UI -----------------------------

def main():
    global LEARN_ON
    st.set_page_config(page_title="US Call/Put Scanner", layout="wide")
    cfg = load_config()

    st.title("📈 US Top-100 Call / Put Scanner")
    st.caption("Scans: 09:45 ET, then every hour 10:45-14:45 ET - each one is sent to Telegram. Next expiry (not same "
               "day) - all Greeks + price action + market filter - results are tracked and the scoring learns from "
               "them - educational only, not advice.")

    with st.sidebar:
        auto = st.toggle("Auto scans (see schedule)", value=True)
        LEARN_ON = st.toggle("Self-learning (use learned signal weights)", value=True)
        with st.expander("Telegram"):
            bot = st.text_input("Bot token", value=cfg.get("telegram_bot_token", ""), type="password")
            chat = st.text_input("Chat ID", value=cfg.get("telegram_chat_id", ""))
            b1, b2 = st.columns(2)
            if b1.button("Save"):
                cfg["telegram_bot_token"], cfg["telegram_chat_id"] = bot.strip(), chat.strip()
                save_config(cfg)
                st.success("Saved")
            if b2.button("Test"):
                st.success("Sent") if send_telegram(cfg, "✅ Scanner test message") else st.error("Not sent")
        st.markdown("**Manual run** (screen only, no Telegram)")
        run_pre = st.button("Premarket scan now")
        run_open = st.button("Open-style scan now")
        run_hourly = st.button("Hourly-style scan now")
        st.caption("The big 'Scan CALL & PUT now' button on the page works at any time of day.")

    refresh_weights()

    # layout: the CALL / PUT picks (and the other tabs) on TOP, the market lists below them
    col_main, col_mid, col_news = st.columns([2.8, 2.1, 2.3])
    with col_main:
        btn_area = st.container()                  # the scan button (always visible, never faded)
        top = st.empty()                           # call / put picks (and the other tabs) - one slot, replaced as a whole
        shares_box = st.empty()                    # top 10 US shares (with signals) at the bottom of column 1
    idx_box = col_mid.empty()                      # US indices
    opt_box = col_mid.empty()                      # top-10 options (next expiry), directly below the indices
    news_box = col_news.empty()                    # live news - third column
    com_box = col_news.empty()                     # commodities, directly below the live news
    render_market(idx_box, com_box, shares_box)

    with btn_area:
        scan_any = st.button("🔎 Scan CALL & PUT now (any time of day)", type="primary", key="scan_any")
        st.caption("Works any time: live data in the session, premarket data before the open, last session's data "
                   "after the close / weekends (option prices may then be stale - check bid/ask before ordering).")
    top.info("⏳ Updating scans and results ...")      # replaces the old results at once instead of leaving a faded copy

    try:
        settle_pending(cfg)                      # fetch results of finished trades + re-learn (quick when nothing is due)
    except Exception:
        pass

    if auto:
        run_scheduled(cfg)

    manual = st.session_state.setdefault("manual", {})
    for flag, mode, label in ((run_pre, "pre", "Premarket (manual)"), (run_open, "open", "Open (manual)"),
                              (run_hourly, "hourly", "Hourly (manual)")):
        if flag:
            with st.spinner(f"Scanning {len(SYMBOLS)} stocks..."):
                manual[mode] = scan_once(mode, pre_scores_today(load_state()) if mode != "pre" else None, label)

    if scan_any:
        n_et = dt.datetime.now(ET)
        open_day = n_et.weekday() < 5
        md = "pre" if (open_day and n_et.time() < MARKET_OPEN) else "hourly"
        lbl = "Manual - premarket data" if md == "pre" else (
            "Manual - live" if (open_day and n_et.time() < MARKET_CLOSE) else "Manual - last session data")
        top.info(f"🔎 Scanning {len(SYMBOLS)} stocks - about a minute ...")
        manual["any"] = scan_once(md, None, lbl)

    state = load_state()
    now = dt.datetime.now(ET)
    today = str(now.date())

    def ok(r):
        return isinstance(r, dict) and r.get("time") and not r.get("running") and not r.get("error")

    def when_of(r):
        return dt.datetime.fromisoformat(r["time"])

    scheduled = {k: v for k, v in state.items() if k != "date" and ok(v)}
    everything = list(scheduled.values()) + [r for r in manual.values() if ok(r)]
    latest = max(everything, key=when_of) if everything else None

    def newest(key, mode):
        a, b = scheduled.get(key), manual.get(mode)
        return max([x for x in (a, b) if ok(x)], key=when_of) if (ok(a) or ok(b)) else (state.get(key) or None)

    with top.container():
        marks = []
        for key, _, start, _, label in SCHEDULE:
            marks.append(f"{'✅' if _done(state, key, today) else '⏳'} {start:%H:%M}")
        st.caption("Today (ET): " + "  ".join(marks))

        t_latest, t_open, t_hour, t_top10, t_learn, t_bt = st.tabs(
            ["Latest scan", "Open 09:45", "Hourly scans", "Top 10 options", "Learning", "Backtest"])
        with t_latest:
            render_scan(latest, "No scan yet - the first one runs at 09:45 ET (or press the Scan button above).", "lt")
        with t_open:
            render_scan(newest("open", "open"), "No open scan yet - runs at 09:45 ET.", "op")
        with t_hour:
            hourly = sorted([r for r in everything if r.get("mode") == "hourly"], key=when_of, reverse=True)
            if hourly:
                names = [f"{h.get('label', 'Hourly')} - {when_of(h):%a %H:%M} ET" for h in hourly]
                pick = st.selectbox("Hourly scan", names, index=0)
                render_scan(hourly[names.index(pick)], "", "hr")
            else:
                st.info("No hourly scan yet - they run at 10:45, 11:45, 12:45, 13:45 and 14:45 ET.")
        with t_top10:
            render_top10_options(latest)
        with t_learn:
            render_learning_tab(cfg)
        with t_bt:
            render_backtest_tab()
        box = st.empty()

    try:
        render_top10_box(opt_box, latest)
    except Exception:
        pass
    pick_syms = tuple(dict.fromkeys(latest[sd]["opt"]["symbol"] for sd in ("CALL", "PUT")
                                    if latest and latest.get(sd)))
    try:
        render_news(news_box, pick_syms)
    except Exception:
        pass

    if auto:
        last_mkt = last_settle = last_news = time.time()
        while True:
            now = dt.datetime.now(ET)
            if due_scan(now, load_state()):
                st.rerun()
            if time.time() - last_mkt >= MARKET_REFRESH_SEC:          # keep the index / commodity prices fresh
                render_market(idx_box, com_box, shares_box)
                last_mkt = time.time()
            if time.time() - last_news >= NEWS_REFRESH_SEC:           # refresh the headlines
                try:
                    render_news(news_box, pick_syms)
                except Exception:
                    pass
                last_news = time.time()
            if time.time() - last_settle >= SETTLE_EVERY_SEC:         # pick up finished trades while the page is open
                try:
                    settle_pending(cfg)
                except Exception:
                    pass
                last_settle = time.time()
            entry, when = next_scan(now)
            left = int((when - now).total_seconds())
            box.caption(f"Next: **{entry[4]}** at {when.strftime('%a %H:%M')} ET ({fmt_local(when)} your time) - in "
                        f"{left // 3600}h {(left % 3600) // 60:02d}m {left % 60:02d}s. Keep this page open.")
            time.sleep(1)


if __name__ == "__main__":
    if "--settle" in sys.argv:
        cli_settle()
    elif "--sweep" in sys.argv:
        i = sys.argv.index("--sweep")
        cli_sweep(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 40)
    elif "--backtest" in sys.argv:
        i = sys.argv.index("--backtest")
        cli_backtest(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 20)
    else:
        main()
