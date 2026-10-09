"""
US Top-50 Call/Put Scanner - scheduled scans every US trading day (yfinance + Greeks + price action)

  Scan 1  PREMARKET    09:00 ET - gap, premarket range/volume, trend; option prices are indicative
  Scan 2  OPEN +10min  09:40 ET - re-scores everything, ADDS the premarket scan result
                                  (opening-range breakout vs premarket high/low, direction consistency)
  Scan 3+ HOURLY       10:40, 11:40, 12:40, 13:40, 14:40 ET - fresh picks through the day (volume is judged
                                  against the time of day; 14:40 is the last, trades exit by 15:45 ET)

Per stock: EMA 9 / EMA 15, RSI, MACD, volume (relative to normal), ATR, VWAP, gap
           + PRICE ACTION: swing structure (HH/HL, LH/LL), break of structure (BOS), candle patterns
             (engulfing, hammer, shooting star, strong bar), previous-day high/low breaks, and a
             "blocked" check when the next support/resistance is too close for the expected move.
             The stop loss is placed beyond the last swing low (call) / swing high (put).
Per option: Black-Scholes Greeks (delta, gamma, theta, vega, rho) from yfinance implied vol,
            next expiry AFTER today, ranked by Greeks + spread + liquidity + IV vs historical vol.
Output:     best CALL and best PUT in a box (+ one TOP PICK), predicted day price (close/high/low),
            entry, stop loss, take profit, profit for 1 lot. Compact Telegram alert for each scan.

BACKTEST:   "Backtest" tab (or command line) replays the same scans on the last N trading days of 5-minute
            data, prices the option with Black-Scholes, walks forward bar by bar to the SL / TP / 15:45 exit,
            and compares results WITH and WITHOUT price action.

Run:   streamlit run us_itm_scanner.py        (leave the page open - it fires the scans on time)
       python us_itm_scanner.py --backtest 20 (backtest from the command line, 20 trading days)
Needs: pip install streamlit yfinance pandas numpy requests
Educational use only - estimates are heuristics, not advice.
"""

import csv
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

SCAN_PRE_TIME = dt.time(9, 0)                # premarket scan (ET)
SCAN_OPEN_TIME = dt.time(9, 40)              # 10 min after the 09:30 open (ET)
SCAN_OPEN_LATEST = dt.time(10, 30)           # if the app starts late, still run the open scan until here
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
EXIT_TIME = dt.time(15, 45)                  # trades are closed by this time
HOURLY_TIMES = [dt.time(h, 40) for h in range(10, 15)]   # 10:40 ... 14:40 ET, every hour after the open scan
HOURLY_GRACE_MIN = 20                                    # a late start can still run an hourly scan this long after its time


def _plus(t, minutes):
    return (dt.datetime.combine(dt.date(2000, 1, 1), t) + dt.timedelta(minutes=minutes)).time()


# (key, mode, start time ET, latest start ET, label) - in time order
SCHEDULE = (
    [("pre", "pre", SCAN_PRE_TIME, MARKET_OPEN, "Premarket"),
     ("open", "open", SCAN_OPEN_TIME, SCAN_OPEN_LATEST, "Open +10min")]
    + [(f"h{t:%H%M}", "hourly", t, _plus(t, HOURLY_GRACE_MIN), f"Hourly {t:%H:%M}") for t in HOURLY_TIMES]
)

RISK_FREE = 0.04
CONTRACT_SIZE = 100                          # 1 lot = 1 contract = 100 shares

MIN_SCORE = 3                                # minimum |score| to be considered
STRONG_SCORE = 5                             # |score| for a "strong" signal
CANDIDATES = 12                              # stocks (best scores) whose option chains are checked

# option filters / ranking
TARGET_DELTA = 0.65
DELTA_MIN, DELTA_MAX = 0.55, 0.80
MAX_SPREAD_PCT = 10.0
MIN_OI = 100

# trade plan (stock based, converted to option prices with the Greeks / Black-Scholes)
HOLD_HOURS = 6.0                             # time decay assumed before exit
SL_MOVE_FRAC = 0.5                           # stock stop = 50% of the expected move against the trade
MIN_LOSS, MAX_LOSS = 0.10, 0.30              # option stop loss clamped to 10-30% of premium
MIN_RR = 1.3                                 # skip trades with reward:risk below this

# price action
SWING_K = 3                                  # a swing high/low = extreme of 3 bars either side (5-min bars)
BLOCKED_FRAC = 0.35                          # next S/R closer than 35% of the expected move = trade is "blocked"

# backtest assumptions (yfinance has no history of option quotes, so the option is MODELLED)
BT_IV_MULT = 1.10                            # assumed option IV = 20-day historical vol x this
BT_SLIPPAGE = 0.02                           # 2% of premium lost on entry and on stop / time exits
BT_COMMISSION = 0.65                         # $ per contract per side

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
]

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


def finalize(a, use_pa=True, pre_prev=None):
    """Final score, day prediction and stop reference. use_pa=False gives the original indicator-only score
    (used by the backtest to compare)."""
    a = dict(a)
    sym = a["sym"]
    s, tags = a["s_core"], list(a["tags_core"])

    def add(v, tag):
        nonlocal s
        s += v
        if v:
            tags.append(("+" if v > 0 else "-") + tag)

    if a["mode"] != "pre" and pre_prev and pre_prev.get(sym) and s != 0:
        # premarket scan agreed with today's direction -> strengthen it; disagreed -> weaken it
        agree = (pre_prev[sym] > 0) == (s > 0)
        add((1 if s > 0 else -1) * (1 if agree else -1), "PM-scan")
    add(*a["ob"])
    if use_pa:
        s += a["pa_s"]
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
    a.update(score=int(s), tags=tags, move=float(move),
             pred_close=float(pred_close),
             pred_high=float(max(spot, pred_close) + 0.25 * a["day_atr"]),
             pred_low=float(min(spot, pred_close) - 0.25 * a["day_atr"]),
             pre_score=(pre_prev or {}).get(sym),
             sl_ref=(a["pa"]["sw_lo"] if s > 0 else a["pa"]["sw_hi"]) if use_pa else None,
             pa_on=use_pa)
    return a


def analyse(sym, mode, pre_prev):
    """Live analysis (current data) with price action on."""
    try:
        base = analyse_df(sym, fetch_intraday(sym), fetch_daily(sym), mode)
        return finalize(base, True, pre_prev) if base else None
    except Exception:
        return None

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


def trade_plan(opt, a):
    """Stock targets from the day prediction, converted to option prices with Black-Scholes.
       With price action on, the stock stop goes just beyond the last swing low (call) / swing high (put)."""
    side = opt["side"]
    sign = 1 if side == "CALL" else -1
    S, K, T, iv = a["spot"], opt["strike"], opt["T"], opt["iv"]
    T_exit = max(T - HOLD_HOURS / (24 * 365), 1.0 / (24 * 365))

    now_px = bs_price(S, K, T, iv, side)
    stock_tp = S + sign * a["move"]
    stock_sl = S - sign * a["move"] * SL_MOVE_FRAC
    ref = a.get("sl_ref")
    if ref is not None:                                   # structural stop, kept within a sensible distance
        dist = abs(S - ref) + 0.1 * a["atr5"]
        if 0.25 * a["move"] <= dist <= 1.0 * a["move"]:
            stock_sl = S - sign * dist
    gain = bs_price(stock_tp, K, T_exit, iv, side) - now_px
    loss = now_px - bs_price(stock_sl, K, T_exit, iv, side)

    entry = opt["entry"]
    if gain <= 0:
        return None
    loss = min(max(loss, entry * MIN_LOSS), entry * MAX_LOSS)

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

# ----------------------------- SCAN -----------------------------

def scan_once(mode, pre_prev=None, label=None):
    now = dt.datetime.now(ET)
    with ThreadPoolExecutor(max_workers=6) as ex:
        analysed = [a for a in ex.map(lambda s: analyse(s, mode, pre_prev), SYMBOLS) if a]

    table = [dict(symbol=a["sym"], price=round(a["spot"], 2), gap_pct=round(a["gap"], 2), rsi=round(a["rsi"], 1),
                  rvol=round(a["rvol"], 2), score=a["score"], price_action=a["pa_s"], pre_score=a["pre_score"],
                  structure=a["pa"]["struct"], day_est=round(a["pred_close"], 2)) for a in analysed]

    cands = sorted([a for a in analysed if abs(a["score"]) >= MIN_SCORE], key=lambda a: -abs(a["score"]))[:CANDIDATES]

    def build(a):
        side = "CALL" if a["score"] > 0 else "PUT"
        opt = pick_option(a["sym"], a["spot"], side, a["hv"])
        if not opt:
            return None
        plan = trade_plan(opt, a)
        if not plan or plan["rr"] < MIN_RR:
            return None
        rank = abs(a["score"]) * 10 + opt["quality"] * 0.25 + min(plan["rr"], 3.0) * 5
        return dict(side=side, a=a, opt=opt, plan=plan, rank=rank, strong=abs(a["score"]) >= STRONG_SCORE)

    with ThreadPoolExecutor(max_workers=6) as ex:
        built = [b for b in ex.map(build, cands) if b]

    res = {"mode": mode, "label": label or {"pre": "Premarket", "open": "Open +10min", "hourly": "Hourly"}[mode],
           "time": now.isoformat(timespec="seconds"), "CALL": None, "PUT": None, "top": None,
           "table": table, "scores": {a["sym"]: a["score"] for a in analysed}, "scanned": len(analysed)}
    for b in built:
        if res[b["side"]] is None or b["rank"] > res[b["side"]]["rank"]:
            res[b["side"]] = b
    picks = [s for s in ("CALL", "PUT") if res[s]]
    if picks:
        res["top"] = max(picks, key=lambda s: res[s]["rank"])
    log_picks(res)
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
#   3. enters at the open of the bar after the scan (premarket scan: the 09:30 open), plus slippage,
#   4. walks forward bar by bar: stop loss checked first (worst case) then take profit, otherwise exit 15:45 ET,
#   5. runs the same scans twice - WITH price action and WITHOUT - so you can see what it adds.

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

    with ThreadPoolExecutor(max_workers=6) as ex:
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


def simulate_trade(opt, plan, bars, slip):
    """Walk the 5-min bars forward. Returns (exit_option_price, reason, exit_time) or None."""
    if bars is None or bars.empty:
        return None
    side, K, iv, exp_dt = opt["side"], opt["strike"], opt["iv"], opt["exp_dt"]
    tp, sl = plan["tp"], plan["sl"]

    def val(S, t_end):
        T = max((exp_dt - t_end).total_seconds() / (365 * 86400), 1.0 / (365 * 24))
        return bs_price(S, K, T, iv, side)

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


def run_backtest(data, days=20, iv_mult=BT_IV_MULT, slip=BT_SLIPPAGE, commission=BT_COMMISSION, progress=None):
    """Returns a DataFrame with one row per trade, variant = 'Price action' or 'Indicators only'."""
    if not data:
        return pd.DataFrame()
    today_et = dt.datetime.now(ET).date()
    ref = max(data.values(), key=lambda x: len(x[0]))[0]
    regr = ref[np.array([x >= MARKET_OPEN for x in ref.index.time])]
    full = [d for d, g in regr.groupby(regr.index.date)
            if d < today_et and len(g) >= 70 and g.index[-1].time() >= dt.time(15, 30)]
    test_days = full[-days:]
    variants = {"Price action": True, "Indicators only": False}
    rows = []

    for di, day in enumerate(test_days):
        pre_scores = {v: {} for v in variants}
        taken = set()
        exit_ts = pd.Timestamp(dt.datetime.combine(day, EXIT_TIME), tz=ET)
        for key, mode, start, _, label in SCHEDULE:
            when = pd.Timestamp(dt.datetime.combine(day, start), tz=ET)
            entry_ts = max(when, pd.Timestamp(dt.datetime.combine(day, MARKET_OPEN), tz=ET))
            base = []
            for sym, (df, dd) in data.items():
                i = df.index.searchsorted(when)
                a0 = analyse_df(sym, df.iloc[max(0, i - 720):i], dd, mode)
                if a0:
                    base.append(a0)

            for vname, use_pa in variants.items():
                fin = [finalize(a0, use_pa, pre_scores[vname] if mode != "pre" else None) for a0 in base]
                if mode == "pre":
                    pre_scores[vname] = {a["sym"]: a["score"] for a in fin}
                cands = sorted([a for a in fin if abs(a["score"]) >= MIN_SCORE],
                               key=lambda a: -abs(a["score"]))[:CANDIDATES]
                best = {"CALL": None, "PUT": None}
                for a in cands:
                    side = "CALL" if a["score"] > 0 else "PUT"
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
                    plan = trade_plan(opt, dict(a, spot=S_in))
                    if not plan or plan["rr"] < MIN_RR:
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
        if progress:
            progress((di + 1) / len(test_days), f"Backtest day {di + 1}/{len(test_days)} ({day})")
    return pd.DataFrame(rows)


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

def telegram_text(res):
    when = dt.datetime.fromisoformat(res["time"])
    title = "PREMARKET (indicative)" if res["mode"] == "pre" else res.get("label", "SCAN").upper()
    lines = [f"📊 {title} · {when.strftime('%H:%M')} ET"]
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
        pa_txt = " ".join(t.replace("PA:", "") for t in a["pa_tags"] + (["PA:blocked"] if "PA:blocked" in a["tags"] else []))
        if pa_txt:
            lines.append(f"Price action: {a['pa']['struct']} | {pa_txt}")
    if not shown:
        lines.append("No qualifying setup this scan.")
    return "\n".join(lines)

# ----------------------------- DISPLAY -----------------------------

def fmt_local(when):
    return when.astimezone(LOCAL_TZ).strftime("%a %d %b %H:%M")


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
        f'<div style="color:{_MUT};font-size:12px;margin-top:2px">{o["symbol"]} - {o["side"]} - scanner values</div>'
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


def draw_box(item, side, is_top):
    icon = "🟢" if side == "CALL" else "🔴"
    if not item:
        with st.container(border=True):
            st.subheader(f"{icon} {side}")
            st.info("No qualifying contract (needs |score| >= 3, next expiry, delta 0.55-0.80, spread <= 10%, "
                    "OI >= 100, reward:risk >= 1.3).")
        return
    o, p, a = item["opt"], item["plan"], item["a"]
    with st.container(border=True):
        st.subheader(f"{icon} {side} - {o['symbol']}" + ("   ⭐ TOP PICK" if is_top else ""))
        pre = f" | premarket scan score {a['pre_score']}" if a.get("pre_score") is not None else ""
        st.caption(f"{'STRONG' if item['strong'] else 'Moderate'} signal | score {a['score']}{pre} | "
                   f"{' '.join(a['tags'])}")
        if not o["market_open"]:
            st.caption("Option quotes not live (market closed / pre-open) - last price used. Confirm bid/ask before entry.")

        left, right = st.columns([3, 2])
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
                       "otherwise by 15:45 ET.")
        with right:
            st.markdown(order_ticket_html(item), unsafe_allow_html=True)


def render_scan(res, empty_msg):
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
    st.write(f"Scan time: **{when.strftime('%a %d %b %H:%M')} ET** ({fmt_local(when)} your time) | "
             f"{res.get('scanned', '?')}/{len(SYMBOLS)} stocks read{tg}")
    draw_box(res.get("CALL"), "CALL", res.get("top") == "CALL")
    draw_box(res.get("PUT"), "PUT", res.get("top") == "PUT")
    with st.expander("All stocks - direction score & day estimate"):
        if res.get("table"):
            st.dataframe(pd.DataFrame(res["table"]).sort_values("score", key=abs, ascending=False))


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
               "While it runs (a few minutes) the live auto-scan loop is paused.")
    c1, c2, c3, c4 = st.columns(4)
    days = c1.slider("Trading days", 5, 40, 20)
    iv_mult = c2.slider("Option IV vs 20-day HV (x)", 0.8, 1.6, BT_IV_MULT, 0.05)
    slip = c3.slider("Slippage per side (%)", 0.0, 5.0, BT_SLIPPAGE * 100, 0.5) / 100
    comm = c4.number_input("Commission $/contract/side", 0.0, 5.0, BT_COMMISSION, 0.05)
    nsym = st.slider("Stocks", 10, len(SYMBOLS), len(SYMBOLS), 5)

    if st.button("Run backtest", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        cb = lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t)
        data = fetch_history(SYMBOLS[:nsym], cb)
        if not data:
            bar.empty()
            st.error("Could not download history from yfinance.")
        else:
            tr = run_backtest(data, days, iv_mult, slip, comm, cb)
            bar.empty()
            if not tr.empty:
                tr.to_csv(BT_FILE, index=False)
            st.session_state["bt"] = tr
    tr = st.session_state.get("bt")
    if tr is None and os.path.exists(BT_FILE):
        tr = pd.read_csv(BT_FILE)
        st.caption(f"Showing the last saved backtest ({BT_FILE}).")
    if tr is not None:
        show_backtest(tr)


def cli_backtest(days):
    print(f"Downloading history for {len(SYMBOLS)} stocks ...")
    data = fetch_history(SYMBOLS, lambda f, t: None)
    print(f"{len(data)} stocks with data. Running {days}-day backtest ...")
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

# ----------------------------- STREAMLIT UI -----------------------------

def main():
    st.set_page_config(page_title="US Call/Put Scanner", layout="wide")
    cfg = load_config()

    st.title("📈 US Top-50 Call / Put Scanner")
    st.caption("Scans: premarket 09:00 ET, open +10 min 09:40 ET, then every hour 10:40-14:40 ET - each one is sent to "
               "Telegram. Next expiry (not same day) - all Greeks + price action - educational only, not advice.")

    with st.sidebar:
        auto = st.toggle("Auto scans (see schedule)", value=True)
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
        run_open = st.button("Open scan now")
        run_hourly = st.button("Hourly-style scan now")

    if auto:
        run_scheduled(cfg)

    manual = st.session_state.setdefault("manual", {})
    for flag, mode, label in ((run_pre, "pre", "Premarket (manual)"), (run_open, "open", "Open +10min (manual)"),
                              (run_hourly, "hourly", "Hourly (manual)")):
        if flag:
            with st.spinner(f"Scanning {len(SYMBOLS)} stocks..."):
                manual[mode] = scan_once(mode, pre_scores_today(load_state()) if mode != "pre" else None, label)

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

    # today's schedule at a glance
    marks = []
    for key, _, start, _, label in SCHEDULE:
        if _done(state, key, today):
            marks.append(f"✅ {start:%H:%M}")
        else:
            marks.append(f"⏳ {start:%H:%M}")
    st.caption("Today (ET): " + "  ".join(marks))

    def newest(key, mode):
        a, b = scheduled.get(key), manual.get(mode)
        return max([x for x in (a, b) if ok(x)], key=when_of) if (ok(a) or ok(b)) else (state.get(key) or None)

    t_latest, t_pre, t_open, t_hour, t_bt = st.tabs(["Latest scan", "Premarket", "Open +10min", "Hourly scans",
                                                     "Backtest"])
    with t_latest:
        render_scan(latest, "No scan yet - the first one runs at 09:00 ET (or use a manual run).")
    with t_pre:
        render_scan(newest("pre", "pre"), "No premarket scan yet - runs at 09:00 ET.")
    with t_open:
        render_scan(newest("open", "open"), "No open scan yet - runs at 09:40 ET.")
    with t_hour:
        hourly = sorted([r for r in everything if r.get("mode") == "hourly"], key=when_of, reverse=True)
        if hourly:
            names = [f"{h.get('label', 'Hourly')} - {when_of(h):%a %H:%M} ET" for h in hourly]
            pick = st.selectbox("Hourly scan", names, index=0)
            render_scan(hourly[names.index(pick)], "")
        else:
            st.info("No hourly scan yet - they run at 10:40, 11:40, 12:40, 13:40 and 14:40 ET.")
    with t_bt:
        render_backtest_tab()

    if auto:
        box = st.empty()
        while True:
            now = dt.datetime.now(ET)
            if due_scan(now, load_state()):
                st.rerun()
            entry, when = next_scan(now)
            left = int((when - now).total_seconds())
            box.caption(f"Next: **{entry[4]}** at {when.strftime('%a %H:%M')} ET ({fmt_local(when)} your time) - in "
                        f"{left // 3600}h {(left % 3600) // 60:02d}m {left % 60:02d}s. Keep this page open.")
            time.sleep(1)


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        i = sys.argv.index("--backtest")
        cli_backtest(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 20)
    else:
        main()
