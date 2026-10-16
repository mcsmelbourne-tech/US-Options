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
TP_FRAC = 1.0                                # stock target = this x the expected move (smaller = closer target, higher win rate)
SL_MOVE_FRAC = 0.5                           # stock stop = 50% of the expected move against the trade
MIN_LOSS, MAX_LOSS = 0.10, 0.30              # option stop loss clamped to 10-30% of premium
MIN_RR = 1.3                                 # skip trades with reward:risk below this
REQUIRE_PA = False                           # True = only trade when price action points the same way as the trade

# price action
SWING_K = 3                                  # a swing high/low = extreme of 3 bars either side (5-min bars)
BLOCKED_FRAC = 0.35                          # next S/R closer than 35% of the expected move = trade is "blocked"

# backtest assumptions
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
    return _clean_intraday(yf.Ticker(sym).history(period="5d", interval="5m", prepost=True))


def fetch_daily(sym, period="3mo"):
    d = yf.Ticker(sym).history(period=period, interval="1d")
    if d is None or d.empty:
        return None
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    return d

# ----------------------------- PRICE ACTION -----------------------------

def _swings(df, k=SWING_K):
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

    if spot > pdh:
        add(1, "PDH-break")
    elif spot < pdl:
        add(-1, "PDL-break")

    levels = [pdh, pdl] + [x for x in (pm_high, pm_low) if x is not None] + sh[-4:] + sl[-4:]
    gap = 0.1 * atr5
    above = [x - spot for x in levels if x > spot + gap]
    below = [spot - x for x in levels if x < spot - gap]
    info["room_up"] = min(above) if above else None
    info["room_dn"] = min(below) if below else None
    info["sw_lo"] = next((x for x in reversed(sl) if x < spot - gap), None)
    info["sw_hi"] = next((x for x in reversed(sh) if x > spot + gap), None)
    return pts, tags, info

# ----------------------------- ANALYSIS & SCANNING -----------------------------

def analyse_df(sym, df, dd, mode):
    try:
        if df is None or dd is None or len(df) < 40:
            return None

        sess = df.index[-1].date()
        hist_d = dd[np.array([x < sess for x in dd.index.date])]
        if len(hist_d) < 20:
            return None
        prev_close = float(hist_d["close"].iloc[-1])
        day_atr = float(atr(hist_d).iloc[-1])
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
            elapsed = (last_bar.hour * 60 + last_bar.minute + 5) - (9 * 60 + 30)
            frac = float(np.interp(elapsed, [0, 10, 30, 60, 120, 180, 240, 300, 360, 390],
                                   [0, 0.07, 0.16, 0.26, 0.40, 0.50, 0.60, 0.72, 0.85, 1.0]))
            rvol = reg_vol / (avg_vol * max(frac, 0.02))
        elif avg_vol > 0:
            rvol = pm_vol / (avg_vol * 0.03)
        else:
            rvol = 0.0

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

        pa_s, pa_tags, pa = price_action(df, hist_d, spot, atr5, pm_high, pm_low, mode)

        return dict(sym=sym, mode=mode, spot=spot, vwap=vwap, rsi=rsi_last, gap=gap, rvol=float(rvol),
                    day_atr=day_atr, atr5=atr5, hv=hv, prev_close=prev_close, pm_high=pm_high, pm_low=pm_low,
                    s_core=int(s), tags_core=tags, ob=ob, pa_s=int(pa_s), pa_tags=pa_tags, pa=pa)
    except Exception:
        return None


def finalize(a, use_pa=True, pre_prev=None):
    a = dict(a)
    sym = a["sym"]
    s, tags = a["s_core"], list(a["tags_core"])

    def add(v, tag):
        nonlocal s
        s += v
        if v:
            tags.append(("+" if v > 0 else "-") + tag)

    if a["mode"] != "pre" and pre_prev and pre_prev.get(sym) and s != 0:
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
        if room is not None and room < BLOCKED_FRAC * move:
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


def pa_agrees(a):
    return a["pa_s"] != 0 and (a["pa_s"] > 0) == (a["score"] > 0)


def analyse(sym, mode, pre_prev):
    try:
        base = analyse_df(sym, fetch_intraday(sym), fetch_daily(sym), mode)
        return finalize(base, True, pre_prev) if base else None
    except Exception:
        return None


def pick_option(sym, spot, side, hv):
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
            elif last > 0:
                mid, spread, entry = last, 0.0, last
            else:
                continue

            if not (0.05 <= iv <= 3.0) or oi < MIN_OI or spread > MAX_SPREAD_PCT:
                continue
            g = greeks(spot, float(o["strike"]), T, iv, side)
            if not g or not (DELTA_MIN <= abs(g["delta"]) <= DELTA_MAX):
                continue

            d = abs(g["delta"])
            theta_pct = abs(g["theta"]) / max(mid, 0.01) * 100
            gamma_pct = g["gamma"] * spot / max(mid, 0.01) * 100
            vega_pct = g["vega"] / max(mid, 0.01) * 100
            iv_ratio = iv / hv if hv > 0 else 1.0
            quality = (
                100 - abs(d - TARGET_DELTA) * 100 - spread * 2 - theta_pct * 4
                + gamma_pct * 2 - vega_pct * 1.0 - max(0.0, iv_ratio - 1.2) * 10
                + math.log10(oi + vol + 1) * 3
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
    if ref is not None:
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


def scan_once(mode, pre_prev=None, label=None):
    now = dt.datetime.now(ET)
    with ThreadPoolExecutor(max_workers=6) as ex:
        analysed = [a for a in ex.map(lambda s: analyse(s, mode, pre_prev), SYMBOLS) if a]

    table = [dict(symbol=a["sym"], price=round(a["spot"], 2), gap_pct=round(a["gap"], 2), rsi=round(a["rsi"], 1),
                  rvol=round(a["rvol"], 2), score=a["score"], price_action=a["pa_s"], pre_score=a["pre_score"],
                  structure=a["pa"]["struct"], day_est=round(a["pred_close"], 2)) for a in analysed]

    cands = sorted([a for a in analysed if abs(a["score"]) >= MIN_SCORE and (not REQUIRE_PA or pa_agrees(a))],
                   key=lambda a: -abs(a["score"]))[:CANDIDATES]

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
