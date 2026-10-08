import csv
import os
import time
import json
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

# ----------------------------- SETTINGS -----------------------------
ET = ZoneInfo("America/New_York")

LOG_FILE = "scan_log.csv"
CONFIG_FILE = "config.json"

SCAN_SECONDS = 300            # 5 minutes
MIN_SCORE = 3                 # |score| needed to call a trade
STOP_LOSS_PCT = 25.0          # suggested premium stop

SYMBOLS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","TSLA","BRK-B","UNH","JNJ",
    "V","PG","JPM","HD","MA",
    "XOM","BAC","PFE","KO","PEP",
    "CSCO","ABBV","ADBE","NFLX","CRM"
]

# ----------------------------- CONFIG -----------------------------

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ----------------------------- INDICATORS -----------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ----------------------------- DATA FETCH -----------------------------

def fetch_1m_data(sym):
    # Try 1-minute data first
    df = yf.download(sym, period="2d", interval="1m", progress=False)

    # Fallback if market closed
    if df is None or df.empty:
        df = yf.download(sym, period="5d", interval="5m", progress=False)

    if df is None or df.empty:
        return None

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    try:
        df["time_key"] = df.index.tz_convert(ET)
    except:
        df["time_key"] = df.index

    return df

# ----------------------------- ANALYSIS -----------------------------

def analyse(sym):
    df = fetch_1m_data(sym)
    if df is None or len(df) < 40:
        return None

    close = df["close"]
    e9, e20 = ema(close, 9), ema(close, 20)
    macd = ema(close, 12) - ema(close, 26)
    hist = macd - ema(macd, 9)
    r = rsi(close)
    a = atr(df)

    today = dt.datetime.now(ET).date()
    td = df[df["time_key"].dt.date == today]
    td = td if len(td) > 5 else df.tail(60)

    # VWAP FIXED
    tp = (td["high"] + td["low"] + td["close"]) / 3
    vol_cum = td["volume"].cumsum().iloc[-1]

    if vol_cum == 0 or pd.isna(vol_cum):
        vwap = tp.iloc[-1]
    else:
        vwap = (tp * td["volume"]).cumsum().iloc[-1] / vol_cum

    spot = close.iloc[-1]
    score = 0
    score += 1 if e9.iloc[-1] > e20.iloc[-1] else -1
    score += 1 if spot > vwap else -1
    score += 1 if r.iloc[-1] > 55 else (-1 if r.iloc[-1] < 45 else 0)
    score += 1 if (hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]) else (
        -1 if (hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]) else 0
    )
    mom = (spot - close.iloc[-6]) / max(a.iloc[-1], 1e-9)
    score += 1 if mom > 0.5 else (-1 if mom < -0.5 else 0)

    # Price projection
    y = close.tail(12).values
    x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1)
    base = slope * (len(y)-1) + icpt
    proj = {m: base + slope*m for m in (5,10,15)}
    band = a.iloc[-1] * np.sqrt(10)

    return dict(
        sym=sym,
        spot=spot,
        vwap=vwap,
        rsi=r.iloc[-1],
        score=score,
        p5=proj[5],
        p10=proj[10],
        p15=proj[15],
        band=band
    )

# ----------------------------- OPTIONS -----------------------------

def pick_option(sym, spot, side):
    try:
        t = yf.Ticker(sym)
        expiries = t.options
        if not expiries:
            return None

        today = dt.datetime.now(ET).date()
        future = [e for e in expiries if dt.datetime.strptime(e, "%Y-%m-%d").date() > today]
        if not future:
            return None

        expiry = sorted(future)[0]
        chain = t.option_chain(expiry)
        df_opts = chain.calls if side == "CALL" else chain.puts

        if df_opts.empty:
            return None

        if side == "CALL":
            itm = df_opts[df_opts["strike"] < spot]
        else:
            itm = df_opts[df_opts["strike"] > spot]

        if itm.empty:
            return None

        itm["dist"] = (itm["strike"] - spot).abs()
        itm = itm.sort_values("dist").head(6)

        itm["spread_pct"] = (itm["ask"] - itm["bid"]) / itm["ask"].replace(0, np.nan) * 100
        itm = itm[(itm["bid"] > 0) & (itm["ask"] > 0) & (itm["spread_pct"] <= 10)]

        if itm.empty:
            return None

        best = itm.iloc[0]

        return dict(
            symbol=sym,
            expiry=expiry,
            strike=float(best["strike"]),
            bid=float(best["bid"]),
            ask=float(best["ask"]),
            iv=float(best.get("impliedVolatility", np.nan)),
            vol=int(best.get("volume", 0)),
            oi=int(best.get("openInterest", 0)),
            spread=float(best["spread_pct"])
        )
    except:
        return None

# ----------------------------- TELEGRAM -----------------------------

def send_telegram_alert(cfg, msg):
    token = cfg.get("telegram_bot_token")
    chat_id = cfg.get("telegram_chat_id")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": msg})
    except:
        pass

# ----------------------------- LOGGING -----------------------------

def init_log():
    new = not os.path.exists(LOG_FILE)
    f = open(LOG_FILE, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow([
            "time_et","symbol
