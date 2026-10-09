"""
US Top-50 Call/Put Scanner - two scans per US trading day (yfinance + Greeks)

  Scan 1  PREMARKET   09:00 ET  - gap, premarket range/volume, trend; option prices are indicative
  Scan 2  OPEN +10min 09:40 ET  - re-scores everything, ADDS the premarket scan result
                                  (opening-range breakout vs premarket high/low, direction consistency)

Per stock: EMA 9 / EMA 15, RSI, MACD, volume (relative to normal), ATR, VWAP, gap.
Per option: Black-Scholes Greeks (delta, gamma, theta, vega, rho) from yfinance implied vol,
            next expiry AFTER today, ranked by Greeks + spread + liquidity + IV vs historical vol.
Output:     best CALL and best PUT in a box (+ one TOP PICK), predicted day price (close/high/low),
            entry, stop loss, take profit, profit for 1 lot. Compact Telegram alert for each scan.

Run:   streamlit run us_itm_scanner.py        (leave the page open - it fires the scans on time)
Needs: pip install streamlit yfinance pandas numpy requests
Educational use only - estimates are heuristics, not advice.
"""

import csv
import os
import json
import math
import time
import datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

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

SCAN_PRE_TIME = dt.time(9, 0)                # premarket scan (ET)
SCAN_OPEN_TIME = dt.time(9, 40)              # 10 min after the 09:30 open (ET)
SCAN_OPEN_LATEST = dt.time(10, 30)           # if the app starts late, still run the open scan until here
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)

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


def fetch_intraday(sym):
    """5-minute bars incl. premarket (04:00-16:00 ET), last 5 days."""
    df = yf.Ticker(sym).history(period="5d", interval="5m", prepost=True)
    if df is None or df.empty:
        return None
    df = _to_et(df).rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    keep = np.array([dt.time(4, 0) <= x < MARKET_CLOSE for x in df.index.time])
    return df[keep]


def fetch_daily(sym):
    d = yf.Ticker(sym).history(period="3mo", interval="1d")
    if d is None or d.empty:
        return None
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    return d

# ----------------------------- STOCK ANALYSIS + DAY PREDICTION -----------------------------

def analyse(sym, mode, pre_prev):
    """mode: 'pre' or 'open'. pre_prev: {symbol: score} from today's premarket scan (or None)."""
    try:
        df, dd = fetch_intraday(sym), fetch_daily(sym)
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

        if mode == "open" and len(reg) > 0 and avg_vol > 0:
            rvol = reg_vol / (avg_vol * 0.07)       # first ~10 min normally ~7% of the day's volume
        elif avg_vol > 0:
            rvol = pm_vol / (avg_vol * 0.03)        # premarket normally ~3% of the day's volume
        else:
            rvol = 0.0

        # ---- direction score ----
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
        if mode == "open" and len(reg) > 0 and pm_high is not None:
            add(1 if spot > pm_high else (-1 if spot < pm_low else 0), "PM-break")
        if mode == "open" and pre_prev and pre_prev.get(sym) and s != 0:
            # premarket scan agreed with today's direction -> strengthen it; disagreed -> weaken it
            agree = (pre_prev[sym] > 0) == (s > 0)
            add((1 if s > 0 else -1) * (1 if agree else -1), "PM-scan")
        if rsi_last > 75:
            add(-1, "overbought")
        elif rsi_last < 25:
            add(1, "oversold")

        # ---- day price prediction (ATR-based, scaled by conviction) ----
        conf = float(np.clip(s / 7.0, -1, 1))
        sign = 1 if s > 0 else -1
        move = day_atr * (0.25 + 0.20 * abs(conf))
        pred_close = spot + sign * move
        pred_high = max(spot, pred_close) + 0.25 * day_atr
        pred_low = min(spot, pred_close) - 0.25 * day_atr

        return dict(sym=sym, spot=spot, vwap=vwap, rsi=rsi_last, score=int(s), tags=tags,
                    gap=gap, rvol=float(rvol), day_atr=day_atr, atr5=atr5, hv=hv, prev_close=prev_close,
                    pm_high=pm_high, pm_low=pm_low, move=float(move),
                    pred_close=float(pred_close), pred_high=float(pred_high), pred_low=float(pred_low),
                    pre_score=(pre_prev or {}).get(sym))
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
    """Stock targets from the day prediction, converted to option prices with Black-Scholes."""
    side = opt["side"]
    sign = 1 if side == "CALL" else -1
    S, K, T, iv = a["spot"], opt["strike"], opt["T"], opt["iv"]
    T_exit = max(T - HOLD_HOURS / (24 * 365), 1.0 / (24 * 365))

    now_px = bs_price(S, K, T, iv, side)
    stock_tp = S + sign * a["move"]
    stock_sl = S - sign * a["move"] * SL_MOVE_FRAC
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
                        "vega", "rho", "iv", "score", "top"])
        for side in ("CALL", "PUT"):
            it = res.get(side)
            if not it:
                continue
            o, p, a = it["opt"], it["plan"], it["a"]
            w.writerow([res["time"], res["mode"], o["symbol"], side, o["expiry"], o["strike"], round(a["spot"], 2),
                        round(a["pred_close"], 2), round(a["pred_high"], 2), round(a["pred_low"], 2),
                        p["entry"], p["sl"], p["tp"], p["profit_1lot"], round(o["delta"], 3), round(o["gamma"], 4),
                        round(o["theta"], 3), round(o["vega"], 3), round(o["rho"], 3), round(o["iv"], 3),
                        a["score"], res.get("top") == side])

# ----------------------------- SCAN -----------------------------

def scan_once(mode, pre_prev=None):
    now = dt.datetime.now(ET)
    with ThreadPoolExecutor(max_workers=6) as ex:
        analysed = [a for a in ex.map(lambda s: analyse(s, mode, pre_prev), SYMBOLS) if a]

    table = [dict(symbol=a["sym"], price=round(a["spot"], 2), gap_pct=round(a["gap"], 2), rsi=round(a["rsi"], 1),
                  rvol=round(a["rvol"], 2), score=a["score"], pre_score=a["pre_score"],
                  day_est=round(a["pred_close"], 2)) for a in analysed]

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

    res = {"mode": mode, "time": now.isoformat(timespec="seconds"), "CALL": None, "PUT": None, "top": None,
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
    if now.weekday() >= 5:
        return None
    t, today = now.time(), str(now.date())
    if SCAN_OPEN_TIME <= t < SCAN_OPEN_LATEST and not _done(state, "open", today):
        return "open"
    if SCAN_PRE_TIME <= t < MARKET_OPEN and not _done(state, "pre", today):
        return "pre"
    return None


def next_scan(now):
    for i in range(8):
        day = now.date() + dt.timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for key, t in (("pre", SCAN_PRE_TIME), ("open", SCAN_OPEN_TIME)):
            when = dt.datetime.combine(day, t, tzinfo=ET)
            if when > now:
                return key, when
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
    mode = due_scan(now, state)
    if not mode:
        return None
    today = str(now.date())
    if state.get("date") != today:
        state = {"date": today}
    state[mode] = {"running": True, "time": now.isoformat(timespec="seconds")}   # claim it
    save_state(state)
    try:
        res = scan_once(mode, pre_scores_today(state) if mode == "open" else None)
        state[mode] = res
        save_state(state)
        res["telegram"] = send_telegram(cfg, telegram_text(res))
        state[mode] = res
        save_state(state)
    except Exception as e:
        state[mode] = {"error": str(e), "time": now.isoformat(timespec="seconds"), "mode": mode}
        save_state(state)
    return mode

# ----------------------------- TELEGRAM TEXT (compact) -----------------------------

def telegram_text(res):
    when = dt.datetime.fromisoformat(res["time"])
    title = "PREMARKET (indicative)" if res["mode"] == "pre" else "OPEN +10min"
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

# ----------------------------- STREAMLIT UI -----------------------------

def main():
    st.set_page_config(page_title="US Call/Put Scanner", layout="wide")
    cfg = load_config()

    st.title("📈 US Top-50 Call / Put Scanner")
    st.caption("2 scans per US trading day: premarket (09:00 ET) and open +10 min (09:40 ET, includes premarket result). "
               "Next expiry (not same day) - all Greeks - educational only, not advice.")

    with st.sidebar:
        auto = st.toggle("Auto scans (09:00 & 09:40 ET)", value=True)
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

    if auto:
        run_scheduled(cfg)

    manual = st.session_state.setdefault("manual", {})
    if run_pre or run_open:
        mode = "open" if run_open else "pre"
        with st.spinner(f"Scanning {len(SYMBOLS)} stocks..."):
            manual[mode] = scan_once(mode, pre_scores_today(load_state()) if mode == "open" else None)

    state = load_state()

    def newest(key):
        a, b = state.get(key), manual.get(key)
        if a and b and not a.get("running") and not a.get("error"):
            return b if b["time"] > a["time"] else a
        return a or b

    tab_open, tab_pre = st.tabs(["Open scan (09:40 ET)", "Premarket scan (09:00 ET)"])
    with tab_open:
        render_scan(newest("open"), "No open scan yet - runs automatically at 09:40 ET (or use 'Open scan now').")
    with tab_pre:
        render_scan(newest("pre"), "No premarket scan yet - runs automatically at 09:00 ET (or use 'Premarket scan now').")

    if auto:
        box = st.empty()
        while True:
            now = dt.datetime.now(ET)
            if due_scan(now, load_state()):
                st.rerun()
            key, when = next_scan(now)
            left = int((when - now).total_seconds())
            name = "Premarket scan" if key == "pre" else "Open scan"
            box.caption(f"Next: **{name}** at {when.strftime('%a %H:%M')} ET ({fmt_local(when)} your time) - in "
                        f"{left // 3600}h {(left % 3600) // 60:02d}m {left % 60:02d}s. Keep this page open.")
            time.sleep(1)


if __name__ == "__main__":
    main()
