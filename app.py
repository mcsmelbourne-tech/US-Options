"""
US Top-25 Call/Put Scanner (yfinance + Greeks)

- Scans 25 big US stocks, scores direction on 1-min data (EMA, VWAP, RSI, MACD, momentum)
- Uses the NEXT expiry after today (never same-day)
- Picks the best CALL and best PUT by Greeks (delta, gamma, theta, vega computed with Black-Scholes
  from yfinance implied volatility), spread and liquidity
- Shows each pick in a box: contract, Greeks, entry, stop loss, take profit (+ profit for 1 lot), exit rule
- Auto-scans every 30 minutes and sends the two picks to Telegram

Run:  streamlit run us_itm_scanner.py
Needs: pip install streamlit yfinance pandas numpy requests
Educational use only - not financial advice.
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

LOG_FILE = "scan_log.csv"
CONFIG_FILE = "config.json"

SCAN_SECONDS = 30 * 60        # scan every 30 minutes
MIN_SCORE = 3                 # direction score needed for a "strong" signal
RISK_FREE = 0.04              # used for Greeks
CONTRACT_SIZE = 100           # 1 lot = 1 contract = 100 shares

STOP_LOSS_PCT = 25.0          # option stop loss, % below entry
TAKE_PROFIT_PCT = 50.0        # option take profit, % above entry

TARGET_DELTA = 0.65           # sweet spot: ITM, moves well with the stock
DELTA_MIN, DELTA_MAX = 0.55, 0.80
MAX_SPREAD_PCT = 10.0
MIN_OI = 100

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "PG", "JPM", "HD", "MA",
    "XOM", "BAC", "PFE", "KO", "PEP",
    "CSCO", "ABBV", "ADBE", "NFLX", "CRM",
]

# ----------------------------- CONFIG / TELEGRAM -----------------------------

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def send_telegram(cfg, msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg},
            timeout=15,
        )
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
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

# ----------------------------- GREEKS (Black-Scholes) -----------------------------

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def greeks(S, K, T, sigma, side, r=RISK_FREE):
    """Returns delta, gamma, theta (per day, per share), vega (per 1 vol point, per share)."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    gamma = _npdf(d1) / (S * sq)
    vega = S * _npdf(d1) * math.sqrt(T) / 100.0
    if side == "CALL":
        delta = _ncdf(d1)
        theta = (-S * _npdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365.0
    else:
        delta = _ncdf(d1) - 1.0
        theta = (-S * _npdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365.0
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega)

# ----------------------------- DATA FETCH -----------------------------

def fetch_ohlc(sym):
    t = yf.Ticker(sym)
    df = t.history(period="2d", interval="1m", prepost=False)
    if df is None or df.empty:
        df = t.history(period="5d", interval="5m", prepost=False)
    if df is None or df.empty:
        return None
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    idx = df.index
    df["time_key"] = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize(ET)
    return df

# ----------------------------- DIRECTION ANALYSIS -----------------------------

def analyse(sym):
    try:
        df = fetch_ohlc(sym)
        if df is None or len(df) < 40:
            return None

        close = df["close"]
        e9, e20 = ema(close, 9), ema(close, 20)
        macd = ema(close, 12) - ema(close, 26)
        hist = macd - ema(macd, 9)
        r = rsi(close)
        a = atr(df)

        last_day = df["time_key"].dt.date.iloc[-1]
        td = df[df["time_key"].dt.date == last_day]
        td = td if len(td) > 5 else df.tail(60)

        tp = (td["high"] + td["low"] + td["close"]) / 3
        cum_vol = float(td["volume"].cumsum().iloc[-1])
        vwap = float((tp * td["volume"]).cumsum().iloc[-1]) / cum_vol if cum_vol > 0 else float(tp.iloc[-1])

        spot = float(close.iloc[-1])
        rsi_last = float(r.iloc[-1])
        hist_last, hist_prev = float(hist.iloc[-1]), float(hist.iloc[-2])

        score = 0
        score += 1 if e9.iloc[-1] > e20.iloc[-1] else -1
        score += 1 if spot > vwap else -1
        score += 1 if rsi_last > 55 else (-1 if rsi_last < 45 else 0)
        score += 1 if (hist_last > 0 and hist_last > hist_prev) else (
            -1 if (hist_last < 0 and hist_last < hist_prev) else 0)
        mom = (spot - float(close.iloc[-6])) / max(float(a.iloc[-1]), 1e-9)
        score += 1 if mom > 0.5 else (-1 if mom < -0.5 else 0)

        y = close.tail(12).values.astype(float)
        x = np.arange(len(y))
        slope, icpt = np.polyfit(x, y, 1)
        p15 = slope * (len(y) - 1) + icpt + slope * 15

        return dict(sym=sym, spot=spot, vwap=vwap, rsi=rsi_last, score=score, p15=float(p15))
    except Exception:
        return None

# ----------------------------- OPTION PICKER (Greeks based) -----------------------------

def pick_option(sym, spot, side):
    """Next expiry AFTER today -> best contract by delta / gamma / theta / spread / liquidity."""
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
            oi = int(o.get("openInterest") or 0) if not pd.isna(o.get("openInterest")) else 0
            vol = int(o.get("volume") or 0) if not pd.isna(o.get("volume")) else 0

            market_open = bid > 0 and ask > 0
            if market_open:
                mid = (bid + ask) / 2
                spread = (ask - bid) / ask * 100
                entry = ask
            elif last > 0:                       # market closed: fall back to last price
                mid, spread, entry = last, 0.0, last
            else:
                continue

            if not (0.05 <= iv <= 3.0) or oi < MIN_OI or spread > MAX_SPREAD_PCT:
                continue

            g = greeks(spot, float(o["strike"]), T, iv, side)
            if not g:
                continue
            d = abs(g["delta"])
            if not (DELTA_MIN <= d <= DELTA_MAX):
                continue

            theta_pct = abs(g["theta"]) / max(mid, 0.01) * 100      # daily decay as % of premium
            gamma_pct = g["gamma"] * spot / max(mid, 0.01) * 100    # gamma relative to premium
            quality = (
                100
                - abs(d - TARGET_DELTA) * 100      # near the target delta
                - spread * 2                        # tight spread
                - theta_pct * 4                     # low time decay
                + gamma_pct * 2                     # responsive to stock moves
                + math.log10(oi + vol + 1) * 3      # liquidity
            )
            rows.append(dict(
                symbol=sym, side=side, expiry=expiry, dte=dte,
                strike=float(o["strike"]), bid=bid, ask=ask, last=last, entry=entry,
                iv=iv, vol=vol, oi=oi, spread=spread, market_open=market_open,
                delta=g["delta"], gamma=g["gamma"], theta=g["theta"], vega=g["vega"],
                quality=quality,
            ))

        if not rows:
            return None
        return max(rows, key=lambda r: r["quality"])
    except Exception:
        return None


def trade_plan(opt):
    entry = round(opt["entry"], 2)
    sl = round(entry * (1 - STOP_LOSS_PCT / 100), 2)
    tp = round(entry * (1 + TAKE_PROFIT_PCT / 100), 2)
    delta = max(abs(opt["delta"]), 0.05)
    sign = 1 if opt["side"] == "CALL" else -1
    return dict(
        entry=entry, sl=sl, tp=tp,
        profit_1lot=round((tp - entry) * CONTRACT_SIZE, 2),
        loss_1lot=round((entry - sl) * CONTRACT_SIZE, 2),
        cost_1lot=round(entry * CONTRACT_SIZE, 2),
        stock_sl=None, stock_tp=None,   # filled by caller with spot
        stock_sl_move=sign * -(entry - sl) / delta,
        stock_tp_move=sign * (tp - entry) / delta,
    )

# ----------------------------- LOGGING -----------------------------

def log_pick(item):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_et", "symbol", "side", "expiry", "strike", "spot", "pred_15m", "entry",
                        "stop_loss", "take_profit", "profit_1lot", "delta", "gamma", "theta", "vega",
                        "iv", "vol", "oi", "spread", "score"])
        o, p = item["opt"], item["plan"]
        w.writerow([dt.datetime.now(ET).isoformat(timespec="seconds"), o["symbol"], o["side"], o["expiry"],
                    o["strike"], round(item["spot"], 2), round(item["p15"], 2), p["entry"], p["sl"], p["tp"],
                    p["profit_1lot"], round(o["delta"], 3), round(o["gamma"], 4), round(o["theta"], 3),
                    round(o["vega"], 3), round(o["iv"], 3), o["vol"], o["oi"], round(o["spread"], 1),
                    item["score"]])

# ----------------------------- SCAN -----------------------------

def scan_once():
    with ThreadPoolExecutor(max_workers=5) as ex:
        analysed = [a for a in ex.map(analyse, SYMBOLS) if a]

    out = {"CALL": None, "PUT": None, "table": [], "time": dt.datetime.now(ET)}
    out["table"] = [dict(symbol=a["sym"], spot=round(a["spot"], 2), score=a["score"],
                         rsi=round(a["rsi"], 1)) for a in analysed]

    for side, order in (("CALL", sorted(analysed, key=lambda a: -a["score"])),
                        ("PUT", sorted(analysed, key=lambda a: a["score"]))):
        for a in order[:6]:                       # try best-scored stocks until one has a good contract
            if (side == "CALL" and a["score"] <= 0) or (side == "PUT" and a["score"] >= 0):
                break
            opt = pick_option(a["sym"], a["spot"], side)
            if not opt:
                continue
            plan = trade_plan(opt)
            plan["stock_sl"] = round(a["spot"] + plan["stock_sl_move"], 2)
            plan["stock_tp"] = round(a["spot"] + plan["stock_tp_move"], 2)
            out[side] = dict(opt=opt, plan=plan, spot=a["spot"], p15=a["p15"], score=a["score"],
                             strong=abs(a["score"]) >= MIN_SCORE)
            log_pick(out[side])
            break
    return out

# ----------------------------- DISPLAY -----------------------------

def telegram_text(item):
    o, p = item["opt"], item["plan"]
    tag = "STRONG" if item["strong"] else "weak"
    return (
        f"{'🟢' if o['side'] == 'CALL' else '🔴'} {o['side']} {o['symbol']} ({tag}, score {item['score']})\n"
        f"Strike {o['strike']} | Exp {o['expiry']} ({o['dte']}d)\n"
        f"Delta {o['delta']:.2f} | Gamma {o['gamma']:.3f} | Theta {o['theta']:.2f} | IV {o['iv']*100:.0f}%\n"
        f"Entry {p['entry']} | SL {p['sl']} | TP {p['tp']}\n"
        f"1 lot: cost ${p['cost_1lot']:.0f} | profit at TP +${p['profit_1lot']:.0f} | loss at SL -${p['loss_1lot']:.0f}\n"
        f"Stock now {item['spot']:.2f} | stock SL {p['stock_sl']} | stock TP {p['stock_tp']}"
    )


def draw_box(item, side):
    icon = "🟢" if side == "CALL" else "🔴"
    if not item:
        with st.container(border=True):
            st.subheader(f"{icon} {side}")
            st.info("No liquid contract found for this side (next expiry, delta 0.55-0.80, spread <= 10%).")
        return

    o, p = item["opt"], item["plan"]
    with st.container(border=True):
        tag = "STRONG signal" if item["strong"] else "weak signal"
        st.subheader(f"{icon} {side} - {o['symbol']}  ({tag}, score {item['score']})")
        if not o["market_open"]:
            st.caption("Market closed - using last traded price; re-check bid/ask when it opens.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Contract**")
            st.write(f"Strike: **{o['strike']}**")
            st.write(f"Expiry: **{o['expiry']}** ({o['dte']} days)")
            st.write(f"Bid / Ask: {o['bid']:.2f} / {o['ask']:.2f}")
            st.write(f"Spread: {o['spread']:.1f}%")
            st.write(f"Volume / OI: {o['vol']} / {o['oi']}")
            st.write(f"Stock: {item['spot']:.2f} -> 15m est. {item['p15']:.2f}")
        with c2:
            st.markdown("**Greeks**")
            st.write(f"Delta: **{o['delta']:.3f}**")
            st.write(f"Gamma: {o['gamma']:.4f}")
            st.write(f"Theta: {o['theta']:.3f} / day")
            st.write(f"Vega: {o['vega']:.3f}")
            st.write(f"IV: {o['iv']*100:.1f}%")
        with c3:
            st.markdown("**Trade plan (1 lot = 100 shares)**")
            st.write(f"Entry (buy @ ask): **${p['entry']:.2f}**  (cost ${p['cost_1lot']:,.0f})")
            st.write(f"Stop loss: **${p['sl']:.2f}**  -> loss -${p['loss_1lot']:,.0f}")
            st.write(f"Take profit: **${p['tp']:.2f}**  -> profit **+${p['profit_1lot']:,.0f}**")
            st.write(f"Stock SL / TP: {p['stock_sl']} / {p['stock_tp']}")
            st.write("Exit: at TP or SL, or if the signal flips, or by 15:45 ET at the latest.")

# ----------------------------- STREAMLIT UI -----------------------------

def main():
    st.set_page_config(page_title="US Call/Put Scanner", layout="wide")
    cfg = load_config()

    st.title("📈 US Top-25 Call / Put Scanner")
    st.caption("Next expiry (not same day) - Greeks-ranked contracts - scans every 30 min - educational only, not advice")

    with st.sidebar:
        auto = st.toggle("Auto-scan every 30 min", value=True)
        with st.expander("Telegram"):
            bot = st.text_input("Bot token", value=cfg.get("telegram_bot_token", ""), type="password")
            chat = st.text_input("Chat ID", value=cfg.get("telegram_chat_id", ""))
            if st.button("Save"):
                cfg["telegram_bot_token"], cfg["telegram_chat_id"] = bot.strip(), chat.strip()
                save_config(cfg)
                st.success("Saved")
        st.caption(f"SL {STOP_LOSS_PCT:.0f}% | TP {TAKE_PROFIT_PCT:.0f}% | watchlist: {', '.join(SYMBOLS)}")

    due = time.time() - st.session_state.get("last_scan", 0) >= SCAN_SECONDS
    if st.button("Run Scan Now") or (auto and due):
        with st.spinner("Scanning 25 stocks..."):
            res = scan_once()
        st.session_state["res"] = res
        st.session_state["last_scan"] = time.time()
        sent = False
        for side in ("CALL", "PUT"):
            if res[side]:
                sent = send_telegram(cfg, telegram_text(res[side])) or sent
        st.session_state["sent"] = sent

    res = st.session_state.get("res")
    if res:
        st.write(f"Last scan: **{res['time'].strftime('%a %d %b %H:%M:%S')} ET**"
                 + ("  |  Telegram alert sent ✅" if st.session_state.get("sent") else ""))
        col_call, col_put = st.columns(2)
        with col_call:
            draw_box(res["CALL"], "CALL")
        with col_put:
            draw_box(res["PUT"], "PUT")
        with st.expander("All 25 stocks (direction score)"):
            st.dataframe(pd.DataFrame(res["table"]).sort_values("score", ascending=False),
                         use_container_width=True)
    else:
        st.info("Press 'Run Scan Now' or leave auto-scan on.")

    if auto:
        box = st.empty()
        while True:
            left = int(SCAN_SECONDS - (time.time() - st.session_state.get("last_scan", 0)))
            if left <= 0:
                st.rerun()
            box.caption(f"Next scan in {left // 60}m {left % 60:02d}s")
            time.sleep(1)


if __name__ == "__main__":
    main()
