# app.py
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
MIN_SCORE = 3                 # |score| (max 5) needed to call a trade
STOP_LOSS_PCT = 25.0          # suggested premium stop
TRADING_MIN_PER_DAY = 390     # used to scale daily theta to the holding window (if needed)

TARGET_DELTA = 0.70           # kept for reference, but Greeks not available via yfinance
MAX_SPREAD_PCT = 10.0
MIN_OPTION_VOLUME = 50

# Top 25 US stocks (you can edit this list)
SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "PG", "JPM", "HD", "MA",
    "XOM", "BAC", "PFE", "KO", "PEP",
    "CSCO", "ABBV", "ADBE", "NFLX", "CRM"
]
# --------------------------------------------------------------------


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs()
        ],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def market_open_now():
    n = dt.datetime.now(ET)
    if n.weekday() >= 5:
        return False
    t = n.time()
    # skip first 5 min and last 10 min
    return dt.time(9, 35) <= t <= dt.time(15, 50)


def fetch_1m_data(sym, lookback_days=2):
    # yfinance 1m data is only available for recent days
    try:
        df = yf.download(
            sym,
            period=f"{lookback_days}d",
            interval="1m",
            auto_adjust=False,
            progress=False
        )
        if df.empty:
            return None
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
        )
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["time_key"] = df.index.tz_convert(ET)
        return df
    except Exception:
        return None


def analyse(sym):
    df = fetch_1m_data(sym)
    if df is None or len(df) < 40:
        return None

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)

    close = df["close"]
    e9, e20 = ema(close, 9), ema(close, 20)
    macd = ema(close, 12) - ema(close, 26)
    hist = macd - ema(macd, 9)
    r = rsi(close)
    a = atr(df)

    today = dt.datetime.now(ET).date()
    td = df[df["time_key"].dt.date == today]
    td = td if len(td) > 5 else df.tail(60)

    tp = (td["high"] + td["low"] + td["close"]) / 3
    vwap = (tp * td["volume"]).cumsum().iloc[-1] / max(td["volume"].cumsum().iloc[-1], 1)

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

    # price projection: linear regression on last 12 closes, extended 5/10/15 min
    y = close.tail(12).values
    x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1)
    base = slope * (len(y) - 1) + icpt
    proj = {m: base + slope * m for m in (5, 10, 15)}
    band = a.iloc[-1] * np.sqrt(10)  # ~1 sigma for a 10-min window

    return dict(
        sym=sym,
        spot=spot,
        vwap=vwap,
        rsi=r.iloc[-1],
        score=score,
        p5=proj[5],
        p10=proj[10],
        p15=proj[15],
        band=band,
        atr=a.iloc[-1]
    )


def pick_option(sym, spot, side):
    """
    Use yfinance option chain for a single expiry:
    - Skip same-day expiry (0DTE) by picking the first future expiry.
    - Choose ITM contracts with tight spread and real volume.
    NOTE: Greeks are not available via yfinance, so we only use basic fields.
    """
    try:
        t = yf.Ticker(sym)
        expiries = t.options
        if not expiries:
            return None

        # pick the first expiry that is strictly after today
        today = dt.datetime.now(ET).date()
        future_expiries = []
        for e in expiries:
            try:
                d = dt.datetime.strptime(e, "%Y-%m-%d").date()
                if d > today:
                    future_expiries.append(e)
            except Exception:
                continue

        if not future_expiries:
            return None

        expiry = sorted(future_expiries)[0]

        chain = t.option_chain(expiry)
        df_opts = chain.calls if side == "CALL" else chain.puts
        if df_opts.empty:
            return None

        # ITM filter
        if side == "CALL":
            itm = df_opts[df_opts["strike"] < spot]
        else:
            itm = df_opts[df_opts["strike"] > spot]

        if itm.empty:
            return None

        itm = itm.assign(dist=(itm["strike"] - spot).abs()).sort_values("dist").head(6)

        # basic liquidity filters
        itm["spread_pct"] = (itm["ask"] - itm["bid"]) / itm["ask"].replace(0, np.nan) * 100
        itm = itm[
            (itm["bid"] > 0) &
            (itm["ask"] > 0) &
            (itm["spread_pct"] <= MAX_SPREAD_PCT) &
            (itm["volume"] >= MIN_OPTION_VOLUME)
        ]

        if itm.empty:
            return None

        # choose closest to target delta if impliedVolatility is present (rough proxy),
        # otherwise just pick the first row
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
    except Exception:
        return None


def send_telegram_alert(cfg, message):
    token = cfg.get("telegram_bot_token")
    chat_id = cfg.get("telegram_chat_id")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception:
        pass


def init_log_file():
    new_file = not os.path.exists(LOG_FILE)
    f = open(LOG_FILE, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow([
            "time_et", "symbol", "side", "expiry", "strike", "spot", "pred_15m",
            "bid", "ask", "target", "stop", "score",
            "iv", "vol", "oi", "spread"
        ])
    return f, w


def scan_once(cfg, writer):
    now_et = dt.datetime.now(ET)
    header = f"=== Scan {now_et:%Y-%m-%d %H:%M:%S} ET ==="
    print(header)
    log_lines = [header]

    ideas = []

    for sym in SYMBOLS:
        try:
            a = analyse(sym)
            if not a:
                line = f"{sym}: no data"
                print(line)
                log_lines.append(line)
                continue

            side = "CALL" if a["score"] >= MIN_SCORE else ("PUT" if a["score"] <= -MIN_SCORE else None)
            line = (
                f"{sym:5} ${a['spot']:.2f} | score {a['score']:+d} | RSI {a['rsi']:.0f} | "
                f"pred 5/10/15m: {a['p5']:.2f}/{a['p10']:.2f}/{a['p15']:.2f} (+/-{a['band']:.2f})"
            )

            if not side:
                line_no_trade = line + " | NO TRADE"
                print(line_no_trade)
                log_lines.append(line_no_trade)
                continue

            opt = pick_option(sym, a["spot"], side)
            if not opt:
                msg = line + f" | {side} signal but no liquid ITM contract"
                print(msg)
                log_lines.append(msg)
                continue

            # simple target based on projected move vs current spot
            dS = a["p15"] - a["spot"]
            entry = opt["ask"]
            # approximate premium change proportional to underlying move
            est_change = entry * (dS / max(a["spot"], 1e-9))
            target = max(entry + est_change, entry)
            stop = entry * (1 - STOP_LOSS_PCT / 100)
            rr = (target - entry) / (entry - stop) if entry > stop else 0

            # breakeven: underlying move needed to cover half the spread
            cost = (opt["ask"] - opt["bid"])
            be_move = cost / max(0.01 * a["spot"], 1e-9)  # rough proxy

            print(line)
            log_lines.append(line)

            opt_line = (
                f"   -> {side} {sym}  exp {opt['expiry']}  strike {opt['strike']}  "
                f"bid/ask {opt['bid']:.2f}/{opt['ask']:.2f}  vol {opt['vol']:.0f}  "
                f"OI {opt['oi']:.0f}  IV {opt['iv']:.3f}  spread {opt['spread']:.1f}%"
            )
            print(opt_line)
            log_lines.append(opt_line)

            rr_line = (
                f"      entry<= {entry:.2f}  est 15m change {est_change:+.2f}  target ~{target:.2f}  "
                f"stop {stop:.2f}  R:R {rr:.1f}  breakeven move ${be_move:.2f} (pred move ${abs(dS):.2f})"
            )
            print(rr_line)
            log_lines.append(rr_line)

            if abs(dS) < be_move:
                warn = "      WARNING: predicted move is smaller than cost to break even - weak trade"
                print(warn)
                log_lines.append(warn)
                continue

            ideas.append((abs(a["score"]) * rr, sym, side, opt))

            writer.writerow([
                now_et.isoformat(timespec="seconds"),
                sym,
                side,
                opt["expiry"],
                opt["strike"],
                round(a["spot"], 2),
                round(a["p15"], 2),
                opt["bid"],
                opt["ask"],
                round(target, 2),
                round(stop, 2),
                a["score"],
                round(opt["iv"], 4),
                opt["vol"],
                opt["oi"],
                round(opt["spread"], 2)
            ])
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            line = f"{sym}: error {e!r} at line {tb[-1].lineno}"
            print(line)
            log_lines.append(line)

    if ideas:
        ideas.sort(reverse=True)
        _, s, side, opt = ideas[0]
        best_msg = f"BEST IDEA THIS SCAN: {side} {s} exp {opt['expiry']} strike {opt['strike']}"
        print(best_msg)
        log_lines.append(best_msg)

        # Telegram alert
        alert_text = (
            f"*BEST IDEA*\n"
            f"{side} {s}\n"
            f"Expiry: {opt['expiry']}\n"
            f"Strike: {opt['strike']}\n"
            f"Bid/Ask: {opt['bid']:.2f}/{opt['ask']:.2f}\n"
        )
        send_telegram_alert(cfg, alert_text)
    else:
        msg = "No qualifying trade this scan."
        print(msg)
        log_lines.append(msg)

    return "\n".join(log_lines)


# ----------------------------- STREAMLIT APP -----------------------------


def main():
    st.set_page_config(page_title="US ITM Call/Put Scanner", layout="wide")

    st.title("US ITM Call/Put Scanner (yfinance + Telegram)")

    cfg = load_config()

    st.sidebar.header("Telegram Alerts")
    bot_token = st.sidebar.text_input(
        "Bot Token",
        value=cfg.get("telegram_bot_token", ""),
        type="password"
    )
    chat_id = st.sidebar.text_input(
        "Chat ID",
        value=cfg.get("telegram_chat_id", "")
    )

    if st.sidebar.button("Save Telegram Settings"):
        cfg["telegram_bot_token"] = bot_token.strip()
        cfg["telegram_chat_id"] = chat_id.strip()
        save_config(cfg)
        st.sidebar.success("Telegram settings saved (persisted to config.json).")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Scan interval: {SCAN_SECONDS} seconds")
    st.sidebar.write(f"Min score for trade: {MIN_SCORE}")
    st.sidebar.write(f"Stop loss: {STOP_LOSS_PCT:.0f}%")

    st.subheader("Watchlist (Top 25 US stocks)")
    st.write(", ".join(SYMBOLS))

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        run_once = st.button("Run Scan Once")

    with col2:
        auto_scan = st.checkbox("Auto-scan every 5 minutes (while page is open)")

    log_output = st.empty()

    # ensure log file exists
    f, writer = init_log_file()

    if run_once:
        if not market_open_now():
            log_output.warning("Market not in scan window (09:35–15:50 ET, Mon–Fri). Running anyway on latest data.")
        text = scan_once(cfg, writer)
        f.flush()
        log_output.text(text)

    if auto_scan:
        if not market_open_now():
            st.warning("Market not in scan window (09:35–15:50 ET, Mon–Fri). Auto-scan will still use latest data.")
        # simple loop with sleep; note: Streamlit reruns script, so we keep it minimal
        text = scan_once(cfg, writer)
        f.flush()
        log_output.text(text)
        time.sleep(SCAN_SECONDS)
        st.experimental_rerun()

    f.close()


if __name__ == "__main__":
    main()
