"""
Daily US ETF + macro brief.

Runs once a day after the US close:
  1. pulls daily closes for a fixed ETF universe
  2. computes RSI(14), 50/200-day moving averages, 5/20-day returns
  3. reads a few macro series from FRED (optional)
  4. sends a Telegram message and writes docs/index.html

Run locally:
    python main.py            # live data
    python main.py --demo     # synthetic data, no network, for previewing output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from urllib import request, parse, error

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

BROAD = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
}

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health care",
    "XLI": "Industrials",
    "XLY": "Consumer discretionary",
    "XLP": "Consumer staples",
    "XLU": "Utilities",
}

DEFENSIVE = {
    "TLT": "20yr Treasuries",
    "GLD": "Gold",
    "UUP": "US dollar",
}

VOL = {"^VIX": "VIX"}

ALL_TICKERS = {**BROAD, **SECTORS, **DEFENSIVE, **VOL}

# FRED series -> label. Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_SERIES = {
    "DGS10": "10-year yield",
    "DGS2": "2-year yield",
    "T10Y2Y": "10y minus 2y spread",
    "DFF": "Fed funds rate",
}

OUT_HTML = os.path.join("docs", "index.html")


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder's RSI. Returns the latest value, or nan if not enough data."""
    s = series.dropna()
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1])


def pct_change_over(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-1 - days] - 1) * 100)


def ma_distance(series: pd.Series, window: int) -> float:
    """How far the last close sits above/below its moving average, in percent."""
    s = series.dropna()
    if len(s) < window:
        return float("nan")
    ma = s.rolling(window).mean().iloc[-1]
    return float((s.iloc[-1] / ma - 1) * 100)


def cross_state(series: pd.Series) -> str:
    """Golden cross / death cross, and whether it is recent."""
    s = series.dropna()
    if len(s) < 200:
        return "n/a"
    ma50 = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()
    above = ma50 > ma200
    if not bool(above.iloc[-1]):
        state = "50 below 200"
    else:
        state = "50 above 200"
    flipped = above.iloc[-20:] != above.iloc[-21:-1].values
    if bool(flipped.any()):
        state += " (crossed in last month)"
    return state


@dataclass
class Row:
    ticker: str
    name: str
    group: str
    last: float = float("nan")
    chg_1d: float = float("nan")
    chg_5d: float = float("nan")
    chg_20d: float = float("nan")
    rsi14: float = float("nan")
    d_ma50: float = float("nan")
    d_ma200: float = float("nan")
    cross: str = "n/a"
    flags: list[str] = field(default_factory=list)


def build_row(ticker: str, name: str, group: str, closes: pd.Series) -> Row:
    r = Row(ticker=ticker, name=name, group=group)
    s = closes.dropna()
    if s.empty:
        return r
    r.last = float(s.iloc[-1])
    r.chg_1d = pct_change_over(s, 1)
    r.chg_5d = pct_change_over(s, 5)
    r.chg_20d = pct_change_over(s, 20)
    r.rsi14 = rsi(s)
    r.d_ma50 = ma_distance(s, 50)
    r.d_ma200 = ma_distance(s, 200)
    r.cross = cross_state(s)

    if r.rsi14 == r.rsi14:  # not nan
        if r.rsi14 >= 70:
            r.flags.append("RSI overbought")
        elif r.rsi14 <= 30:
            r.flags.append("RSI oversold")
    if "crossed in last month" in r.cross:
        r.flags.append("Golden cross" if "50 above" in r.cross else "Death cross")
    if r.d_ma200 == r.d_ma200 and abs(r.d_ma200) < 1.0:
        r.flags.append("Sitting on the 200-day")
    return r


# --------------------------------------------------------------------------
# Regime read
# --------------------------------------------------------------------------

def regime(rows: dict[str, Row]) -> tuple[str, list[str]]:
    """A blunt risk-on / risk-off read. Evidence is listed so you can disagree with it."""
    score = 0
    evidence: list[str] = []

    spy = rows.get("SPY")
    if spy and spy.d_ma200 == spy.d_ma200:
        if spy.d_ma200 > 0:
            score += 1
            evidence.append(f"S&P is {spy.d_ma200:+.1f}% vs its 200-day")
        else:
            score -= 1
            evidence.append(f"S&P is {spy.d_ma200:+.1f}% vs its 200-day")

    spy20 = rows["SPY"].chg_20d if "SPY" in rows else float("nan")
    tlt20 = rows["TLT"].chg_20d if "TLT" in rows else float("nan")
    if spy20 == spy20 and tlt20 == tlt20:
        gap = spy20 - tlt20
        if gap > 0:
            score += 1
        else:
            score -= 1
        evidence.append(f"Stocks vs bonds over 20 days: {gap:+.1f} points")

    iwm20 = rows["IWM"].chg_20d if "IWM" in rows else float("nan")
    if iwm20 == iwm20 and spy20 == spy20:
        if iwm20 > spy20:
            score += 1
            evidence.append("Small caps leading large caps")
        else:
            evidence.append("Large caps leading small caps")

    vix = rows.get("^VIX")
    if vix and vix.last == vix.last:
        if vix.last < 16:
            score += 1
            evidence.append(f"VIX at {vix.last:.1f}, calm")
        elif vix.last > 25:
            score -= 2
            evidence.append(f"VIX at {vix.last:.1f}, stressed")
        else:
            evidence.append(f"VIX at {vix.last:.1f}")

    gld = rows.get("GLD")
    if gld and gld.chg_20d == gld.chg_20d and spy20 == spy20 and gld.chg_20d > spy20 + 3:
        score -= 1
        evidence.append("Gold outrunning stocks")

    if score >= 3:
        label = "Risk on"
    elif score >= 1:
        label = "Leaning risk on"
    elif score >= -1:
        label = "Mixed"
    elif score >= -3:
        label = "Leaning risk off"
    else:
        label = "Risk off"
    return label, evidence


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_prices(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        print(f"warning: no data returned for {missing}", file=sys.stderr)
    return closes


def demo_prices(tickers: list[str], days: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    n = len(idx)
    out = {}
    for i, t in enumerate(tickers):
        drift = 0.0004 if t not in ("^VIX", "UUP") else -0.0001
        vol = 0.02 if t == "^VIX" else 0.009
        steps = rng.normal(drift, vol, n)
        base = 18.0 if t == "^VIX" else 50.0 + i * 11
        out[t] = base * np.exp(np.cumsum(steps))
    return pd.DataFrame(out, index=idx)


def load_macro(api_key: str | None) -> dict[str, dict]:
    """Latest value and 1-month change for each FRED series. Returns {} without a key."""
    if not api_key:
        return {}
    start = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%d")
    out: dict[str, dict] = {}
    for sid, label in FRED_SERIES.items():
        q = parse.urlencode({
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        })
        url = f"https://api.stlouisfed.org/fred/series/observations?{q}"
        try:
            with request.urlopen(url, timeout=20) as resp:
                obs = json.load(resp).get("observations", [])
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"warning: FRED {sid} failed: {exc}", file=sys.stderr)
            continue
        vals = [(o["date"], float(o["value"])) for o in obs if o["value"] not in (".", "")]
        if not vals:
            continue
        date, latest = vals[-1]
        month_ago = vals[max(0, len(vals) - 22)][1]
        out[sid] = {
            "label": label,
            "value": latest,
            "change": latest - month_ago,
            "date": date,
        }
    return out


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    req = request.Request(url, data=body)
    try:
        with request.urlopen(req, timeout=20) as resp:
            resp.read()
    except error.HTTPError as exc:
        print(f"Telegram rejected the message: {exc.read().decode()[:300]}", file=sys.stderr)
        raise


def fmt_pct(v: float, digits: int = 1) -> str:
    return "—" if v != v else f"{v:+.{digits}f}%"


def build_message(rows: dict[str, Row], macro: dict, label: str,
                  evidence: list[str], page_url: str | None) -> str:
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    lines = [f"<b>{label}</b> · {today}", ""]

    for t in BROAD:
        r = rows[t]
        lines.append(f"{r.ticker} {r.last:,.2f}  {fmt_pct(r.chg_1d)}   "
                     f"5d {fmt_pct(r.chg_5d)}  RSI {r.rsi14:.0f}")
    vix = rows.get("^VIX")
    if vix and vix.last == vix.last:
        lines.append(f"VIX {vix.last:.1f}")

    ranked = sorted(
        (rows[t] for t in SECTORS if rows[t].chg_20d == rows[t].chg_20d),
        key=lambda r: r.chg_20d,
        reverse=True,
    )
    if ranked:
        lines += ["", "<b>Sectors, 20 days</b>"]
        for r in ranked[:3]:
            lines.append(f"  {r.ticker} {r.name} {fmt_pct(r.chg_20d)}")
        lines.append("  ...")
        for r in ranked[-2:]:
            lines.append(f"  {r.ticker} {r.name} {fmt_pct(r.chg_20d)}")

    flagged = [r for r in rows.values() if r.flags]
    if flagged:
        lines += ["", "<b>Worth a look</b>"]
        for r in flagged:
            lines.append(f"  {r.ticker}: {', '.join(r.flags)}")

    if macro:
        lines += ["", "<b>Macro</b>"]
        for sid, m in macro.items():
            arrow = "↑" if m["change"] > 0.01 else ("↓" if m["change"] < -0.01 else "→")
            lines.append(f"  {m['label']}: {m['value']:.2f} {arrow} ({m['change']:+.2f} in a month)")

    lines += ["", "<i>Why:</i> " + "; ".join(evidence)]
    if page_url:
        lines += ["", f"Full table: {page_url}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Dashboard page
# --------------------------------------------------------------------------

def cls(v: float) -> str:
    if v != v:
        return "flat"
    return "up" if v > 0 else ("down" if v < 0 else "flat")


def table_rows(rows: list[Row]) -> str:
    out = []
    for r in rows:
        flags = "".join(f"<span class='flag'>{f}</span>" for f in r.flags)
        rsi_cls = "hot" if r.rsi14 >= 70 else ("cold" if r.rsi14 <= 30 else "")
        out.append(f"""
        <tr>
          <th scope="row"><span class="tick">{r.ticker}</span><span class="nm">{r.name}</span>{flags}</th>
          <td class="num">{'—' if r.last != r.last else f'{r.last:,.2f}'}</td>
          <td class="num {cls(r.chg_1d)}">{fmt_pct(r.chg_1d)}</td>
          <td class="num {cls(r.chg_5d)}">{fmt_pct(r.chg_5d)}</td>
          <td class="num {cls(r.chg_20d)}">{fmt_pct(r.chg_20d)}</td>
          <td class="num rsi {rsi_cls}">{'—' if r.rsi14 != r.rsi14 else f'{r.rsi14:.0f}'}</td>
          <td class="num {cls(r.d_ma50)}">{fmt_pct(r.d_ma50)}</td>
          <td class="num {cls(r.d_ma200)}">{fmt_pct(r.d_ma200)}</td>
        </tr>""")
    return "".join(out)


def build_page(rows: dict[str, Row], macro: dict, label: str, evidence: list[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    macro_html = ""
    if macro:
        cards = "".join(f"""
        <div class="macro-item">
          <div class="macro-val">{m['value']:.2f}</div>
          <div class="macro-lbl">{m['label']}</div>
          <div class="macro-chg {cls(m['change'])}">{m['change']:+.2f} over a month</div>
        </div>""" for m in macro.values())
        macro_html = f"<section class='macro'><h2>Rates</h2><div class='macro-grid'>{cards}</div></section>"

    ev = "".join(f"<li>{e}</li>" for e in evidence)

    def section(title: str, keys: dict) -> str:
        body = table_rows([rows[k] for k in keys if k in rows])
        return f"""
      <section>
        <h2>{title}</h2>
        <div class="scroll">
        <table>
          <thead>
            <tr><th scope="col">Fund</th><th scope="col">Last</th><th scope="col">1 day</th>
            <th scope="col">5 days</th><th scope="col">20 days</th><th scope="col">RSI</th>
            <th scope="col">vs 50-day</th><th scope="col">vs 200-day</th></tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
        </div>
      </section>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Market brief</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --ground: #edeee9;
    --ink: #1b2a2f;
    --muted: #6b7a7f;
    --rule: #c9cdc5;
    --up: #0e6f52;
    --down: #a83c2c;
    --panel: #f6f7f3;
    box-sizing: border-box;
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #141a1c; --ink: #e6e9e3; --muted: #8b9a9e;
      --rule: #2b3639; --up: #4fbf92; --down: #e07a63; --panel: #1b2325;
    }}
  }}
  *, *::before, *::after {{ box-sizing: inherit; }}
  body {{
    margin: 0; background: var(--ground); color: var(--ink);
    font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 14px; line-height: 1.5;
    -webkit-text-size-adjust: 100%;
  }}
  .wrap {{ max-width: 62rem; margin: 0 auto; padding: 2rem 1.1rem 4rem; }}

  header {{ border-bottom: 2px solid var(--ink); padding-bottom: 1.4rem; }}
  .verdict {{
    font-family: "Bricolage Grotesque", system-ui, sans-serif;
    font-weight: 700; font-size: clamp(2.6rem, 11vw, 5rem);
    line-height: 0.95; letter-spacing: -0.03em; margin: 0.3rem 0 0.9rem;
  }}
  .stamp {{ color: var(--muted); font-size: 12px; }}
  .why {{ margin: 0; padding-left: 1.1rem; color: var(--muted); font-size: 13px; }}
  .why li {{ margin-bottom: 0.15rem; }}

  h2 {{
    font-family: "Bricolage Grotesque", system-ui, sans-serif;
    font-weight: 500; font-size: 1.05rem; letter-spacing: -0.01em;
    margin: 2.4rem 0 0.6rem;
  }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 40rem; }}
  th, td {{ text-align: right; padding: 0.5rem 0.45rem; border-bottom: 1px solid var(--rule); }}
  thead th {{
    font-weight: 500; font-size: 11px; color: var(--muted);
    border-bottom: 1px solid var(--ink); white-space: nowrap;
  }}
  tbody th {{ text-align: left; font-weight: 400; }}
  .tick {{ font-weight: 600; margin-right: 0.5rem; }}
  .nm {{ color: var(--muted); font-size: 12px; }}
  .num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .up {{ color: var(--up); }}
  .down {{ color: var(--down); }}
  .flat {{ color: var(--muted); }}
  .rsi.hot {{ color: var(--down); font-weight: 600; }}
  .rsi.cold {{ color: var(--up); font-weight: 600; }}
  .flag {{
    display: inline-block; margin-left: 0.4rem; padding: 0.05rem 0.4rem;
    border: 1px solid var(--rule); border-radius: 2px;
    font-size: 10px; color: var(--muted);
  }}

  .macro-grid {{ display: grid; gap: 1px; background: var(--rule);
    grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); border: 1px solid var(--rule); }}
  .macro-item {{ background: var(--panel); padding: 0.9rem 0.8rem; }}
  .macro-val {{ font-family: "Bricolage Grotesque", system-ui, sans-serif;
    font-weight: 700; font-size: 1.7rem; letter-spacing: -0.02em; }}
  .macro-lbl {{ font-size: 12px; color: var(--muted); }}
  .macro-chg {{ font-size: 11px; margin-top: 0.3rem; }}

  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule);
    color: var(--muted); font-size: 11px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="stamp">{stamp}</div>
    <h1 class="verdict">{label}</h1>
    <ul class="why">{ev}</ul>
  </header>
  {section("Broad market", BROAD)}
  {section("Sectors", SECTORS)}
  {section("Bonds, gold, dollar", DEFENSIVE)}
  {macro_html}
  <footer>
    Prices from Yahoo Finance, delayed. Rates from FRED.
    This page reports numbers; it does not tell you what to buy.
  </footer>
</div>
</body>
</html>"""


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data, send nothing")
    args = ap.parse_args()

    tickers = list(ALL_TICKERS)
    closes = demo_prices(tickers) if args.demo else load_prices(tickers)

    rows: dict[str, Row] = {}
    for t, name in ALL_TICKERS.items():
        group = "broad" if t in BROAD else "sector" if t in SECTORS else "other"
        series = closes[t] if t in closes.columns else pd.Series(dtype=float)
        rows[t] = build_row(t, name, group, series)

    macro = {} if args.demo else load_macro(os.environ.get("FRED_API_KEY"))
    if args.demo:
        macro = {
            "DGS10": {"label": "10-year yield", "value": 4.21, "change": 0.13, "date": "demo"},
            "T10Y2Y": {"label": "10y minus 2y spread", "value": 0.55, "change": -0.08, "date": "demo"},
        }

    label, evidence = regime(rows)

    os.makedirs("docs", exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(build_page(rows, macro, label, evidence))
    print(f"wrote {OUT_HTML}")

    page_url = os.environ.get("PAGE_URL")
    text = build_message(rows, macro, label, evidence, page_url)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if args.demo or not (token and chat_id):
        print("\n--- Telegram message ---\n")
        print(text)
        if not args.demo:
            print("\n(no TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID set, so nothing was sent)",
                  file=sys.stderr)
        return 0

    send_telegram(token, chat_id, text)
    print("sent to Telegram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
