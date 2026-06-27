#!/usr/bin/env python3
"""
Crypto trading bot -- EMA-crossover + RSI-momentumfilter + ATR risk management.
Alles in 1 bestand, bedoeld om te draaien via een GitHub Actions schedule
(elke run = 1 cyclus, state wordt opgeslagen in data/paper_state.json en
door de workflow terug naar de repo gecommit).

Standaard: paper trading (simulatie, geen echte orders).
Live trading vereist EXPLICIET: mode: live in config.yaml +
EXCHANGE_API_KEY/EXCHANGE_API_SECRET + CONFIRM_LIVE_TRADING=YES_I_UNDERSTAND_THE_RISK
als environment variables (in GitHub: Settings > Secrets and variables > Actions).

Dit is geen financieel advies. Crypto-trading is risicovol.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
import yaml


# ============================================================
# Indicatoren
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ============================================================
# Strategie
# ============================================================

@dataclass
class Signal:
    action: str
    reason: str
    price: float
    atr: float


class EmaRsiAtrStrategy:
    def __init__(self, params: dict):
        self.ema_fast = params["ema_fast"]
        self.ema_slow = params["ema_slow"]
        self.rsi_period = params["rsi_period"]
        self.rsi_buy_threshold = params["rsi_buy_threshold"]
        self.rsi_sell_threshold = params["rsi_sell_threshold"]
        self.atr_period = params["atr_period"]
        self.atr_sl_multiplier = params["atr_sl_multiplier"]
        self.atr_tp_multiplier = params["atr_tp_multiplier"]

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["atr"] = atr(df, self.atr_period)
        return df

    def generate_entry_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < max(self.ema_slow, self.rsi_period, self.atr_period) + 2:
            return None
        prev, curr = df.iloc[-2], df.iloc[-1]
        crossed_up = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
        if crossed_up and curr["rsi"] > self.rsi_buy_threshold:
            return Signal(
                "buy",
                f"EMA{self.ema_fast} kruist boven EMA{self.ema_slow}, RSI={curr['rsi']:.1f} > {self.rsi_buy_threshold}",
                curr["close"], curr["atr"],
            )
        return None

    def check_exit_signal(self, df: pd.DataFrame, stop_loss: float, take_profit: float) -> Optional[Signal]:
        curr, prev = df.iloc[-1], df.iloc[-2]
        price = curr["close"]
        if price <= stop_loss:
            return Signal("sell", f"Stop-loss geraakt ({price:.2f} <= {stop_loss:.2f})", price, curr["atr"])
        if price >= take_profit:
            return Signal("sell", f"Take-profit geraakt ({price:.2f} >= {take_profit:.2f})", price, curr["atr"])
        crossed_down = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]
        if crossed_down:
            return Signal("sell", f"EMA{self.ema_fast} kruist onder EMA{self.ema_slow} (trendomkeer)", price, curr["atr"])
        return None

    def compute_stop_and_target(self, entry_price: float, atr_value: float):
        return (entry_price - atr_value * self.atr_sl_multiplier,
                entry_price + atr_value * self.atr_tp_multiplier)


# ============================================================
# Risk management
# ============================================================

@dataclass
class RiskParams:
    risk_per_trade_pct: float
    max_position_pct: float


def position_size(balance_quote: float, entry_price: float, stop_loss: float, risk_params: RiskParams) -> float:
    stop_distance = entry_price - stop_loss
    if stop_distance <= 0 or balance_quote <= 0:
        return 0.0
    risk_amount = balance_quote * (risk_params.risk_per_trade_pct / 100)
    size_by_risk = risk_amount / stop_distance
    max_size_by_cap = (balance_quote * (risk_params.max_position_pct / 100)) / entry_price
    return max(0.0, min(size_by_risk, max_size_by_cap))


# ============================================================
# Paper trading portfolio (persistente state in JSON)
# ============================================================

@dataclass
class Position:
    side: str
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: str


@dataclass
class Trade:
    side: str
    amount: float
    price: float
    timestamp: str
    reason: str
    pnl_quote: Optional[float] = None


class PaperPortfolio:
    def __init__(self, state_path: str, starting_balance: float):
        self.state_path = state_path
        self.balance_quote = starting_balance
        self.position: Optional[Position] = None
        self.trade_log = []
        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                data = json.load(f)
            self.balance_quote = data["balance_quote"]
            self.position = Position(**data["position"]) if data.get("position") else None
            self.trade_log = [Trade(**t) for t in data.get("trade_log", [])]

    def save(self):
        data = {
            "balance_quote": self.balance_quote,
            "position": asdict(self.position) if self.position else None,
            "trade_log": [asdict(t) for t in self.trade_log],
        }
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def open_long(self, amount, price, stop_loss, take_profit, reason):
        cost = amount * price
        if cost > self.balance_quote:
            amount = self.balance_quote / price
            cost = amount * price
        self.balance_quote -= cost
        self.position = Position("long", amount, price, stop_loss, take_profit,
                                  datetime.now(timezone.utc).isoformat())
        self.trade_log.append(Trade("buy", amount, price, datetime.now(timezone.utc).isoformat(), reason))
        self.save()

    def close_long(self, price, reason):
        if not self.position:
            return
        proceeds = self.position.amount * price
        pnl = proceeds - self.position.amount * self.position.entry_price
        self.balance_quote += proceeds
        self.trade_log.append(Trade("sell", self.position.amount, price,
                                     datetime.now(timezone.utc).isoformat(), reason, pnl_quote=pnl))
        self.position = None
        self.save()

    def equity(self, current_price: float) -> float:
        if not self.position:
            return self.balance_quote
        return self.balance_quote + self.position.amount * current_price


# ============================================================
# Exchange-wrapper (ccxt) met veiligheidschecks voor live trading
# ============================================================

class ExchangeClient:
    def __init__(self, exchange_id: str, mode: str, log):
        self.exchange_id = exchange_id
        self.mode = mode
        self.log = log
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange
        import ccxt
        exchange_class = getattr(ccxt, self.exchange_id)
        config = {"enableRateLimit": True}
        if self.mode == "live":
            api_key, api_secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
            if not api_key or not api_secret:
                raise RuntimeError("Live mode vereist EXCHANGE_API_KEY en EXCHANGE_API_SECRET als env vars.")
            config["apiKey"], config["secret"] = api_key, api_secret
        self._exchange = exchange_class(config)
        return self._exchange

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        raw = self._get_exchange().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def _assert_live_confirmed(self):
        if os.environ.get("CONFIRM_LIVE_TRADING") != "YES_I_UNDERSTAND_THE_RISK":
            raise RuntimeError(
                "Live trading geblokkeerd: zet CONFIRM_LIVE_TRADING exact op "
                "'YES_I_UNDERSTAND_THE_RISK' (bewuste extra stap)."
            )

    def create_market_buy_order(self, symbol: str, amount: float):
        if self.mode != "live":
            raise RuntimeError("Alleen toegestaan in live mode.")
        self._assert_live_confirmed()
        self.log(f"LIVE ORDER: BUY {amount} {symbol}")
        return self._get_exchange().create_market_buy_order(symbol, amount)

    def create_market_sell_order(self, symbol: str, amount: float):
        if self.mode != "live":
            raise RuntimeError("Alleen toegestaan in live mode.")
        self._assert_live_confirmed()
        self.log(f"LIVE ORDER: SELL {amount} {symbol}")
        return self._get_exchange().create_market_sell_order(symbol, amount)


# ============================================================
# Hoofdlogica
# ============================================================

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} UTC | {msg}", flush=True)


def run_cycle(config, exchange, strategy, portfolio, risk_params):
    symbol, timeframe = config["symbol"], config["timeframe"]
    df = exchange.fetch_ohlcv_df(symbol, timeframe, limit=200)
    df = strategy.compute_indicators(df)
    current_price = df.iloc[-1]["close"]

    if portfolio.position is None:
        signal = strategy.generate_entry_signal(df)
        if signal and signal.action == "buy":
            stop_loss, take_profit = strategy.compute_stop_and_target(signal.price, signal.atr)
            amount = position_size(portfolio.balance_quote, signal.price, stop_loss, risk_params)
            if amount <= 0:
                log("Entry-signaal genegeerd: positiegrootte 0 (te weinig saldo / ongeldige stop).")
                return
            log(f"BUY: {signal.reason} | amount={amount:.6f} @ {signal.price:.2f} | SL={stop_loss:.2f} TP={take_profit:.2f}")
            if config["mode"] == "live":
                exchange.create_market_buy_order(symbol, amount)
            portfolio.open_long(amount, signal.price, stop_loss, take_profit, signal.reason)
        else:
            log(f"Geen positie, geen entry-signaal. Prijs={current_price:.2f}")
    else:
        pos = portfolio.position
        signal = strategy.check_exit_signal(df, pos.stop_loss, pos.take_profit)
        if signal and signal.action == "sell":
            log(f"SELL: {signal.reason} | amount={pos.amount:.6f} @ {signal.price:.2f}")
            if config["mode"] == "live":
                exchange.create_market_sell_order(symbol, pos.amount)
            portfolio.close_long(signal.price, signal.reason)
        else:
            log(f"Positie open: entry={pos.entry_price:.2f} prijs={current_price:.2f} "
                f"SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f} | equity={portfolio.equity(current_price):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log(f"Bot gestart in '{config['mode'].upper()}' mode | {config['symbol']} | {config['timeframe']}")
    if config["mode"] == "live":
        log("LIVE MODE ACTIEF -- er kunnen echte orders met echt geld geplaatst worden.")

    exchange = ExchangeClient(config["exchange"], config["mode"], log)
    strategy = EmaRsiAtrStrategy(config["strategy"])
    risk_params = RiskParams(**config["risk"])
    portfolio = PaperPortfolio(config["paper"]["state_file"], config["paper"]["starting_balance"])

    try:
        run_cycle(config, exchange, strategy, portfolio, risk_params)
    except Exception as e:
        log(f"FOUT in trading cyclus: {e}")
        sys.exit(1)

    if not args.once:
        import time
        poll_seconds = config.get("loop", {}).get("poll_seconds", 60)
        while True:
            time.sleep(poll_seconds)
            try:
                run_cycle(config, exchange, strategy, portfolio, risk_params)
            except Exception as e:
                log(f"FOUT in trading cyclus: {e}")


if __name__ == "__main__":
    main()
