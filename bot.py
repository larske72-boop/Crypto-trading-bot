#!/usr/bin/env python3
"""
Crypto trading bot -- multi-asset scanner met EMA-momentum + RSI + ATR risicobeheer.
Scant elke run alle geconfigureerde symbolen en koopt de sterkste trend.
Paper-trading standaard; live trading vereist expliciete configuratie.
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
    strength: float = 0.0   # EMA-spread % — hoger = sterkere trend


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
        curr = df.iloc[-1]
        # State-based: enter whenever EMA fast is above EMA slow AND RSI confirms momentum.
        if curr["ema_fast"] > curr["ema_slow"] and curr["rsi"] > self.rsi_buy_threshold:
            strength = (curr["ema_fast"] - curr["ema_slow"]) / curr["ema_slow"] * 100
            return Signal(
                "buy",
                f"EMA{self.ema_fast} > EMA{self.ema_slow}, RSI={curr['rsi']:.1f} > {self.rsi_buy_threshold}",
                curr["close"], curr["atr"], strength,
            )
        return None

    def check_exit_signal(self, df: pd.DataFrame, stop_loss: float, take_profit: float) -> Optional[Signal]:
        curr, prev = df.iloc[-1], df.iloc[-2]
        price = curr["close"]
        if price <= stop_loss:
            return Signal("sell", f"Stop-loss geraakt ({price:.4g} <= {stop_loss:.4g})", price, curr["atr"])
        if price >= take_profit:
            return Signal("sell", f"Take-profit geraakt ({price:.4g} >= {take_profit:.4g})", price, curr["atr"])
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
# Paper trading portfolio
# ============================================================

@dataclass
class Position:
    side: str
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: str
    symbol: str = "BTC/USD"


@dataclass
class Trade:
    side: str
    amount: float
    price: float
    timestamp: str
    reason: str
    pnl_quote: Optional[float] = None
    symbol: str = "BTC/USD"


class PaperPortfolio:
    def __init__(self, state_path: str, starting_balance: float, mode: str = "paper"):
        self.state_path = state_path
        self.balance_quote = starting_balance
        self.mode = mode
        self.position: Optional[Position] = None
        self.trade_log = []
        self.last_price: float = 0.0
        self.last_updated: Optional[str] = None
        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                data = json.load(f)
            self.balance_quote = data["balance_quote"]
            if data.get("position"):
                pos = data["position"]
                pos.setdefault("symbol", "BTC/USD")
                self.position = Position(**pos)
            else:
                self.position = None
            self.trade_log = []
            for t in data.get("trade_log", []):
                t.setdefault("symbol", "BTC/USD")
                self.trade_log.append(Trade(**t))
            self.last_price = data.get("last_price", 0.0)
            self.last_updated = data.get("last_updated")
        else:
            self.save()

    def save(self):
        data = {
            "mode": self.mode,
            "balance_quote": self.balance_quote,
            "position": asdict(self.position) if self.position else None,
            "trade_log": [asdict(t) for t in self.trade_log],
            "last_price": self.last_price,
            "last_updated": self.last_updated,
        }
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def update_price(self, price: float):
        self.last_price = price
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()

    def open_long(self, symbol: str, amount: float, price: float, stop_loss: float, take_profit: float, reason: str):
        cost = amount * price
        if cost > self.balance_quote:
            amount = self.balance_quote / price
            cost = amount * price
        self.balance_quote -= cost
        self.position = Position(
            "long", amount, price, stop_loss, take_profit,
            datetime.now(timezone.utc).isoformat(), symbol=symbol,
        )
        self.trade_log.append(Trade(
            "buy", amount, price, datetime.now(timezone.utc).isoformat(), reason, symbol=symbol,
        ))
        self.save()

    def close_long(self, price: float, reason: str):
        if not self.position:
            return
        symbol = self.position.symbol
        proceeds = self.position.amount * price
        pnl = proceeds - self.position.amount * self.position.entry_price
        self.balance_quote += proceeds
        self.trade_log.append(Trade(
            "sell", self.position.amount, price,
            datetime.now(timezone.utc).isoformat(), reason, pnl_quote=pnl, symbol=symbol,
        ))
        self.position = None
        self.save()

    def equity(self, current_price: float) -> float:
        if not self.position:
            return self.balance_quote
        return self.balance_quote + self.position.amount * current_price


# ============================================================
# Exchange-wrapper (ccxt)
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
            api_key = os.environ.get("EXCHANGE_API_KEY")
            api_secret = os.environ.get("EXCHANGE_API_SECRET")
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
                "'YES_I_UNDERSTAND_THE_RISK'."
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
# Multi-asset scanner
# ============================================================

def scan_symbols(symbols: list, exchange: ExchangeClient, strategy: EmaRsiAtrStrategy, timeframe: str):
    """Fetch OHLCV voor elk symbool, bereken indicatoren en retourneer kansen gesorteerd op sterkte."""
    candidates = []
    for sym in symbols:
        try:
            df = exchange.fetch_ohlcv_df(sym, timeframe, limit=200)
            df = strategy.compute_indicators(df)
            sig = strategy.generate_entry_signal(df)
            curr = df.iloc[-1]
            if sig:
                log(f"  + {sym}: RSI={curr['rsi']:.1f}  EMA-spread={sig.strength:.3f}%  prijs={curr['close']:.4g}")
                candidates.append((sym, sig))
            else:
                trend = "omhoog" if curr["ema_fast"] > curr["ema_slow"] else "omlaag"
                log(f"  - {sym}: RSI={curr['rsi']:.1f}  trend={trend}  geen signaal")
        except Exception as e:
            log(f"  ! {sym}: overgeslagen — {e}")
    candidates.sort(key=lambda x: x[1].strength, reverse=True)
    return candidates


# ============================================================
# Hoofdlogica
# ============================================================

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} UTC | {msg}", flush=True)


def run_cycle(config, exchange, strategy, portfolio, risk_params):
    symbols = config.get("symbols") or [config.get("symbol", "BTC/USD")]
    timeframe = config["timeframe"]

    if portfolio.position is None:
        log(f"Scannen {len(symbols)} symbolen ({timeframe})...")
        candidates = scan_symbols(symbols, exchange, strategy, timeframe)

        if candidates:
            sym, signal = candidates[0]
            log(f"Beste kans: {sym}  sterkte={signal.strength:.3f}%")
            stop_loss, take_profit = strategy.compute_stop_and_target(signal.price, signal.atr)
            amount = position_size(portfolio.balance_quote, signal.price, stop_loss, risk_params)
            if amount <= 0:
                log("Signaal genegeerd: positiegrootte 0 (te weinig saldo of ongeldige stop).")
            else:
                log(f"BUY {sym}: {signal.reason} | {amount:.6f} @ {signal.price:.4g} | SL={stop_loss:.4g} TP={take_profit:.4g}")
                if config["mode"] == "live":
                    exchange.create_market_buy_order(sym, amount)
                portfolio.open_long(sym, amount, signal.price, stop_loss, take_profit, signal.reason)
        else:
            log(f"Geen entry-signalen voor alle {len(symbols)} symbolen.")

        try:
            df0 = exchange.fetch_ohlcv_df(symbols[0], timeframe, limit=5)
            portfolio.update_price(df0.iloc[-1]["close"])
        except Exception:
            portfolio.update_price(portfolio.last_price)

    else:
        pos = portfolio.position
        df = exchange.fetch_ohlcv_df(pos.symbol, timeframe, limit=200)
        df = strategy.compute_indicators(df)
        current_price = df.iloc[-1]["close"]
        signal = strategy.check_exit_signal(df, pos.stop_loss, pos.take_profit)
        if signal and signal.action == "sell":
            log(f"SELL {pos.symbol}: {signal.reason} | {pos.amount:.6f} @ {signal.price:.4g}")
            if config["mode"] == "live":
                exchange.create_market_sell_order(pos.symbol, pos.amount)
            portfolio.close_long(signal.price, signal.reason)
        else:
            log(f"Positie {pos.symbol}: entry={pos.entry_price:.4g}  prijs={current_price:.4g}  "
                f"SL={pos.stop_loss:.4g}  TP={pos.take_profit:.4g}  equity={portfolio.equity(current_price):.2f}")
        portfolio.update_price(current_price)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    symbols = config.get("symbols") or [config.get("symbol", "BTC/USD")]
    log(f"Bot gestart | mode={config['mode'].upper()} | {len(symbols)} symbolen | {config['timeframe']}")
    if config["mode"] == "live":
        log("LIVE MODE ACTIEF — echte orders met echt geld zijn mogelijk.")

    exchange = ExchangeClient(config["exchange"], config["mode"], log)
    strategy = EmaRsiAtrStrategy(config["strategy"])
    risk_params = RiskParams(**config["risk"])
    portfolio = PaperPortfolio(config["paper"]["state_file"], config["paper"]["starting_balance"], config["mode"])

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
