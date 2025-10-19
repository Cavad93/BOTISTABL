import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import math
import time
import os
import json

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, api_key: str, api_secret: str, cfg: Optional[object] = None):
        """Инициализация менеджера данных Binance"""
        self.client = Client(api_key, api_secret)
        self.cfg = cfg
        # PAPER-режим, если явно не LIVE
        self._paper_mode = (getattr(self.cfg, 'TRADING_MODE', 'PAPER').upper() != 'LIVE') if self.cfg else True
        self._paper_state_path = getattr(self.cfg, 'PAPER_STATE_PATH', os.path.join('state', 'paper_state.json'))
        self._paper_balances: Dict[str, float] = {}
        if self._paper_mode:
            self._load_paper_state()
        self.symbol_info = {}
        self.update_symbol_info()
        # кэш последней цены (для вебсокетов)
        self.last_price: Dict[str, float] = {}
        # кэш детекции стейблкоинов по символам
        self._stable_cache: Dict[str, bool] = {}


    # ---------- Информация по символам ----------
    def _get_base_asset(self, symbol: str) -> Optional[str]:
        info = self.symbol_info.get(symbol)
        if info:
            return info.get('baseAsset')
        if symbol.endswith('USDT'):
            return symbol[:-4]
        return None

    def is_stablecoin_pair(self, symbol: str, priceChangePercent: Optional[float] = None) -> bool:
        try:
            if symbol in self._stable_cache:
                return self._stable_cache[symbol]

            base = (self._get_base_asset(symbol) or '').upper()
            stable_bases = set([b.strip().upper() for b in getattr(self.cfg, 'STABLE_BASES', [])])

            # Быстрая фильтрация по имени базового актива
            if base and base in stable_bases:
                self._stable_cache[symbol] = True
                return True

            # Доп. фильтр по 24h %-изменению (малое изменение — кандидат на стейбл)
            pct_small = True
            if priceChangePercent is not None:
                try:
                    pct_small = abs(float(priceChangePercent)) <= float(getattr(self.cfg, 'STABLE_24H_PCT_MAX', 3.0))
                except Exception:
                    pct_small = True

            # Поведенческая проверка: средняя цена ~ $1 и низкая вариативность
            looks_stable = False
            if pct_small:
                looks_stable = self._is_stable_by_behavior(symbol)

            self._stable_cache[symbol] = looks_stable
            return looks_stable
        except Exception as e:
            logger.warning(f"Ошибка определения стейблкоина для {symbol}: {e}")
            return False

    def _is_stable_by_behavior(self, symbol: str) -> bool:
        lookback_days = int(getattr(self.cfg, 'STABLE_LOOKBACK_DAYS', 7))
        peg_low = float(getattr(self.cfg, 'STABLE_PEG_LOW', 0.9))
        peg_high = float(getattr(self.cfg, 'STABLE_PEG_HIGH', 1.1))
        max_cv = float(getattr(self.cfg, 'STABLE_MAX_CV', 0.02))
        try:
            # 1h свечи за lookback_days, без агрессивного кеширования — вызов редкий
            df = self.fetch_klines_full(symbol, '1h', min_days=lookback_days)
            if df.empty or 'close' not in df.columns:
                return False
            closes = df['close'].astype(float)
            median = float(closes.median())
            if median <= 0:
                return False
            cv = float(closes.std(ddof=0) / median)
            in_band = (peg_low <= median <= peg_high)
            return in_band and (cv <= max_cv)
        except Exception as e:
            logger.warning(f"Ошибка поведенческой проверки стабильности {symbol}: {e}")
            return False



    def update_symbol_info(self):
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol in exchange_info['symbols']:
                if symbol['status'] == 'TRADING':
                    self.symbol_info[symbol['symbol']] = symbol
        except Exception as e:
            logger.error(f"Ошибка обновления информации о символах: {e}")

    def get_top_coins(self, count: int = 200, min_volume: float = 1_000_000) -> List[str]:
            try:
                tickers = self.client.get_ticker()
                usdt_pairs = []
                for ticker in tickers:
                    symbol = ticker['symbol']
                    if symbol.endswith('USDT') and symbol != 'USDT':
                        volume = float(ticker.get('quoteVolume', 0.0))
                        if volume >= min_volume:
                            pct = float(ticker.get('priceChangePercent', 0.0))
                            if getattr(self.cfg, 'EXCLUDE_STABLES', True):
                                if self.is_stablecoin_pair(symbol, priceChangePercent=pct):
                                    continue
                            usdt_pairs.append({
                                'symbol': symbol,
                                'volume': volume,
                                'priceChangePercent': pct
                            })
                usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
                return [pair['symbol'] for pair in usdt_pairs[:count]]
            except Exception as e:
                logger.error(f"Ошибка получения топовых монет: {e}")
                return []


    # ---------- История свечей ----------
    def _interval_to_ms(self, interval: str) -> int:
        m = {'1m':60, '3m':180, '5m':300, '15m':900, '30m':1800, '1h':3600,
             '2h':7200, '4h':14400, '1d':86400}
        return m.get(interval, 900) * 1000

    def fetch_klines_full(self, symbol: str, interval: str, min_days: int = 30) -> pd.DataFrame:
        """Гарантированно собрать историю минимум за min_days дней (разбивкой по 1000 свечей)."""
        try:
            limit_per_call = 1000
            ms_per_candle = self._interval_to_ms(interval)
            needed_candles = math.ceil((min_days * 24 * 3600 * 1000) / ms_per_candle)

            frames = []
            end_time = int(time.time() * 1000)
            remaining = needed_candles

            while remaining > 0:
                limit = min(limit_per_call, remaining)
                start_time = end_time - limit * ms_per_candle
                kl = self.client.get_klines(symbol=symbol, interval=interval,
                                            startTime=start_time, endTime=end_time, limit=limit)
                if not kl:
                    break
                frames.extend(kl)
                end_time = int(kl[0][0]) - 1
                remaining -= len(kl)
                if len(kl) < limit:
                    break

            if not frames:
                return pd.DataFrame()

            frames.sort(key=lambda x: x[0])
            df = pd.DataFrame(frames, columns=[
                'open_time','open','high','low','close','volume','close_time','quote_asset_volume',
                'number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            for col in ['open','high','low','close','volume','quote_asset_volume']:
                df[col] = df[col].astype(float)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Ошибка получения истории для {symbol}: {e}")
            return pd.DataFrame()

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Быстрый метод (оставлен для совместимости)."""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'open_time','open','high','low','close','volume','close_time','quote_asset_volume',
                'number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            for col in ['open','high','low','close','volume','quote_asset_volume']:
                df[col] = df[col].astype(float)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()

    # ---------- Индикаторы ----------
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    # ---------- Цены и стаканы ----------
    def get_current_price(self, symbol: str) -> float:
        try:
            # приоритезируем вебсокетный кэш
            if symbol in self.last_price:
                return float(self.last_price[symbol])
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return 0.0

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Ошибка получения стакана для {symbol}: {e}")
            return {'bids': [], 'asks': []}

    # ---------- Балансы и сделки ----------
    def get_account_balance(self, asset: str = 'USDT') -> float:
        if self._paper_mode:
            # баланс виртуального кошелька
            return float(self._paper_balances.get(asset, 0.0))
        try:
            account = self.client.get_account()
            for bal in account['balances']:
                if bal['asset'] == asset:
                    return float(bal['free'])
            return 0.0
        except Exception as e:
            logger.warning(f"Не удалось получить баланс из API: {e}")
            return 0.0


    def place_market_order(self, symbol: str, side: str, quantity: float, paper: bool = True) -> Dict:
        """Размещение рыночного ордера. В PAPER-режиме логика с виртуальным кошельком + запись в state."""
        try:
            if paper or self._paper_mode:
                price = float(self.get_current_price(symbol))
                base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol  # упрощенно
                side_u = side.upper()

                if side_u == 'BUY':
                    cost = float(quantity) * price
                    usdt_before = self._paper_balances.get('USDT', 0.0)
                    if usdt_before < cost:
                        logger.warning(f"PAPER: недостаточно USDT ({usdt_before:.2f}) для покупки {symbol} на {cost:.2f}")
                    self._paper_balances['USDT'] = usdt_before - cost
                    self._paper_balances[base_asset] = self._paper_balances.get(base_asset, 0.0) + float(quantity)
                else:  # SELL
                    base_before = self._paper_balances.get(base_asset, 0.0)
                    if base_before < float(quantity):
                        logger.warning(f"PAPER: недостаточно {base_asset} ({base_before:.6f}) для продажи {quantity}")
                    self._paper_balances[base_asset] = base_before - float(quantity)
                    self._paper_balances['USDT'] = self._paper_balances.get('USDT', 0.0) + float(quantity) * price

                self._save_paper_state()

                return {
                    'orderId': int(time.time() * 1000),
                    'symbol': symbol,
                    'side': side_u,
                    'status': 'FILLED',
                    'executedQty': float(quantity),
                    'fills': [{'price': price}],
                    'transactTime': int(time.time() * 1000)
                }
            else:
                side_e = SIDE_BUY if side.upper() == 'BUY' else SIDE_SELL
                order = self.client.create_order(
                    symbol=symbol, side=side_e, type=ORDER_TYPE_MARKET, quantity=quantity
                )
                return order
        except Exception as e:
            logger.error(f"Ошибка размещения ордера по {symbol}: {e}")
            return {}
            
    # ===== PAPER state: загрузка/сохранение =====
    def _ensure_state_dir(self):
        try:
            os.makedirs(os.path.dirname(self._paper_state_path), exist_ok=True)
        except Exception:
            pass

    def _load_paper_state(self):
        self._ensure_state_dir()
        try:
            if os.path.isfile(self._paper_state_path):
                with open(self._paper_state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._paper_balances = {k: float(v) for k, v in data.get('balances', {}).items()}
            if not self._paper_balances:
                default_usdt = float(getattr(self.cfg, 'DEFAULT_DEPOSIT_USDT', 1000.0)) if self.cfg else 1000.0
                self._paper_balances = {'USDT': default_usdt}
                self._save_paper_state()
            logger.info(f"PAPER кошелек загружен: {self._paper_balances}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить PAPER state ({self._paper_state_path}): {e}")
            default_usdt = float(getattr(self.cfg, 'DEFAULT_DEPOSIT_USDT', 1000.0)) if self.cfg else 1000.0
            self._paper_balances = {'USDT': default_usdt}

    def _save_paper_state(self):
        try:
            self._ensure_state_dir()
            with open(self._paper_state_path, 'w', encoding='utf-8') as f:
                json.dump({'balances': self._paper_balances}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Не удалось сохранить PAPER state ({self._paper_state_path}): {e}")
