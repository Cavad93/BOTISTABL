import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from ml_arf import extract_numeric_features

logger = logging.getLogger(__name__)

class BreakoutDetector:
    def __init__(self, config, data_manager, notifier=None, arf_model=None):
        """Инициализация детектора пробоев"""
        self.config = config
        self.data_manager = data_manager
        self.notifier = notifier
        self.arf_model = arf_model
        self.breakout_candidates = {}
        self.confirmed_breakouts = []

    def _rsi(self, closes: pd.Series, period: int = 14) -> float:
        if len(closes) < period + 1:
            return float("nan")
        delta = closes.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / (roll_down.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    
    def check_breakout(self, symbol: str, levels: List[Dict], df: pd.DataFrame) -> Optional[Dict]:
        """
        Проверка условий пробоя
        
        Args:
            symbol: Символ торговой пары
            levels: Список уровней сопротивления
            df: DataFrame с последними данными
        
        Returns:
            Информация о пробое или None
        """
        if df.empty or not levels:
            return None
        
        current_candle = df.iloc[-1]
        prev_candles = df.iloc[-self.config.BREAKOUT_CONFIRMATION_CANDLES-1:-1]
        
        for level in levels:
            # Проверяем только ближайшие уровни (в пределах 5% от цены)
            if level['distance_percent'] > 5 or level['distance_percent'] < 0:
                continue
            
            breakout_info = self._analyze_single_level_breakout(
                symbol, level, current_candle, prev_candles, df
            )
            
            if breakout_info:
                return breakout_info
        
        return None
    
    def _analyze_single_level_breakout(self, symbol: str, level: Dict, 
                                      current_candle: pd.Series, 
                                      prev_candles: pd.DataFrame,
                                      df: pd.DataFrame) -> Optional[Dict]:
        """
        Анализ пробоя конкретного уровня
        
        Args:
            symbol: Символ
            level: Информация об уровне
            current_candle: Текущая свеча
            prev_candles: Предыдущие свечи
            df: Полные данные
        
        Returns:
            Информация о пробое или None
        """
        level_price = level['price']
        
        # Условие 1: Закрытие свечи выше уровня
        breakout_threshold = level_price * (1 + self.config.BREAKOUT_PERCENT)
        if current_candle['close'] <= breakout_threshold:
            return None  # единственный «жёсткий» стоп — цена не закрылась над уровнем
        
        # Условие 2/3/4 — больше НЕ делаем ранних отказов:
        # просто посчитаем их далее в блоке conditions и учтём в confidence_score.
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = current_candle['volume']
        confirmations = sum(1 for _, candle in prev_candles.iterrows() if candle['close'] > level_price)
        rsi = self.data_manager.calculate_rsi(df['close'])
        current_rsi = rsi.iloc[-1]
        # self._add_to_candidates(symbol, level, current_candle)  # по желанию: можно оставить для трекинга, но без return

        
        # Все условия выполнены - пробой подтвержден
# СТАЛО (добавлен расчёт условий, ARF p и Telegram-уведомление)
        # условия пробоя
        level_price = float(level["price"])
        close_price = float(current_candle["close"])

        # 1) закрытие выше уровня + процентный буфер
        thr_pct = float(self.config.BREAKOUT_PERCENT)
        delta_pct = (close_price - level_price) / level_price
        cond_close = {
            "passed": delta_pct > thr_pct,
            "close": close_price,
            "delta_pct": delta_pct,
            "threshold_pct": thr_pct,
        }

        # 2) всплеск объёма
        vol_window = 20
        current_volume = float(current_candle.get("volume", current_candle.get("quote_volume", 0.0)))
        avg_volume = float(pd.to_numeric(df["volume"], errors="coerce").tail(vol_window).mean())
        vol_mult = float(self.config.VOLUME_SURGE_MULTIPLIER)
        cond_vol = {
            "passed": (avg_volume > 0) and (current_volume / avg_volume >= vol_mult),
            "current": current_volume,
            "average": avg_volume if avg_volume > 0 else 0.0,
            "multiplier": vol_mult,
        }

        # 3) RSI
        rsi_val = self._rsi(pd.to_numeric(df["close"], errors="coerce"))
        rsi_thr = int(self.config.RSI_THRESHOLD)
        cond_rsi = {
            "passed": (rsi_val >= rsi_thr) if np.isfinite(rsi_val) else False,
            "value": rsi_val if np.isfinite(rsi_val) else float("nan"),
            "threshold": rsi_thr,
        }

        # 4) подтверждающие свечи
        need_candles = int(self.config.BREAKOUT_CONFIRMATION_CANDLES)
        prev = df.tail(1 + need_candles).iloc[:-1]  # последние N до текущей
        count_ok = int((pd.to_numeric(prev["close"], errors="coerce") > level_price).sum())
        cond_conf = {
            "passed": count_ok >= need_candles,
            "count_ok": count_ok,
            "required": need_candles,
        }

        # Все условия выполнены — пробой подтверждён
        breakout_info = {
            "symbol": symbol,
            "timeframe": level.get("timeframe") or level.get("tf") or "",
            "level_price": level_price,
            "breakout_price": close_price,
            "level_strength": level.get("strength", 0),
            "level_types": level.get("types", [level.get("type", "unknown")]),
            "timestamp": current_candle.name if hasattr(current_candle, "name") else datetime.now(),
            "conditions": {
                "close_above_level": cond_close,
                "volume_surge": cond_vol,
                "rsi": cond_rsi,
                "confirmation_candles": cond_conf,
            },
        }

        # ARF вероятность (если модель передана)
        # ARF вероятность (если модель передана)
        arf_p = None
        if self.arf_model is not None:
            try:
                # собираем те же числовые фичи, что и в основном пайплайне
                feats = {}
                try:
                    feats["score"] = float(level.get("score", 0.0))
                except Exception:
                    feats["score"] = 0.0

                feats.update(extract_numeric_features(breakout_info, "brk_"))
                feats.update(extract_numeric_features(level, "lvl_"))

                # очистка NaN/inf — как в main.py
                for k, v in list(feats.items()):
                    if v != v or v in (float("inf"), float("-inf")):
                        feats[k] = 0.0

                arf_p = float(self.arf_model.predict_proba(feats))
            except Exception as e:
                logger.warning(f"{symbol}: ошибка расчёта ARF proba: {e}")
        breakout_info["arf_proba"] = arf_p


        # отправляем Telegram-уведомление с чек-листом
        if self.notifier is not None:
            try:
                self.notifier.notify_breakout_conditions(breakout_info)
            except Exception as e:
                logger.warning(f"{symbol}: не удалось отправить уведомление о пробое: {e}")

    
    def _add_to_candidates(self, symbol: str, level: Dict, candle: pd.Series):
        """Добавление потенциального пробоя в кандидаты"""
        key = f"{symbol}_{level['price']:.2f}"
        
        if key not in self.breakout_candidates:
            self.breakout_candidates[key] = {
                'symbol': symbol,
                'level': level,
                'first_touch': candle.name if hasattr(candle, 'name') else datetime.now(),
                'touches': 1
            }
        else:
            self.breakout_candidates[key]['touches'] += 1
    
    def _calculate_confidence_score(self, level: Dict, volume_ratio: float, rsi: float) -> float:
        """
        Расчет уверенности в пробое
        
        Args:
            level: Информация об уровне
            volume_ratio: Отношение текущего объема к среднему
            rsi: Значение RSI
        
        Returns:
            Оценка от 0 до 1
        """
        # Компоненты оценки
        level_strength_score = level['strength']
        
        # Нормализация объема (1.5x = 0.5, 3x = 1.0) с нижней отсечкой 0
        volume_score = max(min((volume_ratio - 1) / 2, 1.0), 0.0)

        # Нормализация RSI (60 = 0.5, 80 = 1.0) с нижней отсечкой 0
        rsi_score = max(min((rsi - 60) / 20, 1.0), 0.0)

        
        # Вес подтверждений
        confirmations_score = min(level.get('confirmations', 1) / 3, 1.0)
        
        # Взвешенная оценка
        confidence = (
            level_strength_score * 0.3 +
            volume_score * 0.3 +
            rsi_score * 0.2 +
            confirmations_score * 0.2
        )
        
        return min(max(confidence, 0), 1)
    
    def validate_breakout_quality(self, breakout_info: Dict) -> Tuple[bool, str]:
        """
        Дополнительная валидация качества пробоя
        
        Args:
            breakout_info: Информация о пробое
        
        Returns:
            (Валиден ли пробой, Причина отклонения)
        """
        symbol = breakout_info['symbol']
        
        # Проверка ликвидности через стакан заявок
        order_book = self.data_manager.get_order_book(symbol, limit=20)
        
        if order_book['bids'] and order_book['asks']:
            # Проверка спреда
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            spread_percent = ((best_ask - best_bid) / best_bid) * 100
            
            if spread_percent > 0.5:  # Спред больше 0.5%
                return False, f"Слишком большой спред: {spread_percent:.2f}%"
            
            # Проверка глубины стакана
            bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:5])
            ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:5])
            
            if bid_volume < ask_volume * 0.5:  # Мало покупателей
                return False, "Недостаточная поддержка покупателей"
        
        # Проверка на ложный пробой
        if breakout_info['confidence_score'] < 0.5:
            return False, f"Низкая уверенность: {breakout_info['confidence_score']:.2f}"
        
        return True, "OK"
    
    def get_active_breakouts(self) -> List[Dict]:
        """Получение списка активных пробоев"""
        # Очистка старых кандидатов (старше 1 часа)
        current_time = datetime.now()
        self.breakout_candidates = {
            k: v for k, v in self.breakout_candidates.items()
            if current_time - v['first_touch'] < timedelta(hours=1)
        }
        
        return self.confirmed_breakouts
    
    def cleanup_old_breakouts(self):
        """Очистка старых подтвержденных пробоев"""
        current_time = datetime.now()
        self.confirmed_breakouts = [
            b for b in self.confirmed_breakouts
            if current_time - b['timestamp'] < timedelta(hours=24)
        ]
