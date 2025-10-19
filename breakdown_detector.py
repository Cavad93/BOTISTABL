import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from ml_arf import extract_numeric_features

logger = logging.getLogger(__name__)

class BreakdownDetector:
    """
    Детектор пробоя уровней поддержки ВНИЗ (для шорт-позиций).
    Зеркальная реализация BreakoutDetector.
    """
    def __init__(self, config, data_manager, notifier=None, arf_model_short=None):
        self.config = config
        self.data_manager = data_manager
        self.notifier = notifier
        self.arf_model_short = arf_model_short
        self.breakdown_candidates = {}
        self.confirmed_breakdowns = []

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
    
    def check_breakdown(self, symbol: str, levels: List[Dict], df: pd.DataFrame) -> Optional[Dict]:
        """
        Проверка условий пробоя поддержки ВНИЗ
        
        Args:
            symbol: Символ торговой пары
            levels: Список уровней поддержки
            df: DataFrame с последними данными
        
        Returns:
            Информация о пробое вниз или None
        """
        if df.empty or not levels:
            return None
        
        current_candle = df.iloc[-1]
        prev_candles = df.iloc[-self.config.BREAKOUT_CONFIRMATION_CANDLES-1:-1]
        
        for level in levels:
            # проверяем только ближайшие уровни (в пределах 5% НИЖЕ цены)
            if level['distance_percent'] > 5 or level['distance_percent'] < 0:
                continue
            
            breakdown_info = self._analyze_single_level_breakdown(
                symbol, level, current_candle, prev_candles, df
            )
            
            if breakdown_info:
                return breakdown_info
        
        return None
    
    def _analyze_single_level_breakdown(self, symbol: str, level: Dict, 
                                       current_candle: pd.Series, 
                                       prev_candles: pd.DataFrame,
                                       df: pd.DataFrame) -> Optional[Dict]:
        """Анализ пробоя конкретного уровня поддержки ВНИЗ"""
        level_price = float(level['price'])
        close_price = float(current_candle['close'])
        
        # Условие 1: Закрытие свечи НИЖЕ уровня
        breakdown_threshold = level_price * (1 - self.config.BREAKOUT_PERCENT)
        if close_price >= breakdown_threshold:
            return None  # цена не закрылась ниже уровня
        
        # Условие 2: всплеск объёма
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
        
        # Условие 3: RSI (для шорта: RSI должен быть НИЗКИМ, < 45)
        rsi_val = self._rsi(pd.to_numeric(df["close"], errors="coerce"))
        rsi_thr_short = int(getattr(self.config, 'RSI_THRESHOLD_SHORT', 45))
        cond_rsi = {
            "passed": (rsi_val <= rsi_thr_short) if np.isfinite(rsi_val) else False,
            "value": rsi_val if np.isfinite(rsi_val) else float("nan"),
            "threshold": rsi_thr_short,
        }
        
        # Условие 4: подтверждающие свечи (закрытие НИЖЕ уровня)
        need_candles = int(self.config.BREAKOUT_CONFIRMATION_CANDLES)
        prev = df.tail(1 + need_candles).iloc[:-1]
        count_ok = int((pd.to_numeric(prev["close"], errors="coerce") < level_price).sum())
        cond_conf = {
            "passed": count_ok >= need_candles,
            "count_ok": count_ok,
            "required": need_candles,
        }
        
        # Условие 5: дельта закрытия (цена ушла НИЖЕ уровня с буфером)
        thr_pct = float(self.config.BREAKOUT_PERCENT)
        delta_pct = (level_price - close_price) / level_price  # положительная, если ниже
        cond_close = {
            "passed": delta_pct > thr_pct,
            "close": close_price,
            "delta_pct": delta_pct,
            "threshold_pct": thr_pct,
        }
        
        breakdown_info = {
            "symbol": symbol,
            "timeframe": level.get("timeframe") or level.get("tf") or "",
            "level_price": level_price,
            "breakdown_price": close_price,
            "level_strength": level.get("strength", 0),
            "level_types": level.get("types", [level.get("type", "unknown")]),
            "timestamp": current_candle.name if hasattr(current_candle, "name") else datetime.now(),
            "direction": "SHORT",
            "conditions": {
                "close_below_level": cond_close,
                "volume_surge": cond_vol,
                "rsi": cond_rsi,
                "confirmation_candles": cond_conf,
            },
        }
        
        # ARF вероятность для ШОРТА (отдельная модель)
        arf_p = None
        if self.arf_model_short is not None:
            try:
                feats = {}
                try:
                    feats["score"] = float(level.get("score", 0.0))
                except Exception:
                    feats["score"] = 0.0

                feats.update(extract_numeric_features(breakdown_info, "brkd_"))
                feats.update(extract_numeric_features(level, "lvl_"))

                for k, v in list(feats.items()):
                    if v != v or v in (float("inf"), float("-inf")):
                        feats[k] = 0.0

                arf_p = float(self.arf_model_short.predict_proba(feats))
            except Exception as e:
                logger.warning(f"{symbol}: ошибка расчёта ARF SHORT proba: {e}")
        breakdown_info["arf_proba"] = arf_p
        
        # уведомление в Telegram
        if self.notifier is not None:
            try:
                self.notifier.notify_breakdown_conditions(breakdown_info)
            except Exception as e:
                logger.warning(f"{symbol}: не удалось отправить уведомление о пробое вниз: {e}")
        
        return breakdown_info
    
    def get_active_breakdowns(self) -> List[Dict]:
        """Получение списка активных пробоев вниз"""
        current_time = datetime.now()
        from datetime import timedelta
        self.breakdown_candidates = {
            k: v for k, v in self.breakdown_candidates.items()
            if current_time - v.get('first_touch', current_time) < timedelta(hours=1)
        }
        
        return self.confirmed_breakdowns