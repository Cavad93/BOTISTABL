import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BreakoutDetector:
    def __init__(self, config, data_manager):
        """Инициализация детектора пробоев"""
        self.config = config
        self.data_manager = data_manager
        self.breakout_candidates = {}
        self.confirmed_breakouts = []
    
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
            return None
        
        # Условие 2: Проверка объема
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = current_candle['volume']
        
        if current_volume < avg_volume * self.config.VOLUME_SURGE_MULTIPLIER:
            logger.debug(f"{symbol}: Недостаточный объем для пробоя")
            return None
        
        # Условие 3: Подтверждение пробоя (X свечей закрылись выше уровня)
        confirmations = 0
        for _, candle in prev_candles.iterrows():
            if candle['close'] > level_price:
                confirmations += 1
        
        if confirmations < self.config.BREAKOUT_CONFIRMATION_CANDLES - 1:
            # Добавляем в кандидаты для отслеживания
            self._add_to_candidates(symbol, level, current_candle)
            return None
        
        # Условие 4: RSI > 60
        rsi = self.data_manager.calculate_rsi(df['close'])
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < self.config.RSI_THRESHOLD:
            logger.debug(f"{symbol}: RSI слишком низкий: {current_rsi:.2f}")
            return None
        
        # Все условия выполнены - пробой подтвержден
        breakout_info = {
            'symbol': symbol,
            'level_price': level_price,
            'breakout_price': current_candle['close'],
            'volume_surge': current_volume / avg_volume,
            'rsi': current_rsi,
            'level_strength': level['strength'],
            'level_types': level.get('types', [level.get('type', 'unknown')]),
            'timestamp': current_candle.name if hasattr(current_candle, 'name') else datetime.now(),
            'entry_price': self.data_manager.get_current_price(symbol),
            'stop_loss': level_price * (1 - self.config.STOP_LOSS_PERCENT),
            'take_profit': current_candle['close'] * (1 + self.config.TAKE_PROFIT_PERCENT),
            'confidence_score': self._calculate_confidence_score(level, current_volume / avg_volume, current_rsi)
        }
        
        return breakout_info
    
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
        
        # Нормализация объема (1.5x = 0.5, 3x = 1.0)
        volume_score = min((volume_ratio - 1) / 2, 1.0)
        
        # Нормализация RSI (60 = 0.5, 80 = 1.0)
        rsi_score = min((rsi - 60) / 20, 1.0)
        
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
