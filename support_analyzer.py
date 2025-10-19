import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class SupportAnalyzer:
    """
    Анализатор уровней поддержки для шорт-торговли.
    Зеркальная реализация ResistanceAnalyzer, но ищет уровни НИЖЕ текущей цены.
    """
    def __init__(self, config):
        self.config = config
        self.levels_cache = {}
    
    def find_support_levels(self, df: pd.DataFrame, symbol: str) -> Dict[str, List[Dict]]:
        """
        Комплексный поиск уровней поддержки (минимумы, зоны, EMA снизу и т.д.)
        
        Args:
            df: DataFrame с OHLCV данными
            symbol: Символ торговой пары
        
        Returns:
            Словарь с различными типами уровней поддержки
        """
        levels = {
            'historical_lows': self._find_historical_lows(df),
            'horizontal_zones': self._find_horizontal_zones(df),
            'ema_levels': self._calculate_ema_levels(df),
            'fibonacci_levels': self._calculate_fibonacci_levels(df),
            'volume_profile_levels': self._find_volume_profile_levels(df),
            'combined': []
        }
        
        levels['combined'] = self._combine_and_rank_levels(levels, df)
        return levels
    
    def _find_historical_lows(self, df: pd.DataFrame) -> List[Dict]:
        """Поиск исторических минимумов (аналог пиков, но для low)"""
        levels: List[Dict] = []

        lows = df['low'].values.astype(float)
        if lows.size == 0:
            return levels

        # инвертируем для поиска минимумов как пиков
        inverted_lows = -lows
        peaks, _ = find_peaks(inverted_lows, distance=10, prominence=max(lows.std() * 0.1, 1e-12))
        if len(peaks) == 0:
            return levels

        low_prices = lows[peaks]
        clusters = np.full(shape=len(peaks), fill_value=-1, dtype=int)

        if len(low_prices) > 1 and low_prices.std() > 0:
            eps = max(low_prices.std() * 0.01, 1e-9)
            labels = DBSCAN(eps=eps, min_samples=2).fit_predict(low_prices.reshape(-1, 1))
            clusters = labels

            for cid in set(labels):
                if cid == -1:
                    continue
                cluster_lows = low_prices[labels == cid]
                level_price = float(np.mean(cluster_lows))
                tolerance = level_price * self.config.LEVEL_TOLERANCE
                touches = int(np.sum(np.abs(lows - level_price) <= tolerance))
                if touches >= self.config.MIN_TOUCHES:
                    last_idx = peaks[labels == cid][-1]
                    levels.append({
                        'price': level_price,
                        'touches': touches,
                        'strength': min(touches / 10, 1.0),
                        'type': 'historical_low',
                        'last_touch': df.index[last_idx]
                    })

        # одиночные минимумы
        for i, peak_idx in enumerate(peaks):
            if clusters[i] == -1:
                level_price = float(lows[peak_idx])
                tolerance = level_price * self.config.LEVEL_TOLERANCE
                touches = int(np.sum(np.abs(lows - level_price) <= tolerance))
                if touches >= self.config.MIN_TOUCHES:
                    levels.append({
                        'price': level_price,
                        'touches': touches,
                        'strength': min(touches / 10, 1.0),
                        'type': 'historical_low',
                        'last_touch': df.index[peak_idx]
                    })

        return sorted(levels, key=lambda x: x['strength'], reverse=True)
    
    def _find_horizontal_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Зоны консолидации как уровни поддержки"""
        zones = []
        
        df['price_range'] = df['high'] - df['low']
        df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        window = 20
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            price_std = window_data['avg_price'].std()
            
            if price_std < df['avg_price'].std() * 0.3:
                zone_high = window_data['high'].max()
                zone_low = window_data['low'].min()
                zone_center = (zone_high + zone_low) / 2
                
                is_duplicate = False
                for existing_zone in zones:
                    if abs(existing_zone['price'] - zone_center) < zone_center * 0.01:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    zones.append({
                        'price': zone_low,  # нижняя граница как поддержка
                        'zone_low': zone_low,
                        'zone_high': zone_high,
                        'strength': 0.7,
                        'type': 'horizontal_zone',
                        'width': (zone_high - zone_low) / zone_center,
                        'last_touch': window_data.index[-1]
                    })
        
        return zones
    
    def _calculate_ema_levels(self, df: pd.DataFrame) -> List[Dict]:
        """EMA как поддержка (когда цена ВЫШЕ EMA)"""
        levels = []
        
        for period in self.config.EMA_PERIODS:
            if len(df) >= period:
                ema = df['close'].ewm(span=period, adjust=False).mean()
                current_ema = ema.iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # EMA как поддержка, если цена выше
                if current_price > current_ema:
                    levels.append({
                        'price': current_ema,
                        'period': period,
                        'strength': 0.5 + (period / 400),
                        'type': f'EMA{period}',
                        'dynamic': True,
                        'last_touch': df.index[-1]
                    })
        
        return levels
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Фибоначчи уровни как поддержка (в восходящем тренде)"""
        levels = []
        
        recent_data = df.tail(100)
        
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()
        
        # в восходящем тренде ищем поддержки
        if high_idx > low_idx:
            swing_high = recent_data.loc[high_idx, 'high']
            swing_low = recent_data.loc[low_idx, 'low']
            diff = swing_high - swing_low
            
            for fib_level in self.config.FIBONACCI_LEVELS:
                fib_price = swing_low + (diff * fib_level)
                
                # фибоначчи как поддержка в восходящей коррекции
                if df['close'].iloc[-1] > fib_price:
                    levels.append({
                        'price': fib_price,
                        'fib_level': fib_level,
                        'strength': 0.6,
                        'type': f'Fib_{int(fib_level*100)}',
                        'swing_high': swing_high,
                        'swing_low': swing_low,
                        'last_touch': df.index[-1]
                    })
        
        return levels
    
    def _find_volume_profile_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Объёмные уровни как поддержка"""
        levels = []
        
        price_bins = 50
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / price_bins
        
        volume_profile = {}
        
        for _, row in df.iterrows():
            candle_low = row['low']
            candle_high = row['high']
            candle_volume = row['volume']
            
            start_bin = int((candle_low - df['low'].min()) / bin_size)
            end_bin = int((candle_high - df['low'].min()) / bin_size)
            
            for bin_idx in range(max(0, start_bin), min(price_bins, end_bin + 1)):
                bin_price = df['low'].min() + (bin_idx + 0.5) * bin_size
                
                if bin_price not in volume_profile:
                    volume_profile[bin_price] = 0
                
                volume_profile[bin_price] += candle_volume / max(1, (end_bin - start_bin + 1))
        
        if volume_profile:
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            max_volume = sorted_levels[0][1]
            
            for price, volume in sorted_levels[:10]:
                if volume > max_volume * 0.3:
                    levels.append({
                        'price': price,
                        'volume': volume,
                        'strength': min(volume / max_volume, 1.0),
                        'type': 'volume_profile',
                        'last_touch': df.index[-1]
                    })
        
        return levels
    
    def _combine_and_rank_levels(self, all_levels: Dict, df: pd.DataFrame) -> List[Dict]:
        """Объединение и ранжирование уровней поддержки"""
        combined = []
        current_price = df['close'].iloc[-1]
        
        for level_type, levels_list in all_levels.items():
            if level_type != 'combined':
                for level in levels_list:
                    level['distance_percent'] = ((current_price - level['price']) / current_price) * 100
                    
                    # фильтруем только уровни НИЖЕ текущей цены (поддержка)
                    if level['price'] < current_price:
                        combined.append(level)
        
        grouped_levels = []
        used_indices = set()
        
        for i, level in enumerate(combined):
            if i in used_indices:
                continue
            
            group = [level]
            for j, other_level in enumerate(combined[i+1:], i+1):
                if j not in used_indices:
                    if abs(level['price'] - other_level['price']) <= level['price'] * self.config.LEVEL_TOLERANCE:
                        group.append(other_level)
                        used_indices.add(j)
            
            if group:
                avg_price = np.mean([l['price'] for l in group])
                max_strength = max([l['strength'] for l in group])
                types = list(set([l['type'] for l in group]))
                
                grouped_levels.append({
                    'price': avg_price,
                    'strength': min(max_strength * (1 + len(group) * 0.1), 1.0),
                    'types': types,
                    'confirmations': len(group),
                    'distance_percent': ((current_price - avg_price) / current_price) * 100,
                    'last_touch': max([l['last_touch'] for l in group])
                })
        
        grouped_levels.sort(key=lambda x: (x['strength'] * 0.7 - abs(x['distance_percent']) * 0.3), reverse=True)
        
        return grouped_levels[:20]