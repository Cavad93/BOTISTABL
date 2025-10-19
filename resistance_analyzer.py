import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class ResistanceAnalyzer:
    def __init__(self, config):
        """Инициализация анализатора уровней сопротивления"""
        self.config = config
        self.levels_cache = {}
    
    def find_resistance_levels(self, df: pd.DataFrame, symbol: str) -> Dict[str, List[Dict]]:
        """
        Комплексный поиск уровней сопротивления
        
        Args:
            df: DataFrame с OHLCV данными
            symbol: Символ торговой пары
        
        Returns:
            Словарь с различными типами уровней
        """
        levels = {
            'historical_peaks': self._find_historical_peaks(df),
            'horizontal_zones': self._find_horizontal_zones(df),
            'ema_levels': self._calculate_ema_levels(df),
            'fibonacci_levels': self._calculate_fibonacci_levels(df),
            'volume_profile_levels': self._find_volume_profile_levels(df),
            'combined': []
        }
        
        # Объединение и ранжирование всех уровней
        levels['combined'] = self._combine_and_rank_levels(levels, df)
        
        return levels
    
    def _find_historical_peaks(self, df: pd.DataFrame) -> List[Dict]:
        levels: List[Dict] = []

        highs = df['high'].values.astype(float)
        if highs.size == 0:
            return levels

        # базовый поиск локальных максимумов
        peaks, _ = find_peaks(highs, distance=10, prominence=max(highs.std() * 0.1, 1e-12))
        if len(peaks) == 0:
            return levels

        peak_prices = highs[peaks]

        # по умолчанию считаем все пики "шумом" (кластер = -1), чтобы не падать при < 2 пика
        clusters = np.full(shape=len(peaks), fill_value=-1, dtype=int)

        # кластеризуем, только если пиков >= 2 и есть разброс
        if len(peak_prices) > 1 and peak_prices.std() > 0:
            eps = max(peak_prices.std() * 0.01, 1e-9)  # защитимся от eps=0
            labels = DBSCAN(eps=eps, min_samples=2).fit_predict(peak_prices.reshape(-1, 1))
            clusters = labels

            # уровни из кластеров (подтверждённые множественными касаниями)
            for cid in set(labels):
                if cid == -1:
                    continue
                cluster_peaks = peak_prices[labels == cid]
                level_price = float(np.mean(cluster_peaks))
                tolerance = level_price * self.config.LEVEL_TOLERANCE
                touches = int(np.sum(np.abs(highs - level_price) <= tolerance))
                if touches >= self.config.MIN_TOUCHES:
                    last_idx = peaks[labels == cid][-1]
                    levels.append({
                        'price': level_price,
                        'touches': touches,
                        'strength': min(touches / 10, 1.0),
                        'type': 'historical_peak',
                        'last_touch': df.index[last_idx]
                    })

        # одиночные/шумовые пики — как отдельные уровни
        for i, peak_idx in enumerate(peaks):
            if clusters[i] == -1:
                level_price = float(highs[peak_idx])
                tolerance = level_price * self.config.LEVEL_TOLERANCE
                touches = int(np.sum(np.abs(highs - level_price) <= tolerance))
                if touches >= self.config.MIN_TOUCHES:
                    levels.append({
                        'price': level_price,
                        'touches': touches,
                        'strength': min(touches / 10, 1.0),
                        'type': 'historical_peak',
                        'last_touch': df.index[peak_idx]
                    })

        return sorted(levels, key=lambda x: x['strength'], reverse=True)

    
    def _find_horizontal_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Поиск горизонтальных зон консолидации
        
        Args:
            df: DataFrame с данными
        
        Returns:
            Список зон консолидации
        """
        zones = []
        
        # Анализ периодов низкой волатильности
        df['price_range'] = df['high'] - df['low']
        df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Скользящее окно для поиска зон консолидации
        window = 20
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            price_std = window_data['avg_price'].std()
            
            # Если стандартное отклонение мало, это зона консолидации
            if price_std < df['avg_price'].std() * 0.3:
                zone_high = window_data['high'].max()
                zone_low = window_data['low'].min()
                zone_center = (zone_high + zone_low) / 2
                
                # Проверка, не добавлена ли уже похожая зона
                is_duplicate = False
                for existing_zone in zones:
                    if abs(existing_zone['price'] - zone_center) < zone_center * 0.01:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    zones.append({
                        'price': zone_high,  # Используем верхнюю границу как уровень сопротивления
                        'zone_low': zone_low,
                        'zone_high': zone_high,
                        'strength': 0.7,
                        'type': 'horizontal_zone',
                        'width': (zone_high - zone_low) / zone_center,
                        'last_touch': window_data.index[-1]
                    })
        
        return zones
    
    def _calculate_ema_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Расчет динамических уровней на основе EMA
        
        Args:
            df: DataFrame с данными
        
        Returns:
            Список EMA уровней
        """
        levels = []
        
        for period in self.config.EMA_PERIODS:
            if len(df) >= period:
                ema = df['close'].ewm(span=period, adjust=False).mean()
                current_ema = ema.iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # EMA как сопротивление, если цена ниже
                if current_price < current_ema:
                    levels.append({
                        'price': current_ema,
                        'period': period,
                        'strength': 0.5 + (period / 400),  # Чем больше период, тем сильнее уровень
                        'type': f'EMA{period}',
                        'dynamic': True,
                        'last_touch': df.index[-1]
                    })
        
        return levels
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Расчет уровней Фибоначчи
        
        Args:
            df: DataFrame с данными
        
        Returns:
            Список уровней Фибоначчи
        """
        levels = []
        
        # Находим последние значимые максимум и минимум
        recent_data = df.tail(100)  # Последние 100 свечей
        
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()
        
        # Определяем направление тренда
        if high_idx > low_idx:  # Восходящий тренд
            swing_high = recent_data.loc[high_idx, 'high']
            swing_low = recent_data.loc[low_idx, 'low']
            diff = swing_high - swing_low
            
            for fib_level in self.config.FIBONACCI_LEVELS:
                fib_price = swing_high - (diff * fib_level)
                
                # Фибоначчи как сопротивление в нисходящей коррекции
                if df['close'].iloc[-1] < fib_price:
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
        """
        Поиск уровней на основе профиля объема
        
        Args:
            df: DataFrame с данными
        
        Returns:
            Список уровней с высоким объемом
        """
        levels = []
        
        # Создаем профиль объема
        price_bins = 50
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / price_bins
        
        volume_profile = {}
        
        for _, row in df.iterrows():
            # Распределяем объем по ценовым уровням внутри свечи
            candle_low = row['low']
            candle_high = row['high']
            candle_volume = row['volume']
            
            # Находим бины, которые пересекаются со свечой
            start_bin = int((candle_low - df['low'].min()) / bin_size)
            end_bin = int((candle_high - df['low'].min()) / bin_size)
            
            for bin_idx in range(max(0, start_bin), min(price_bins, end_bin + 1)):
                bin_price = df['low'].min() + (bin_idx + 0.5) * bin_size
                
                if bin_price not in volume_profile:
                    volume_profile[bin_price] = 0
                
                # Распределяем объем равномерно
                volume_profile[bin_price] += candle_volume / max(1, (end_bin - start_bin + 1))
        
        # Находим уровни с высоким объемом
        if volume_profile:
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            max_volume = sorted_levels[0][1]
            
            # Берем топ уровни по объему
            for price, volume in sorted_levels[:10]:
                if volume > max_volume * 0.3:  # Минимум 30% от максимального объема
                    levels.append({
                        'price': price,
                        'volume': volume,
                        'strength': min(volume / max_volume, 1.0),
                        'type': 'volume_profile',
                        'last_touch': df.index[-1]
                    })
        
        return levels
    
    def _combine_and_rank_levels(self, all_levels: Dict, df: pd.DataFrame) -> List[Dict]:
        """
        Объединение и ранжирование всех найденных уровней
        
        Args:
            all_levels: Словарь со всеми типами уровней
            df: DataFrame с данными
        
        Returns:
            Отсортированный список комбинированных уровней
        """
        combined = []
        current_price = df['close'].iloc[-1]
        
        # Собираем все уровни в один список
        for level_type, levels_list in all_levels.items():
            if level_type != 'combined':
                for level in levels_list:
                    # Добавляем расстояние до текущей цены
                    level['distance_percent'] = ((level['price'] - current_price) / current_price) * 100
                    
                    # Фильтруем только уровни выше текущей цены (сопротивление)
                    if level['price'] > current_price:
                        combined.append(level)
        
        # Группируем близкие уровни
        grouped_levels = []
        used_indices = set()
        
        for i, level in enumerate(combined):
            if i in used_indices:
                continue
            
            # Находим все уровни в пределах tolerance
            group = [level]
            for j, other_level in enumerate(combined[i+1:], i+1):
                if j not in used_indices:
                    if abs(level['price'] - other_level['price']) <= level['price'] * self.config.LEVEL_TOLERANCE:
                        group.append(other_level)
                        used_indices.add(j)
            
            # Создаем объединенный уровень
            if group:
                avg_price = np.mean([l['price'] for l in group])
                max_strength = max([l['strength'] for l in group])
                types = list(set([l['type'] for l in group]))
                
                grouped_levels.append({
                    'price': avg_price,
                    'strength': min(max_strength * (1 + len(group) * 0.1), 1.0),  # Усиление за счет подтверждения
                    'types': types,
                    'confirmations': len(group),
                    'distance_percent': ((avg_price - current_price) / current_price) * 100,
                    'last_touch': max([l['last_touch'] for l in group])
                })
        
        # Сортировка по силе и близости к цене
        grouped_levels.sort(key=lambda x: (x['strength'] * 0.7 - abs(x['distance_percent']) * 0.3), reverse=True)
        
        return grouped_levels[:20]  # Возвращаем топ-20 уровней
