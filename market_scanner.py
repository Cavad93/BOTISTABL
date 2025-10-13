import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketScanner:
    def __init__(self, config, data_manager, resistance_analyzer, breakout_detector):
        """Инициализация сканера рынка"""
        self.config = config
        self.data_manager = data_manager
        self.resistance_analyzer = resistance_analyzer
        self.breakout_detector = breakout_detector
        self.market_snapshot = {}
        self.breakout_opportunities = []
    
    def scan_market(self) -> List[Dict]:
        """
        Полное сканирование рынка
        
        Returns:
            Список потенциальных возможностей для входа
        """
        logger.info("Начинаю сканирование рынка...")
        
        # Получаем список топовых монет
        top_coins = self.data_manager.get_top_coins(
            self.config.TOP_COINS_COUNT,
            self.config.MIN_VOLUME_24H
        )
        
        if not top_coins:
            logger.error("Не удалось получить список монет")
            return []
        
        logger.info(f"Анализирую {len(top_coins)} монет...")
        
        # Параллельный анализ монет
        opportunities = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_symbol, symbol): symbol 
                for symbol in top_coins
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        opportunities.append(result)
                except Exception as e:
                    logger.error(f"Ошибка анализа {symbol}: {e}")
        
        # Сортировка по потенциалу
        opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Сохранение результатов
        self.breakout_opportunities = opportunities[:20]  # Топ-20 возможностей
        
        logger.info(f"Найдено {len(opportunities)} потенциальных возможностей")
        
        return self.breakout_opportunities
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Анализ отдельной торговой пары
        
        Args:
            symbol: Символ торговой пары
        
        Returns:
            Информация о возможности или None
        """
        try:
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'timeframes': {}
            }
            
            best_opportunity = None
            max_score = 0
            
            # Анализ на разных таймфреймах
            for timeframe in self.config.TIMEFRAMES:
                # Получение данных
                df = self.data_manager.get_klines(symbol, timeframe, limit=500)
                
                if df.empty:
                    continue
                
                # Поиск уровней сопротивления
                levels = self.resistance_analyzer.find_resistance_levels(df, symbol)
                
                if not levels.get('combined'):
                    continue
                
                # Проверка на пробой
                breakout_info = self.breakout_detector.check_breakout(
                    symbol, 
                    levels['combined'], 
                    df
                )
                
                # Оценка потенциала
                score = self._calculate_opportunity_score(
                    symbol, levels['combined'], df, breakout_info
                )
                
                analysis_result['timeframes'][timeframe] = {
                    'levels_count': len(levels['combined']),
                    'nearest_level': levels['combined'][0] if levels['combined'] else None,
                    'has_breakout': breakout_info is not None,
                    'score': score
                }
                
                if score > max_score:
                    max_score = score
                    best_opportunity = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'levels': levels['combined'][:3],  # Топ-3 уровня
                        'breakout_info': breakout_info,
                        'score': score,
                        'current_price': df['close'].iloc[-1],
                        'volume_24h': df['volume'].tail(96).sum() if timeframe == '15m' else None
                    }
            
            # Добавление в снимок рынка
            self.market_snapshot[symbol] = analysis_result
            
            return best_opportunity
            
        except Exception as e:
            logger.error(f"Ошибка при анализе {symbol}: {e}")
            return None
    
    def _calculate_opportunity_score(self, symbol: str, levels: List[Dict], 
                                    df: pd.DataFrame, 
                                    breakout_info: Optional[Dict]) -> float:
        """
        Расчет оценки потенциала торговой возможности
        
        Args:
            symbol: Символ
            levels: Уровни сопротивления
            df: Данные
            breakout_info: Информация о пробое
        
        Returns:
            Оценка от 0 до 100
        """
        score = 0
        current_price = df['close'].iloc[-1]
        
        # 1. Если есть подтвержденный пробой - максимальный приоритет
        if breakout_info:
            score += 50
            score += breakout_info.get('confidence_score', 0) * 20
            return score
        
        # 2. Близость к уровню сопротивления
        if levels:
            nearest_level = levels[0]
            distance_percent = abs(nearest_level['distance_percent'])
            
            if distance_percent < 1:  # Очень близко к уровню
                score += 30
            elif distance_percent < 2:
                score += 20
            elif distance_percent < 3:
                score += 10
            
            # Сила уровня
            score += nearest_level['strength'] * 15
        
        # 3. Тренд и моментум
        # RSI
        rsi = self.data_manager.calculate_rsi(df['close'])
        current_rsi = rsi.iloc[-1]
        
        if 50 < current_rsi < 70:  # Оптимальная зона
            score += 10
        elif 40 < current_rsi < 50:
            score += 5
        
        # Тренд (цена выше MA50)
        if len(df) >= 50:
            ma50 = df['close'].rolling(50).mean().iloc[-1]
            if current_price > ma50:
                score += 10
        
        # 4. Объем
        recent_volume = df['volume'].tail(10).mean()
        avg_volume = df['volume'].mean()
        
        if recent_volume > avg_volume * 1.2:
            score += 5
        
        # 5. Волатильность (предпочитаем среднюю)
        volatility = df['close'].pct_change().std()
        if 0.01 < volatility < 0.05:  # 1-5% волатильность
            score += 5
        
        return min(score, 100)
    
    def get_breakout_map(self) -> Dict[str, List[Dict]]:
        """
        Построение карты потенциальных пробоев
        
        Returns:
            Словарь с группировкой по таймфреймам
        """
        breakout_map = {tf: [] for tf in self.config.TIMEFRAMES}
        
        for opportunity in self.breakout_opportunities:
            timeframe = opportunity.get('timeframe')
            if timeframe:
                breakout_map[timeframe].append({
                    'symbol': opportunity['symbol'],
                    'score': opportunity['score'],
                    'nearest_level': opportunity['levels'][0] if opportunity.get('levels') else None,
                    'current_price': opportunity.get('current_price'),
                    'has_breakout': opportunity.get('breakout_info') is not None
                })
        
        # Сортировка по оценке в каждом таймфрейме
        for tf in breakout_map:
            breakout_map[tf].sort(key=lambda x: x['score'], reverse=True)
        
        return breakout_map
    
    def get_market_summary(self) -> Dict:
        """
        Получение общей сводки по рынку
        
        Returns:
            Сводная информация
        """
        total_analyzed = len(self.market_snapshot)
        with_levels = sum(
            1 for s in self.market_snapshot.values() 
            if any(tf.get('levels_count', 0) > 0 for tf in s['timeframes'].values())
        )
        with_breakouts = sum(
            1 for s in self.market_snapshot.values()
            if any(tf.get('has_breakout', False) for tf in s['timeframes'].values())
        )
        
        top_opportunities = self.breakout_opportunities[:5] if self.breakout_opportunities else []
        
        return {
            'timestamp': datetime.now(),
            'total_analyzed': total_analyzed,
            'coins_with_levels': with_levels,
            'active_breakouts': with_breakouts,
            'top_opportunities': [
                {
                    'symbol': opp['symbol'],
                    'score': opp['score'],
                    'timeframe': opp.get('timeframe'),
                    'action': 'BUY' if opp.get('breakout_info') else 'WATCH'
                }
                for opp in top_opportunities
            ],
            'market_strength': self._calculate_market_strength()
        }
    
    def _calculate_market_strength(self) -> str:
        """Оценка общей силы рынка"""
        if not self.breakout_opportunities:
            return "WEAK"
        
        avg_score = np.mean([opp['score'] for opp in self.breakout_opportunities])
        
        if avg_score > 70:
            return "VERY_STRONG"
        elif avg_score > 50:
            return "STRONG"
        elif avg_score > 30:
            return "NEUTRAL"
        else:
            return "WEAK"
