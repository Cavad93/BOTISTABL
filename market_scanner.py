import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketScanner:
    def __init__(self, config, data_manager, resistance_analyzer, breakout_detector):
        self.config = config
        self.data_manager = data_manager
        self.resistance_analyzer = resistance_analyzer
        self.breakout_detector = breakout_detector

        self.market_snapshot: Dict[str, Dict] = {}
        self.breakout_opportunities: List[Dict] = []

    def scan_market(self) -> List[Dict]:
        logger.info("Начинаю сканирование рынка...")
        top_coins = self.data_manager.get_top_coins(self.config.TOP_COINS_COUNT, self.config.MIN_VOLUME_24H)
        if not top_coins:
            logger.error("Не удалось получить список монет")
            return []

        logger.info(f"Анализирую {len(top_coins)} монет...")
        opportunities: List[Dict] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_symbol, symbol): symbol
                for symbol in top_coins
            }
            for future in as_completed(future_to_symbol):
                res = future.result()
                if res:
                    opportunities.append(res)

        opportunities.sort(key=lambda x: x['score'], reverse=True)
        self.breakout_opportunities = opportunities
        return opportunities

    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            analysis_result: Dict = {'symbol': symbol, 'timestamp': datetime.now(), 'timeframes': {}}
            best_opportunity = None
            max_score = 0

            for timeframe in self.config.TIMEFRAMES:
                df = self.data_manager.fetch_klines_full(symbol, timeframe, self.config.MIN_HISTORY_DAYS)
                if df.empty:
                    continue

                # фильтр нисходящего тренда (ниже EMA200)
                if self.config.EXCLUDE_BELOW_EMA200 and len(df) >= 210:
                    ema200 = self.data_manager.calculate_ema(df['close'], 200)
                    if df['close'].iloc[-1] < ema200.iloc[-1]:
                        continue

                levels = self.resistance_analyzer.find_resistance_levels(df, symbol)
                combined = levels.get('combined', [])
                if not combined:
                    continue

                breakout_info = self.breakout_detector.check_breakout(symbol, combined, df)

                score = self._score_symbol(df, combined, breakout_info)
                analysis_result['timeframes'][timeframe] = {
                    'levels_count': len(combined),
                    'has_breakout': bool(breakout_info),
                    'score': score,
                    'breakout_info': breakout_info
                }

                if score > max_score:
                    max_score = score
                    best_opportunity = {
                        'symbol': symbol,
                        'score': score,
                        'timeframe': timeframe,
                        'breakout_info': breakout_info,
                        'levels': combined
                    }

            self.market_snapshot[symbol] = analysis_result
            return best_opportunity
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

    def _score_symbol(self, df: pd.DataFrame, levels: List[Dict], breakout_info: Optional[Dict]) -> float:
        score = 0.0
        current_price = df['close'].iloc[-1]

        if breakout_info:
            score += 50
            score += float(breakout_info.get('confidence_score', 0.0)) * 20
            return score

        if levels:
            nearest = levels[0]
            distance_percent = abs(nearest['distance_percent'])
            if distance_percent < 1: score += 30
            elif distance_percent < 2: score += 20
            elif distance_percent < 3: score += 10
            score += float(nearest['strength']) * 15

        rsi = self.data_manager.calculate_rsi(df['close']).iloc[-1]
        if 50 < rsi < 70:
            score += 10
        elif rsi >= 70:
            score += 5

        return score

    def get_market_summary(self) -> Dict:
        total_analyzed = len(self.market_snapshot)
        with_levels = sum(1 for s in self.market_snapshot.values()
                          if any(tf.get('levels_count', 0) > 0 for tf in s['timeframes'].values()))
        with_breakouts = sum(1 for s in self.market_snapshot.values()
                             if any(tf.get('has_breakout', False) for tf in s['timeframes'].values()))

        top_opps = self.breakout_opportunities[:5] if self.breakout_opportunities else []
        return {
            'timestamp': datetime.now(),
            'total_analyzed': total_analyzed,
            'coins_with_levels': with_levels,
            'active_breakouts': with_breakouts,
            'top_opportunities': [
                {
                    'symbol': o['symbol'],
                    'timeframe': o.get('timeframe', ''),
                    'score': o['score'],
                    'action': 'BUY' if o.get('breakout_info') else 'WATCH',
                    # пробуем вытащить вероятность из breakout_info, если она есть
                    'arf_proba': (o.get('breakout_info') or {}).get('arf_proba')
                } for o in top_opps
            ],
            'market_strength': self._calculate_market_strength()
        }

    def _calculate_market_strength(self) -> str:
        if not self.breakout_opportunities:
            return 'WEAK'
        avg = np.mean([o['score'] for o in self.breakout_opportunities])
        if avg > 70: return 'VERY_STRONG'
        if avg > 50: return 'STRONG'
        if avg > 30: return 'NEUTRAL'
        return 'WEAK'
