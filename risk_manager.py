import numpy as np
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config, data_manager):
        self.cfg = config
        self.dm = data_manager

    def correlation_check(self, candidate_symbol: str, active_symbols: List[str]) -> bool:
        """Возвращает True, если средняя корреляция с активными позициями ниже порога."""
        if not active_symbols:
            return True
        try:
            rets = []
            for sym in [candidate_symbol] + active_symbols:
                df = self.dm.fetch_klines_full(sym, '1h', min_days=7)
                if df.empty:
                    return True  # не можем посчитать — не блокируем
                r = df['close'].pct_change().dropna().tail(self.cfg.CORRELATION_WINDOW).rename(sym)
                rets.append(r)
            mat = pd.concat(rets, axis=1).dropna()
            if mat.shape[0] < 10:
                return True
            corr = mat.corr()
            cand_corrs = corr.loc[candidate_symbol, active_symbols].abs()
            mean_corr = float(cand_corrs.mean())
            return mean_corr <= self.cfg.MAX_MEAN_CORRELATION
        except Exception as e:
            logger.warning(f"Ошибка корреляционного фильтра: {e}")
            return True
