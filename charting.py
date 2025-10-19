# BOTISTABL-main/charting.py
from __future__ import annotations
from typing import List, Dict, Optional
import io
import math
import pandas as pd
import numpy as np

# фикс «main thread is not in main loop» от tkinter: форсим headless-бэкенд до любых импортов matplotlib/mplfinance
import matplotlib
matplotlib.use('Agg', force=True)

import mplfinance as mpf



def render_level_chart(
    df: pd.DataFrame,
    levels: List[Dict],
    symbol: str,
    timeframe: str,
    breakout: Optional[Dict] = None,
    max_bars: int = 220,
) -> bytes:
    """
    Рендерит PNG с:
      • свечами
      • горизонтальными линиями уровней (levels[*]['price'])
      • (опц.) маркером пробоя по цене breakout['price'] и времени breakout['ts'] (в мс или ISO)
    """
    if df is None or df.empty:
        raise ValueError("render_level_chart: пустой DataFrame")

    # нормализуем индекс под mplfinance
    dfp = df[['open', 'high', 'low', 'close', 'volume']].copy()
    if not isinstance(dfp.index, pd.DatetimeIndex):
        # пробуем автоконверсию, если есть 'open_time'/'time'/'ts'
        for col in ('open_time', 'time', 'ts'):
            if col in df.columns:
                dfp.index = pd.to_datetime(df[col], unit='ms', errors='coerce') if str(df[col].dtype).startswith('int') \
                            else pd.to_datetime(df[col], errors='coerce')
                break
    dfp = dfp.dropna().tail(int(max_bars))

    # уровни в hlines (передадим параметр только если есть хоть один уровень)
    hprices: List[float] = []
    for l in (levels or []):
        try:
            if isinstance(l, dict) and 'price' in l:
                hprices.append(float(l['price']))
            elif isinstance(l, (int, float)):
                hprices.append(float(l))
        except Exception:
            continue

    addplots: List[dict] = []
    b_price = None
    if isinstance(breakout, dict):
        if breakout.get('price') is not None:
            try:
                b_price = float(breakout['price'])
            except Exception:
                b_price = None
        elif breakout.get('breakout_price') is not None:
            try:
                b_price = float(breakout['breakout_price'])
            except Exception:
                b_price = None
    # дальше — как было: если b_price валиден, строим scatter-точку на нужной свече

        # определим, к какой свече привязать маркер:
        # 1) Если есть breakout['ts'] → ставим на ближайший индекс
        # 2) иначе — на последнюю свечу
        target_idx = None
        if 'ts' in (breakout or {}):
            ts_val = breakout['ts']
            try:
                if isinstance(ts_val, (int, float)):
                    ts_dt = pd.to_datetime(int(ts_val), unit='ms')
                else:
                    ts_dt = pd.to_datetime(ts_val)
                # найти ближайший индекс
                if not dfp.empty:
                    pos = np.searchsorted(dfp.index.values, np.datetime64(ts_dt), side='left')
                    pos = max(0, min(pos, len(dfp.index) - 1))
                    target_idx = dfp.index[pos]
            except Exception:
                target_idx = None

        if target_idx is None and not dfp.empty:
            target_idx = dfp.index[-1]

        if b_price is not None and target_idx is not None:
            s = pd.Series(index=dfp.index, data=np.nan)
            s.loc[target_idx] = b_price
            # стрелка вверх/вниз по направлению (если есть)
            mrk = '^'
            try:
                direction = str(breakout.get('direction', '')).lower()
                if direction.startswith('down'):
                    mrk = 'v'
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")
            addplots.append(
                mpf.make_addplot(
                    s,
                    type='scatter',
                    marker=mrk,
                    markersize=80,  # крупно, чтобы в тг читалось
                )
            )

    # собираем kwargs для mpf.plot, добавляя только непустые опции
    plot_kwargs = dict(
        type='candle',
        volume=True,
        style='yahoo',
        returnfig=True,
        tight_layout=True,
    )
    if hprices:
        plot_kwargs['hlines'] = dict(
            hlines=hprices,
            colors=['#888888'] * len(hprices),
            linestyle='--',
            linewidths=0.8,
        )
    if addplots:
        plot_kwargs['addplot'] = addplots
    fig, _ = mpf.plot(dfp, **plot_kwargs)


    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    # важно: закрыть фигуру, чтобы не накапливать память при циклах
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        from error_logger import log_exception
        log_exception("Failed to import matplotlib")
    return png_bytes
