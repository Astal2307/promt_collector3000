import ccxt
import time
from datetime import datetime, timezone
from db_controller import DBController

def to_iso_utc_ms(ms):
    """ms -> ISO8601 'YYYY-MM-DD HH:MM:SS' в UTC (строка)"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def fetch_ohlcv_all(exchange, market_symbol: str, timeframe: str, since_ms=None, limit: int = 1000):
    all_rows = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        cursor = batch[-1][0] + 1
        time.sleep(0.3)
    return all_rows

def make_db_rows(symbol_db: str, timeframe: str, ohlcv_rows):
    out = []
    for ts, o, h, l, c, v in ohlcv_rows:
        out.append((
            str(symbol_db),            # symbol
            str(timeframe),            # interval
            to_iso_utc_ms(int(ts)),    # datetime (str)
            float(o),
            float(h),
            float(l),
            float(c),
            float(v)
        ))
    return out

def load_all_from_tickers():
    db = DBController()
    exchange = ccxt.binance()
    
    tickers = db.select("tickers")

    for row in tickers:
        symbol_db, base, quote, exch = row[0], row[1], row[2], row[3]
        market_symbol = f"{base}/{quote}"
        print(f"[LOADING] {market_symbol}")

        try:
            # недельные
            weekly = fetch_ohlcv_all(exchange, market_symbol, "1w")
            weekly_rows = make_db_rows(symbol_db, "1w", weekly)
            if weekly_rows:
                db.executemany(
                    table="candles",
                    columns=["symbol", "interval", "datetime", "open", "high", "low", "close", "volume"],
                    rows=weekly_rows
                )

            # дневные
            daily = fetch_ohlcv_all(exchange, market_symbol, "1d")
            daily_rows = make_db_rows(symbol_db, "1d", daily)
            if daily_rows:
                db.executemany(
                    table="candles",
                    columns=["symbol", "interval", "datetime", "open", "high", "low", "close", "volume"],
                    rows=daily_rows
                )

            # часовые
            hourly = fetch_ohlcv_all(exchange, market_symbol, "1h")
            hourly_rows = make_db_rows(symbol_db, "1h", hourly)
            if hourly_rows:
                db.executemany(
                    table="candles",
                    columns=["symbol", "interval", "datetime", "open", "high", "low", "close", "volume"],
                    rows=hourly_rows
                )

            print(f"[UPLOADED] {market_symbol}: {len(weekly_rows)} (1W)  &&  {len(daily_rows)} (1D)  &&  {len(hourly_rows)} (1H)")

        except Exception as e:
            print(f"[ERROR] {market_symbol}: {e}")
            continue

    db.close()

if __name__ == "__main__":
    load_all_from_tickers()
