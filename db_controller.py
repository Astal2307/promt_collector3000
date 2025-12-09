import aiosqlite
from pathlib import Path
import numpy as np
import datetime as dt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import random
import asyncio

TOP50_TICKERS = [
    ("BTCUSDT", "BTC", "USDT", "Binance"),
    ("ETHUSDT", "ETH", "USDT", "Binance"),
    ("BNBUSDT", "BNB", "USDT", "Binance"),
    ("XRPUSDT", "XRP", "USDT", "Binance"),
    ("SOLUSDT", "SOL", "USDT", "Binance"),
    ("ADAUSDT", "ADA", "USDT", "Binance"),
    ("DOGEUSDT", "DOGE", "USDT", "Binance"),
    ("TRXUSDT", "TRX", "USDT", "Binance"),
    ("DOTUSDT", "DOT", "USDT", "Binance"),
    ("MATICUSDT", "MATIC", "USDT", "Binance"),
    ("LTCUSDT", "LTC", "USDT", "Binance"),
    ("AVAXUSDT", "AVAX", "USDT", "Binance"),
    ("LINKUSDT", "LINK", "USDT", "Binance"),
    ("ATOMUSDT", "ATOM", "USDT", "Binance"),
    ("UNIUSDT", "UNI", "USDT", "Binance"),
    ("XLMUSDT", "XLM", "USDT", "Binance"),
    ("ETCUSDT", "ETC", "USDT", "Binance"),
    ("XMRUSDT", "XMR", "USDT", "Binance"),
    ("BCHUSDT", "BCH", "USDT", "Binance"),
    ("VETUSDT", "VET", "USDT", "Binance"),
    ("FILUSDT", "FIL", "USDT", "Binance"),
    ("APTUSDT", "APT", "USDT", "Binance"),
    ("ARBUSDT", "ARB", "USDT", "Binance"),
    ("NEARUSDT", "NEAR", "USDT", "Binance"),
    ("ALGOUSDT", "ALGO", "USDT", "Binance"),
    ("QNTUSDT", "QNT", "USDT", "Binance"),
    ("SANDUSDT", "SAND", "USDT", "Binance"),
    ("MANAUSDT", "MANA", "USDT", "Binance"),
    ("EOSUSDT", "EOS", "USDT", "Binance"),
    ("AAVEUSDT", "AAVE", "USDT", "Binance"),
    ("IMXUSDT", "IMX", "USDT", "Binance"),
    ("GRTUSDT", "GRT", "USDT", "Binance"),
    ("STXUSDT", "STX", "USDT", "Binance"),
    ("RUNEUSDT", "RUNE", "USDT", "Binance"),
    ("FLOWUSDT", "FLOW", "USDT", "Binance"),
    ("HBARUSDT", "HBAR", "USDT", "Binance"),
    ("ICPUSDT", "ICP", "USDT", "Binance"),
    ("LDOUSDT", "LDO", "USDT", "Binance"),
    ("CRVUSDT", "CRV", "USDT", "Binance"),
    ("GMXUSDT", "GMX", "USDT", "Binance"),
    ("KAVAUSDT", "KAVA", "USDT", "Binance"),
    ("SNXUSDT", "SNX", "USDT", "Binance"),
    ("ZECUSDT", "ZEC", "USDT", "Binance"),
    ("CHZUSDT", "CHZ", "USDT", "Binance"),
    ("DYDXUSDT", "DYDX", "USDT", "Binance"),
    ("MINAUSDT", "MINA", "USDT", "Binance"),
    ("KLAYUSDT", "KLAY", "USDT", "Binance"),
    ("ZILUSDT", "ZIL", "USDT", "Binance"),
    ("FTMUSDT", "FTM", "USDT", "Binance"),
    ("PEPEUSDT", "PEPE", "USDT", "Binance")
]


class DBController:
    def __init__(self, db_path="financial_data.db"):
        self.db_path = Path(db_path)
        self.conn = None

    async def connect(self):
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.db_path)
            aiosqlite.register_adapter(np.int64, int)
            aiosqlite.register_adapter(np.int32, int)
            aiosqlite.register_adapter(np.float64, float)
            aiosqlite.register_adapter(np.float32, float)
            # datetime -> ISO8601
            aiosqlite.register_adapter(dt.datetime, lambda x: x.isoformat(sep=" "))
            
            await self.create_tables()
            await self.init_top50_tickers()
    
    async def create_tables(self):
        await self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS tickers (
            symbol TEXT PRIMARY KEY,
            base TEXT,
            quote TEXT,
            exchange TEXT
        );

        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            interval TEXT,
            datetime TIMESTAMP,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, interval, datetime)
        );
        """)
        await self.conn.commit()
    
    async def insert(self, table, data: dict):
        keys = ", ".join(data.keys())
        q_marks = ", ".join("?" * len(data))
        values = tuple(data.values())
        sql = f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({q_marks})"
        await self.conn.execute(sql, values)
        await self.conn.commit()
    
    async def executemany(self, table, columns, rows):
        cols = ", ".join(columns)
        q = ", ".join(["?"] * len(columns))
        sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({q})"
        await self.conn.executemany(sql, rows)
        await self.conn.commit()
    
    async def select(self, table, where=None, params=()):
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return rows
    
    async def init_top50_tickers(self):
        for sym, base, quote, exch in TOP50_TICKERS:
            await self.insert("tickers", {
                "symbol": sym,
                "base": base,
                "quote": quote,
                "exchange": exch
            })
    
    async def sample_data(self, horizon: int = 1, symbols: List[str] = None, 
                         interval: str = None, window_size: int = 20, 
                         num_samples: int = 20, test_period_days: int = 365) -> List[Dict]:
        
        if self.conn is None:
            await self.connect()

        horizon = 1
        symbols = [ticker[0] for ticker in TOP50_TICKERS]
        
        samples = []
        samples_per_symbol = max(1, num_samples // len(symbols))
        
        for symbol in symbols:
            if interval is None:
                interval_s = random.choice(["1h", "1d", "1w"])
            
            symbol_samples = await self._generate_samples_for_symbol(
                symbol=symbol,
                interval=interval_s,
                horizon=horizon,
                window_size=window_size,
                samples_per_symbol=samples_per_symbol,
                test_period_days=test_period_days
            )
            samples.extend(symbol_samples)
        
        random.shuffle(samples)
        if len(samples) > num_samples:
            # pass
            samples = samples[:num_samples]
            
        print(f"Сгенерировано {len(samples)} сэмплов с горизонтом {horizon}")
        return samples
    
    async def _generate_samples_for_symbol(self, symbol: str, interval: str, horizon: int,
                                        window_size: int, samples_per_symbol: int,
                                        test_period_days: int) -> List[Dict]:
        data = await self._get_historical_data(symbol, interval, test_period_days + window_size + horizon + 10)
        if data is None or len(data) < window_size + horizon + 1:
            logging.warning(f"Недостаточно данных для {symbol}")
            return []
        
        samples = []
        available_indices = list(range(window_size, len(data) - horizon))
        
        selected_indices = random.sample(
            available_indices, 
            min(samples_per_symbol, len(available_indices))
        )
        
        for idx in selected_indices:
            context_data = data.iloc[idx - window_size:idx + horizon]
            
            target_idx = idx + horizon
            actual_price = data['Close'].iloc[target_idx]
            test_date = data.index[target_idx]
            
            context_str = self._format_context_data(context_data)
            
            sample = {
                'sample_id': f"{symbol}_{interval}_{test_date.strftime('%Y%m%d')}_{idx}_{horizon}",
                'symbol': symbol,
                'interval': interval,
                'timestamp': test_date.strftime('%Y-%m-%d %H:%M:%S'),
                'context_data': context_str,
                'actual_price': actual_price,
                'test_date': test_date.strftime('%Y-%m-%d'),
                'window_size': window_size,
                'horizon': horizon
            }
            samples.append(sample)
        
        return samples
    
    async def _get_historical_data(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        try:
            query = '''
                SELECT datetime, open, high, low, close, volume 
                FROM candles 
                WHERE symbol = ? AND interval = ?
                AND datetime >= datetime('now', ?)
                ORDER BY datetime ASC
            '''
            
            cursor = await self.conn.execute(query, [symbol, interval, f'-{days} days'])
            rows = await cursor.fetchall()
            await cursor.close()
            
            if rows:
                columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                df = pd.DataFrame(rows, columns=columns)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                return df
            return None
            
        except Exception as e:
            logging.error(f"Ошибка получения данных для {symbol}: {e}")
            return None
    
    def _format_context_data(self, data: pd.DataFrame) -> str:
        lines = ["datetime,open,high,low,close,volume"]
        for timestamp, row in data.iterrows():
            line = (f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')},"
                   f"{row['Open']:.6f},{row['High']:.6f},{row['Low']:.6f},"
                   f"{row['Close']:.6f},{row['Volume']:.2f}")
            lines.append(line)
        return "\n".join(lines)
    
    async def close(self):
        if self.conn:
            await self.conn.close()


if __name__ == "__main__": 
    db = DBController("financial_data.db")
    print("База данных financial_data.db создана и готова к работе")
