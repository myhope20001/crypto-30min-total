# -*- coding: utf-8 -*-
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from datetime import datetime, timedelta
import lightgbm as lgb
import threading
import streamlit as st

st.set_page_config(page_title="AI Crypto Trader", layout="wide")

# -----------------------------
# DB 연결
# -----------------------------
DB = "ai_trader.db"
conn = sqlite3.connect(DB, check_same_thread=False)
cur = conn.cursor()

# -----------------------------
# DB 생성
# -----------------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS wallet(
id INTEGER PRIMARY KEY,
krw REAL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS positions(
ticker TEXT PRIMARY KEY,
qty REAL,
buy_price REAL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS trades(
id INTEGER PRIMARY KEY AUTOINCREMENT,
time TEXT,
ticker TEXT,
price REAL,
qty REAL,
side TEXT,
trade_value REAL,
profit REAL,
profit_percent REAL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS learning(
id INTEGER PRIMARY KEY AUTOINCREMENT,
f1 REAL,f2 REAL,f3 REAL,f4 REAL,f5 REAL,
f6 REAL,f7 REAL,f8 REAL,f9 REAL,f10 REAL,
f11 REAL,f12 REAL,f13 REAL,f14 REAL,f15 REAL,
f16 REAL,f17 REAL,f18 REAL,f19 REAL,f20 REAL,
f21 REAL,f22 REAL,f23 REAL,f24 REAL,f25 REAL,
f26 REAL,f27 REAL,f28 REAL,f29 REAL,f30 REAL,
target INTEGER
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS learning_meta(
id INTEGER PRIMARY KEY,
last_time TEXT
)
""")
conn.commit()

# -----------------------------
# 초기값 설정
# -----------------------------
if cur.execute("SELECT * FROM wallet").fetchone() is None:
    cur.execute("INSERT INTO wallet VALUES(1,10000000)")
    conn.commit()
if cur.execute("SELECT * FROM learning_meta").fetchone() is None:
    cur.execute("INSERT INTO learning_meta VALUES(1,'2000-01-01')")
    conn.commit()

# -----------------------------
# 지갑/포지션 관련
# -----------------------------
def load_wallet():
    return cur.execute("SELECT krw FROM wallet WHERE id=1").fetchone()[0]

def save_wallet(krw):
    cur.execute("UPDATE wallet SET krw=? WHERE id=1", (krw,))
    conn.commit()

def load_positions():
    df = pd.read_sql("SELECT * FROM positions", conn)
    return {r.ticker: {"qty": r.qty, "buy_price": r.buy_price} for _, r in df.iterrows()}

def save_position(ticker, qty, buy_price):
    cur.execute("INSERT OR REPLACE INTO positions VALUES(?,?,?)", (ticker, qty, buy_price))
    conn.commit()

def delete_position(ticker):
    cur.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
    conn.commit()

# -----------------------------
# 지표/피처
# -----------------------------
def indicators(df):
    df["ma5"] = df.close.rolling(5).mean()
    df["ma20"] = df.close.rolling(20).mean()
    delta = df.close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))
    df["momentum"] = df.close.pct_change(3)
    return df

def features(df):
    r = df.iloc[-1]
    f = [r.rsi, r.ma5 / r.close, r.ma20 / r.close, r.momentum]
    while len(f) < 30:
        f.append(np.random.random())
    return f[:30]

# -----------------------------
# 거래 가능한 종목 필터링
# -----------------------------
def tradable():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url).json()
    coins = [x["market"] for x in res if x["market"].startswith("KRW-")]
    
    seven_days_ago = datetime.now() - timedelta(days=7)
    new_list = []
    for m in res:
        if m["market"] in coins:
            try:
                if "market_warning" in m and m["market_warning"] in ["CAUTION", "DELISTING"]:
                    continue
                if "listed_date" in m:
                    listed_date = datetime.strptime(m["listed_date"], "%Y-%m-%d")
                    if listed_date > seven_days_ago:
                        continue
            except:
                pass
            new_list.append(m["market"])
    return new_list

def top100():
    coins = tradable()
    data = []
    for c in coins:
        try:
            df = pyupbit.get_ohlcv(c, "minute1", count=20)
            val = (df.close * df.volume).sum()
            data.append((c, val))
        except:
            pass
    data = sorted(data, key=lambda x: x[1], reverse=True)
    return [x[0] for x in data[:100]]

# -----------------------------
# 학습/모델
# -----------------------------
def build_learning():
    coins = top100()
    last_time = pd.to_datetime(cur.execute("SELECT last_time FROM learning_meta WHERE id=1").fetchone()[0])
    for coin in coins:
        df = pyupbit.get_ohlcv(coin, "minute1", count=200)
        if df is None: continue
        df = indicators(df)
        df["target"] = (df.close.shift(-5) > df.close).astype(int)
        df = df.dropna()
        for i in range(len(df)-1):
            row_time = df.index[i]
            if row_time <= last_time: continue
            f = features(df.iloc[:i+1])
            t = df.iloc[i]["target"]
            cur.execute("INSERT INTO learning VALUES(NULL," + ",".join(["?"]*30) + ",?)", f + [t])
    conn.commit()
    cur.execute("UPDATE learning_meta SET last_time=?", (datetime.now(),))
    conn.commit()

def train():
    df = pd.read_sql("SELECT * FROM learning", conn)
    if len(df) < 3000: return None
    X = df.drop(["id", "target"], axis=1)
    y = df["target"]
    d = lgb.Dataset(X, label=y)
    params = {"objective":"binary","learning_rate":0.03,"num_leaves":64}
    model = lgb.train(params, d, 150)
    return model

# -----------------------------
# 매매
# -----------------------------
def trade(model):
    krw = load_wallet()
    positions = load_positions()
    coins = top100()
    # BUY
    for coin in coins:
        if coin in positions: continue
        df = pyupbit.get_ohlcv(coin, "minute1", count=120)
        if df is None: continue
        df = indicators(df)
        f = features(df)
        prob = model.predict([f])[0]
        if prob < 0.6: continue
        price = pyupbit.get_current_price(coin)
        invest = krw * 0.1
        if invest < 10000: continue
        qty = invest / price
        krw -= invest
        save_wallet(krw)
        save_position(coin, qty, price)
        trade_value = invest
        cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?,?,?,?)",
                    (datetime.now(), coin, price, qty, "BUY", trade_value, 0.0, 0.0))
        conn.commit()
    # SELL
    positions = load_positions()
    for coin, pos in positions.items():
        price = pyupbit.get_current_price(coin)
        profit = price - pos["buy_price"]
        profit_percent = profit / pos["buy_price"] * 100
        f = features(pyupbit.get_ohlcv(coin, "minute1", count=120))
        prob = model.predict([f])[0]
        if prob < 0.45 or profit_percent > 8 or profit_percent < -3:
            qty = pos["qty"]
            krw = load_wallet()
            krw += qty * price
            save_wallet(krw)
            delete_position(coin)
            trade_value = qty * price
            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?,?,?,?)",
                        (datetime.now(), coin, price, qty, "SELL", trade_value, profit, profit_percent))
            conn.commit()

# -----------------------------
# 백그라운드 엔진
# -----------------------------
def ai_engine():
    while True:
        build_learning()
        model = train()
        if model:
            trade(model)
        time.sleep(300)  # 5분마다 반복

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Crypto Trader (백그라운드 24시간)")

if "engine_started" not in st.session_state:
    t = threading.Thread(target=ai_engine, daemon=True)
    t.start()
    st.session_state.engine_started = True

# 대시보드
krw = load_wallet()
positions = load_positions()
coin_value = 0
rows=[]
for coin,pos in positions.items():
    price = pyupbit.get_current_price(coin)
    value = price*pos["qty"]
    coin_value += value
    profit = price - pos["buy_price"]
    profit_percent = profit / pos["buy_price"] * 100
    rows.append({
        "coin": coin, "qty": pos["qty"], "buy_price": pos["buy_price"],
        "price": price, "profit": profit, "profit%": profit_percent
    })

asset = krw + coin_value
c1,c2,c3 = st.columns(3)
c1.metric("총 자산", f"{asset:,.0f}")
c2.metric("현금", f"{krw:,.0f}")
c3.metric("코인 평가", f"{coin_value:,.0f}")

st.subheader("보유 코인")
st.dataframe(pd.DataFrame(rows))

# -----------------------------
# 최근 거래 + 합계 계산
# -----------------------------
hist = pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50", conn)

if not hist.empty:
    total_buy = hist.loc[hist.side=="BUY","trade_value"].sum()
    total_sell = hist.loc[hist.side=="SELL","trade_value"].sum()
    total_profit = hist.loc[hist.side=="SELL","profit"].sum()
    # 수익률 = 총 수익 / 총 매수금액 * 100
    total_profit_percent = (total_profit / total_buy * 100) if total_buy>0 else 0
    # 합계 행 추가
    sum_row = pd.DataFrame([{
        "time": "합계",
        "ticker": "",
        "price": "",
        "qty": "",
        "side": "",
        "trade_value": f"{total_buy:,.0f}/{total_sell:,.0f}",
        "profit": f"{total_profit:,.0f}",
        "profit_percent": f"{total_profit_percent:.2f}%"
    }])
    hist_display = pd.concat([hist, sum_row], ignore_index=True)
else:
    hist_display = hist

st.subheader("최근 거래 (금액/수익 표시)")
st.dataframe(hist_display)

st.write("⚠️ Streamlit 종료 후에도 백그라운드 엔진이 5분마다 학습과 매매를 진행하며 DB에 상태를 저장합니다.")
