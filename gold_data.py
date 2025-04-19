import yfinance as yf
import pandas as pd


def get_recent_gold(window_size=30):
    df = yf.download("GC=F", period="90d", interval="1d")  # lấy 90 ngày
    df = df.reset_index()[['Date', 'Close']].dropna()

    min_val = df['Close'].min()
    max_val = df['Close'].max()

    df['Scaled'] = (df['Close'] - min_val) / (max_val - min_val)

    recent = df[-window_size:].copy()
    return recent[['Date', 'Scaled']].values, min_val, max_val


import yfinance as yf
import pandas as pd

def get_history_for_chart():
    # Lấy dữ liệu giá vàng không group theo ticker
    df = yf.download("GC=F", period="90d", interval="1d", group_by=None)

    # Nếu MultiIndex thì flatten cột
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns]

    df = df.reset_index()

    # Tìm cột gần giống nhất với 'Date' và 'Close'
    date_col = [col for col in df.columns if 'Date' in col][0]
    close_col = [col for col in df.columns if 'Close' in col][0]

    df = df[[date_col, close_col]].dropna()
    df.columns = ['Date', 'Close']  # Rename lại cho frontend dễ hiểu
    df['Date'] = df['Date'].astype(str)

    return df.to_dict(orient="records")
