# ================================
# backtest_app.py
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ta.trend import PSARIndicator

st.set_page_config(page_title="Parabolic SAR Backtest", layout="wide")
st.title("Parabolic SAR Strategy Backtest")

# ----------------
# 1. Upload CSV
# ----------------
uploaded_file = st.file_uploader(
    "Upload CSV (datetime, open, high, low, close, volume)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Ensure datetime is parsed
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    # ----------------
    # 2. SAR Parameters
    # ----------------
    st.subheader("SAR Parameters")
    accel = st.number_input("Acceleration factor", min_value=0.001, max_value=0.1, value=0.005, step=0.001)
    max_accel = st.number_input("Maximum factor", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # ----------------
    # 3. Backtest Logic
    # ----------------
    if st.button("Run Backtest"):
        st.info("Running backtest...")

        # Generate SAR
        psar = PSARIndicator(high=df['high'], low=df['low'], close=df['close'],
                             step=accel, max_step=max_accel)
        df['sar'] = psar.psar()
        df['signal'] = np.where(df['close'] > df['sar'], 1, np.where(df['close'] < df['sar'], -1, 0))

        # Trade simulation
        initial_capital = 100000
        transaction_cost = 100
        df['position'] = df['signal'].shift(1).fillna(0)

        df['trade_return'] = np.where(
            df['position'] == 1, (df['close'] / df['close'].shift(1) - 1),
            np.where(df['position'] == -1, (df['close'].shift(1) / df['close'] - 1), 0)
        )
        df['trade_pnl'] = df['trade_return'] * initial_capital
        df['trade_cost'] = np.where(
            (df['position'] != df['position'].shift(1)) & (df['position'] != 0),
            transaction_cost, 0
        )
        df['net_pnl'] = df['trade_pnl'] - df['trade_cost']
        df['equity'] = initial_capital + df['net_pnl'].cumsum()

        # ----------------
        # 4. Summary Metrics
        # ----------------
        daily = df.resample('D', on='datetime').agg({'net_pnl': 'sum', 'equity': 'last'}).dropna()
        daily['daily_ret'] = daily['net_pnl'] / daily['equity'].shift(1).fillna(initial_capital)

        sharpe = (daily['daily_ret'].mean() / daily['daily_ret'].std()) * np.sqrt(252) if daily['daily_ret'].std() > 0 else np.nan
        max_dd = ((daily['equity'].cummax() - daily['equity']) / daily['equity'].cummax()).max()

        summary_metrics = pd.DataFrame({
            'Final_Equity': [df['equity'].iloc[-1]],
            'Total_Return (%)': [(df['equity'].iloc[-1]/initial_capital -1)*100],
            'Sharpe': [sharpe],
            'Max_Drawdown': [max_dd]
        })

        st.subheader("Summary Metrics")
        st.dataframe(summary_metrics)

        # ----------------
        # 5. Equity Curve
        # ----------------
        st.subheader("Equity Curve")
        fig = px.line(df, x='datetime', y='equity', title="Equity Curve")
        st.plotly_chart(fig, use_container_width=True)

        # ----------------
        # 6. Trade Log
        # ----------------
        df['prev_position'] = df['position'].shift(1).fillna(0)
        df['trade_start'] = (df['position'] != df['prev_position']) & (df['position'] != 0)
        df['trade_id'] = df['trade_start'].cumsum()

        trade_log = df[df['trade_id'] > 0].groupby('trade_id').agg(
            entry_time=('datetime', 'first'),
            entry_price=('close', 'first'),
            exit_time=('datetime', 'last'),
            exit_price=('close', 'last'),
            direction=('position', 'first'),
            pnl=('net_pnl', 'sum')
        ).reset_index(drop=True)

        st.subheader("Trade Log")
        st.dataframe(trade_log)

        # ----------------
        # 7. Monthly Returns
        # ----------------
        monthly = df.resample('M', on='datetime').agg({'net_pnl':'sum', 'equity':'last'}).reset_index()
        monthly['monthly_return (%)'] = monthly['equity'].pct_change() * 100

        st.subheader("Monthly Returns")
        st.dataframe(monthly)

        # ----------------
        # 8. Download buttons
        # ----------------
        st.download_button("Download Summary Metrics CSV", summary_metrics.to_csv(index=False), "summary_metrics.csv")
        st.download_button("Download Trade Log CSV", trade_log.to_csv(index=False), "trade_log.csv")
        st.download_button("Download Monthly Returns CSV", monthly.to_csv(index=False), "monthly_returns.csv")
