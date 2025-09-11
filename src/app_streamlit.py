# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Ensure local src imports work when run via `streamlit run src/app_streamlit.py`
this_dir = Path(__file__).parent
sys.path.append(str(this_dir))

from utils_simulate import (
    calculate_performance_metrics,
    XGBoostRegressorWrapper
)
from single_target_simulator import (
    load_and_prepare_data,
    Simulate,
    BinaryPositionSizer
)


st.set_page_config(page_title="Capstone Simulator", layout="wide")
st.title("Capstone Simulator UI")
st.write("Run single-target simulations and optional XGBoost complexity sweeps.")

with st.sidebar:
    st.header("Data")
    default_etfs = "SPY,XLK,XLF,XLV"
    etf_text = st.text_input("ETFs (comma-separated; include target)", default_etfs)
    target_etf = st.text_input("Target ETF", "SPY")
    start_date = st.text_input("Start Date (YYYY-MM-DD)", "2015-01-01")

    st.header("Simulation")
    window_size = st.number_input("Window Size", min_value=30, max_value=1000, value=200, step=10)
    window_type = st.selectbox("Window Type", ["expanding", "fixed"], index=0)

    st.header("Model")
    model_choice = st.selectbox(
        "Model",
        [
            "LinearRegression (sklearn)",
            "Ridge (sklearn)",
            "ElasticNetCV (sklearn)",
            "XGBoost (gblinear)",
            "XGBoost (gbtree)"
        ]
    )
    run_btn = st.button("Run Simulation")

def run_simulation(etf_list, target_etf, start_date, window_size, window_type, model_choice):
    X, y, _ = load_and_prepare_data(etf_list, target_etf, start_date=start_date)
    # Trim for speed in UI
    max_rows = min(1000, len(X))
    X = X.iloc[:max_rows]
    y = y.iloc[:max_rows]

    if model_choice == "LinearRegression (sklearn)":
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        pipe_steps = [('scaler', StandardScaler()), ('model', LinearRegression())]
    elif model_choice == "Ridge (sklearn)":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe_steps = [('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]
    elif model_choice == "ElasticNetCV (sklearn)":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        alphas = None  # let ElasticNetCV choose default grid
        pipe_steps = [('scaler', StandardScaler()), ('model', ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=alphas, cv=5, n_jobs=None, max_iter=5000))]
    elif model_choice == "XGBoost (gblinear)":
        from sklearn.preprocessing import StandardScaler
        pipe_steps = [('scaler', StandardScaler()), ('model', XGBoostRegressorWrapper(booster='gblinear', n_estimators=1, learning_rate=1.0))]
    else:
        from sklearn.preprocessing import StandardScaler
        pipe_steps = [('scaler', StandardScaler()), ('model', XGBoostRegressorWrapper(booster='gbtree', max_depth=1, n_estimators=10, learning_rate=0.3))]

    result_df, _ = Simulate(
        X=X,
        y=y,
        window_size=window_size,
        window_type=window_type,
        pipe_steps=pipe_steps,
        param_grid={},
        tag=f"ui_{model_choice.replace(' ', '_').lower()}"
    )

    if result_df.empty:
        return pd.DataFrame()

    common_index = result_df.index.intersection(y.index)
    regout = result_df.loc[common_index].copy()

    # Extract actual values, handling both Series and DataFrame cases
    actual_values = y.loc[common_index]
    if isinstance(actual_values, pd.DataFrame):
        # If it's a DataFrame, extract the first column as Series
        actual_values = actual_values.iloc[:, 0]

    regout['actual'] = actual_values
    sizer = BinaryPositionSizer(-1.0, 1.0)
    regout['leverage'] = sizer.calculate_position(regout['prediction'])
    regout['perf_ret'] = regout['leverage'] * regout['actual']
    return regout


if run_btn:
    try:
        etf_list = [s.strip() for s in etf_text.split(',') if s.strip()]
        regout = run_simulation(etf_list, target_etf, start_date, window_size, window_type, model_choice)
        if regout.empty:
            st.warning("Simulation produced no output")
        else:
            st.subheader("Results (head)")
            st.dataframe(regout.head())

            st.subheader("Performance Metrics")
            metrics = calculate_performance_metrics(regout['perf_ret'].dropna())
            st.json(metrics)

            st.subheader("Cumulative Return")
            cum = (1 + regout['perf_ret'].dropna()).cumprod()
            st.line_chart(cum)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Tip: run with `streamlit run src/app_streamlit.py`")


