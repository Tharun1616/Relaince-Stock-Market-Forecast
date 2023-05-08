# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 00:18:52 2023

@author: HP
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import fbprophet as fbp
from fbprophet.plot import plot_plotly, plot_components_plotly
import os


os.environ["HTTP_PROXY"] = "<yourproxy>"

# Set page title
st.set_page_config(page_title='Relaince stock prediction', layout='wide')

# Set the start and end date
start_date = '2015-01-01'
end_date = '2023-02-01'

# Set the ticker
ticker = 'RELIANCE.NS'

# Get the data
stock_data = yf.download(ticker, start_date, end_date)


# Display data
st.title('Relaince Stock Prediction')
st.write('Data for ticker:', ticker)
st.write(stock_data.head())


# Create a new column with Date as index
stock_data.reset_index(inplace=True)
stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
stock_new = stock_data.copy()
stock_data.set_index(stock_data['Date'], inplace=True, drop=True)
stock_data.drop(['Date'], axis=1)


# Display line chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_data['Date'], y=stock_data['Close'], name='Close'))
fig.update_layout(title='Closing Price Over Time',
                  xaxis_title='Date',
                  yaxis_title='Price (INR)')
st.plotly_chart(fig)

# Display candlestick chart
fig = go.Figure()
fig.add_trace(go.Candlestick(x=stock_data['Date'],
                             open=stock_data['Open'],
                             high=stock_data['High'],
                             low=stock_data['Low'],
                             close=stock_data['Close'], name='Candlestick'))
fig.update_layout(title='Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price (INR)')
st.plotly_chart(fig)

# Display volume chart
fig = go.Figure()
fig.add_trace(go.Bar(x=stock_data['Date'],
              y=stock_data['Volume'], name='Volume'))
fig.update_layout(title='Volume Over Time',
                  xaxis_title='Date',
                  yaxis_title='Volume')
st.plotly_chart(fig)

# Display descriptive statistics
st.write('Descriptive Statistics')
st.write(stock_data.describe())

# Display distribution of closing price
fig = go.Figure()
fig.add_trace(go.Histogram(x=stock_data['Close'], nbinsx=30))
fig.update_layout(title='Distribution of Closing Price',
                  xaxis_title='Price (INR)',
                  yaxis_title='Count')
st.plotly_chart(fig)

# Select a column as the time series data
ts_data = stock_data["Close"]

# Perform decomposition using statsmodels
decomposition = sm.tsa.seasonal_decompose(
    ts_data, model="additive", period=365)

# Visualize seasonality, trend, and residuals
st.write("Seasonality")
st.line_chart(decomposition.seasonal)
st.write("Trend")
st.line_chart(decomposition.trend)
st.write("Residuals")
st.line_chart(decomposition.resid)


# Compute the moving averages
# Sidebar
period = st.sidebar.selectbox(
    'Select the Moving Average Period', ('30 days', '60 days', '6 months', '1 year'))

if period == '30 days':
    window_size = 30
elif period == '60 days':
    window_size = 60
elif period == '6 months':
    window_size = 180
else:
    window_size = 365

# Calculate moving average
rolling_mean = stock_data['Close'].rolling(window=window_size).mean()

# Plot data and moving average
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_data['Date'], y=stock_data['Close'], name='Close'))
fig.add_trace(go.Scatter(x=stock_data['Date'], y=rolling_mean, name='MA'))
fig.update_layout(title='Moving Average',
                  xaxis_title='Date',
                  yaxis_title='Price (INR)')
st.plotly_chart(fig)


# Model Building
# DF for HWE
stock_df = stock_new.copy()
##stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
##stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')


# DF for FbProphet
stocks_df_fb = stock_df.copy()
stocks_df_fb = stocks_df_fb[["Date", "Close"]]
stocks_df_fb = stocks_df_fb.rename(columns={"Date": "ds", "Close": "y"})


forecast_box = st.sidebar.checkbox('Select to Forecast')

if forecast_box:

    models = ('HWE_MS_AT', 'FBPROPHET')
    selected_model = st.sidebar.selectbox(
        'Select Model for prediction', models)

    n_days = st.sidebar.number_input('Days to predict', 1, 400)
    period = n_days*1

    # options to select plot
    options = st.sidebar.multiselect(
        'Select Plot',
        ['Total', 'Predicted'])

    # options to select Data Frame
    select_df = st.sidebar.radio(
        "Select Data",
        ('Total_Table', 'First_Five'), index=1)

    # DF for Forecasted Data
    start = '2023-02-01'
    business_days = pd.date_range(start=start, periods=period, freq=BDay())
    forecast_df = pd.DataFrame(business_days, columns=['Date'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.date
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
    forecast_df['Date'] = forecast_df['Date'].dt.tz_localize(None)

    def total_plot(Date_data, forecast_model_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['Close'], name='Actual'))
        fig.add_trace(go.Scatter(
            x=Date_data, y=forecast_model_data, name='Forecasted'))
        fig.update_layout(xaxis_title='Date',
                          yaxis_title='Price (INR)')
        st.plotly_chart(fig)

    def predicted_plot(Date_data, forecast_model_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Date_data, y=forecast_model_data, name='Forecasted'))
        fig.update_layout(xaxis_title='Date',
                          yaxis_title='Price (INR)')
        st.plotly_chart(fig)

    def Total_predicted_Days(total_pred_df):
        st.write('Total_Predicted_Data')
        st.write(total_pred_df)

    def First_Five_Days(pred_df):
        st.write('Predicted Data for first five Days')
        st.write(pred_df.head())

    if selected_model == 'HWE_MS_AT':
        stock_df["Close"] = stock_df["Close"].astype('double')
        hwe_M_a_model = ExponentialSmoothing(
            stock_df["Close"], seasonal="mul", trend="add", seasonal_periods=12).fit()
        forecast = hwe_M_a_model.forecast(period)
        forecast_df['Close'] = forecast.values

        st.markdown('#')
        st.write(
            "**Forecasting Using Holt's Winter Exp with Additive Seasonality and Mutlipicative trend**")

        # Display line chart for Total
        if 'Total' in options:
            total_plot(forecast_df['Date'], forecast_df['Close'])

        # Display line chart for only  predicted Days
        if 'Predicted' in options:
            predicted_plot(forecast_df['Date'], forecast_df['Close'])

        if select_df == 'Total_Table':
            Total_predicted_Days(forecast_df)

        if select_df == 'First_Five':
            First_Five_Days(forecast_df)

    if selected_model == 'FBPROPHET':
        fb_model_tot = fbp.Prophet(daily_seasonality=True)
        fb_model_tot.fit(stocks_df_fb)
        future = fb_model_tot.make_future_dataframe(periods=period)
        fb_predict = fb_model_tot.predict(future)
        st.markdown('#')
        st.write("**Forecasting Using FBPROPHET**")

        # Display line chart
        if 'Total' in options:
            total_plot(fb_predict['ds'], fb_predict['yhat'])

        # Copying the predicted df into another df
        stocks_df_fb = fb_predict.copy()
        stocks_df_fb = stocks_df_fb[["ds", "yhat"]]
        stocks_df_fb = stocks_df_fb.rename(
            columns={"ds": "Date", "yhat": "Close"})
        stocks_df_fb = stocks_df_fb.loc[stocks_df_fb.Date >= start]
        stocks_df_fb = stocks_df_fb.reset_index(drop=True)

        # Merging the tables using inner join
        #merged_df = pd.merge(forecast_df,fb_predict, left_on = 'Date', right_on = 'ds', how = 'right')
        # Renaming yhat column name
        #merged_df.rename(columns={"yhat": "Close","ds":"DATE"}, inplace=True)

        # Display line chart for only  predicted Days
        if 'Predicted' in options:
            predicted_plot(stocks_df_fb['Date'], stocks_df_fb['Close'])

        if select_df == 'Total_Table':
            Total_predicted_Days(stocks_df_fb)

        if select_df == 'First_Five':
            First_Five_Days(stocks_df_fb)
