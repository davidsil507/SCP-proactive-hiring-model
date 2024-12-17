# Forecasting Algorithms

import pandas as pd
import numpy as np

from pandas.tseries.offsets import DateOffset
from prophet import Prophet

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def seasonality_forecasting(data, county, service_line):
    # Filter data for specific county and service line
    df_filtered = data[(data['County'] == county) & (data['Service_Line'] == service_line)]
    if df_filtered.empty:
        return pd.DataFrame()

    # Create dummy variables for the filtered data
    df_filtered['Month'] = df_filtered['Time'].dt.month
    df_filtered_dummies = pd.get_dummies(df_filtered, columns=['Month'], drop_first=True)

    # Define features and check if they exist
    features = ['Month_' + str(i) for i in range(2, 13)]  # Months 2 to 12
    missing_columns = [col for col in features if col not in df_filtered_dummies.columns]
    if missing_columns:
        st.warning(f"Warning: Missing columns for months {', '.join(missing_columns)}. The forecast may be inaccurate.")
        return pd.DataFrame()

    # Filter data to match the available columns
    X_filtered = df_filtered_dummies[features].fillna(0)
    y_filtered = df_filtered_dummies['FTEs']

    # Train model
    model_filtered = LinearRegression()
    model_filtered.fit(X_filtered, y_filtered)

    # Future data
    last_date = data['Time'].max()
    future_dates = pd.date_range(start=last_date + DateOffset(months=1), periods=12, freq='M')
    future_months = future_dates.month
    future_data = pd.DataFrame({'Month': future_months})
    future_data = pd.get_dummies(future_data, columns=['Month'], drop_first=True)

    # Ensure future_data matches feature set
    for col in features:
        if col not in future_data.columns:
            future_data[col] = 0

    # Predict future demand
    y_future_pred_filtered = model_filtered.predict(future_data)
    future_predictions_filtered = pd.DataFrame({
        'Time': future_dates,
        'Predicted_FTEs': y_future_pred_filtered
    })

    return future_predictions_filtered

def prophet_model(df, county, service_line):
    # Filter data for specific county and service line
    df_filtered = df[(df['County'] == county) & (df['Service_Line'] == service_line)]
    if df_filtered.empty:
        return pd.DataFrame()

    # Prepare data for Prophet
    df_for_prophet = df_filtered[['Time', 'FTEs']].rename(columns={'Time': 'ds', 'FTEs': 'y'})
    prophet_model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
    prophet_model.fit(df_for_prophet)

    # Create future dates
    future = prophet_model.make_future_dataframe(periods=12, freq='M')
    forecast = prophet_model.predict(future)

    # Extract future forecast
    last_historical_date = df_filtered['Time'].max()
    future_forecast = forecast[forecast['ds'] > last_historical_date]
    future_predictions_filtered = future_forecast[['ds', 'yhat']].rename(columns={'ds': 'Time', 'yhat': 'Predicted_FTEs'})

    return future_predictions_filtered