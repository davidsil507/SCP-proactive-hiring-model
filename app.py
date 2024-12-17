import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from pandas.tseries.offsets import DateOffset
from prophet import Prophet

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

import streamlit as st

from Forecasts import *

def calculate_understaffing_alert(predicted_df, threshold=0.1, model_name="Model"):
    """
    Function to calculate potential understaffing instances for the predicted data.
    Assumes that the dataframe contains 'Predicted_FTEs' and 'Capacity' columns.
    """
    understaffing_alerts = []

    for index, row in predicted_df.iterrows():
        # Check if the predicted shift demand is within the threshold of the predicted capacity
        if (row['Capacity'] <= row['Predicted_FTEs']) or (abs(row['Predicted_FTEs'] - row['Capacity']) <= threshold * row['Capacity']):
            understaffing_alerts.append({
                'Time': row['Time'],
                'Predicted FTEs': row['Predicted_FTEs'],
                'Capacity': row['Capacity'],
                'Difference': row['Capacity'] - row['Predicted_FTEs'],
                'Alert': 'Understaffing Predicted',
                'Model': model_name  # Indicate which model triggered the alert
            })

    # Return as a DataFrame
    return pd.DataFrame(understaffing_alerts)

def plot_shift_demand_forecast(models_selected, data, county, service_line):
    fig = go.Figure()

    # Plot historical data
    df_filtered = data[(data['County'] == county) & (data['Service_Line'] == service_line)]
    fig.add_trace(go.Scatter(
        x=df_filtered['Time'],
        y=df_filtered['FTEs'],
        mode='lines',
        name='Historical FTEs',
        line=dict(color='blue'),
        hovertemplate='<b>Time:</b> %{x}<br><b>FTEs Requested:</b> %{y}<extra></extra>'
    ))

    model_colors = {'Cyclic Trend Model': 'lightgreen', 'Prophet Model': 'coral'}

    # Store each model's forecast
    for model_name, model_func in models_selected.items():
        future_predictions_filtered = model_func(data, county, service_line)
        if not future_predictions_filtered.empty:
            fig.add_trace(go.Scatter(
                x=future_predictions_filtered['Time'],
                y=future_predictions_filtered['Predicted_FTEs'],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=model_colors.get(model_name, 'gray')),
                hovertemplate='<b>Time:</b> %{x}<br><b>Predicted FTEs:</b> %{y}<extra></extra>'
            ))

    # Calculate and plot capacity
    capacity_df = capacity_data[(capacity_data['County'] == county) & (capacity_data['Service_Line'] == service_line)]

    fig.add_trace(go.Scatter(
        x=capacity_df['Time'],
        y=capacity_df['Capacity'],
        mode='lines',
        name='Capacity',
        line=dict(color='purple', dash='dash'),
        hovertemplate='<b>Time:</b> %{x}<br><b>Capacity:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': f'Shift Demand Forecast for {county} ({service_line})', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Time',
        yaxis_title='No. of FTEs',
        xaxis_tickformat='%Y-%m',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        margin=dict(l=40, r=40, t=80, b=40),
        height=500,
        width=1250
    )

    return fig

def calculate_metrics(y_true, y_pred):
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return wape, mape, mse, rmse

def display_metrics_table(county, service_line, metrics_seasonality, metrics_prophet):
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Model': ['Cyclic Trend', 'Prophet'],
        'WAPE': [metrics_seasonality[0], metrics_prophet[0]],
        'MAPE': [metrics_seasonality[1], metrics_prophet[1]],
        'MSE': [metrics_seasonality[2], metrics_prophet[2]],
        'RMSE': [metrics_seasonality[3], metrics_prophet[3]]
    })

    st.write(f"Metrics for {county} - {service_line}")
    st.table(metrics_df)

    # Provide explanations for each metric with expanders
    with st.expander("What is WAPE?"):
        st.write("WAPE (Weighted Absolute Percentage Error) is the sum of the absolute errors, weighted by the actual values. "
                 "It helps understand the error in terms of total volume. A lower WAPE indicates a better model fit.")
        st.latex(r"\text{WAPE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|}")

    with st.expander("What is MAPE?"):
        st.write("MAPE (Mean Absolute Percentage Error) measures the average percentage error between predicted and actual values. "
                 "It’s a common metric in forecasting, where a lower MAPE value represents higher accuracy.")
        st.latex(r"\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|")

    with st.expander("What is MSE?"):
        st.write("MSE (Mean Squared Error) calculates the average of the squared differences between predicted and actual values. "
                 "This metric penalizes larger errors more heavily, and a lower MSE indicates a more accurate model.")
        st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    with st.expander("What is RMSE?"):
        st.write("RMSE (Root Mean Squared Error) is the square root of the MSE. It provides an indication of the model’s error in the "
                 "same units as the original data, making it easier to interpret. Lower RMSE values indicate better accuracy.")
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")

def evaluate_models(data, county, service_line):
    # Filter historical data for the specific facility and service line
    df_filtered = data[(data['County'] == county) & (data['Service_Line'] == service_line)]
    if df_filtered.empty:
        st.write(f"No data available for {county} - {service_line}.")
        return

    # Sort and split data into train (80%) and test (20%)
    df_filtered = df_filtered.sort_values('Time')
    train_size = int(len(df_filtered) * 0.8)
    train_data = df_filtered.iloc[:train_size]
    test_data = df_filtered.iloc[train_size:]

    # Ensure train and test sets are non-empty
    if train_data.empty or test_data.empty:
        st.write("Insufficient data in the selected date range for training or testing.")
        return

    # Extract month from the 'Time' column for Cyclic Trend Modeling
    train_data['Month'] = train_data['Time'].dt.month
    test_data['Month'] = test_data['Time'].dt.month

    # Generate dummy variables based on the 'Month' column
    df_train_dummies = pd.get_dummies(train_data, columns=['Month'], drop_first=True)
    features = [col for col in df_train_dummies.columns if 'Month_' in col]  # Adjusted to check for 'Month_' prefix

    # Check for presence of seasonal features
    if not features:
        st.write("No seasonal features available in the dataset for modeling.")
        return
    
    X_train = df_train_dummies[features]
    y_train = train_data['FTEs']

    # Validate non-empty X_train and y_train
    if X_train.empty or y_train.empty:
        st.write("Training data is empty after filtering or encoding.")
        return

    seasonality_model = LinearRegression()
    seasonality_model.fit(X_train, y_train)

    # Prepare test data with same dummies as train
    df_test_dummies = pd.get_dummies(test_data, columns=['Month'], drop_first=True)
    for col in features:
        if col not in df_test_dummies.columns:
            df_test_dummies[col] = 0
    X_test = df_test_dummies[features]
    y_test = test_data['FTEs']

    # Validate non-empty X_test and y_test
    if X_test.empty or y_test.empty:
        st.write("Testing data is empty after filtering or encoding.")
        return

    # Predict and calculate metrics for Cyclic Trend Model
    y_pred_seasonality = seasonality_model.predict(X_test)
    metrics_seasonality = calculate_metrics(y_test, y_pred_seasonality)

    # Prophet Model
    prophet_data = train_data[['Time', 'FTEs']].rename(columns={'Time': 'ds', 'FTEs': 'y'})
    prophet_model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
    prophet_model.fit(prophet_data)
    
    future = test_data[['Time']].rename(columns={'Time': 'ds'})
    prophet_forecast = prophet_model.predict(future)
    y_pred_prophet = prophet_forecast['yhat'].values

    metrics_prophet = calculate_metrics(y_test, y_pred_prophet)

    # Display the metrics in Streamlit as a table
    display_metrics_table(county, service_line, metrics_seasonality, metrics_prophet)

file_1 = st.file_uploader(
    "Submit Volume Data", accept_multiple_files=False, type=["csv", "xlsx"]
)

file_2 = st.file_uploader(
    "Submit Capacity Data", accept_multiple_files=False, type=["csv", "xlsx"]
)

# Check if both files are uploaded
if file_1 and file_2:
    # Read the uploaded files
    try:
        # Assuming both are CSV files for simplicity; adapt if needed
        volume_data = pd.read_csv(file_1)
        capacity_data = pd.read_csv(file_2)

        # Display a success message
        st.success("Both files uploaded successfully!")

        # Streamlit App UI for county and service line


        client_data = pd.read_csv('ClientData.csv') # Ensure ClientData.csv is in the same directory as code

        df = pd.merge(volume_data, client_data, left_on='GlobalFacilityID', right_on='Global Facility ID', how='inner')
        df['Account / Hospital Name'] = df['Account / Hospital Name'].str.replace(r'[ -]?\s?(HM|CC)$', '', regex=True)
        df.rename(columns={'MonthYear':'Time', 'PatientVolume':'FTEs', 'Account / Hospital Name':'Facility', 'Legacy Service Line':'Service_Line'}, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y')
        df['FTEs'] = df['FTEs'] / 120 # Converting from shift hours to # of FTEs. This assumes that 120 hours = 1 FTE. Adapt if necessary
        df = df.groupby(['Time', 'County', 'Service_Line'], as_index=False).agg({'FTEs': 'sum'})

        capacity_data = pd.merge(capacity_data, client_data, left_on='GlobalFacilityID', right_on='Global Facility ID', how='inner')
        capacity_data['Account / Hospital Name'] = capacity_data['Account / Hospital Name'].str.replace(r'[ -]?\s?(HM|CC)$', '', regex=True)
        capacity_data.rename(columns={'MonthYear':'Time', 'Account / Hospital Name':'Facility', 'Legacy Service Line':'Service_Line'}, inplace=True)
        capacity_data['Time'] = pd.to_datetime(capacity_data['Time'])
        capacity_data['Capacity'] = capacity_data['Capacity'] / 120 # Converting from shift hours to # of FTEs. This assumes that 120 hours = 1 FTE. Adapt if necessary
        capacity_data = capacity_data.groupby(['Time', 'County', 'Service_Line'], as_index=False).agg({'Capacity': 'sum'})

        # Update counties and service lines map
        counties = df['County'].unique().tolist()
        county_service_map = df.groupby('County')['Service_Line'].unique().to_dict()

        st.markdown("<div class='title-box'><h1>Shift Demand Forecast</h1></div>", unsafe_allow_html=True)

        # Layout dropdowns and checkboxes
        col1, col2 = st.columns([1.5, 2])

        with st.markdown("<div class='container'>", unsafe_allow_html=True):
            with col1:
                county = st.selectbox("Select County:", counties)
                available_service_lines = county_service_map[county]
                service_line = st.selectbox("Select Service Line:", available_service_lines)

            with col2:
                st.write("Select Model:")
                models = {"Cyclic Trend Model": seasonality_forecasting, "Prophet Model": prophet_model}
                selected_models = {name: models[name] for name in models if st.checkbox(name)}

                with st.expander("What is the Cyclic Trend Model?"):
                    st.write("The Cyclic Trend Model is a linear regression model that captures seasonal patterns "
                                "by using dummy variables for each month. This model assumes that the demand follows "
                                "a regular seasonal cycle and uses this information to forecast future values. It is "
                                "suitable for data with strong, recurring seasonal trends.")

                with st.expander("What is the Prophet Model?"):
                    st.write("Prophet is an open-source time series forecasting tool developed by Facebook, known for "
                                "handling seasonality, holidays, and trend changes in the data. It uses an additive model "
                                "to combine seasonal and trend components, making it effective for data with irregular "
                                "seasonal patterns and abrupt shifts. Prophet is especially useful for datasets with varying "
                                "trends over time.")

            # Slider for understaffing threshold
            threshold = st.slider("Set Understaffing Threshold (%)", 0.0, 0.5, 0.1)

        if selected_models:
            # Generate forecasts with selected models
            forecast_fig = plot_shift_demand_forecast(selected_models, df, county, service_line)

            # Combine forecasts and add capacity column
            all_forecast_data = []
            for model_name, model_func in selected_models.items():
                future_predictions_filtered = model_func(df, county, service_line)
                if not future_predictions_filtered.empty:
                    future_predictions_filtered['Model'] = model_name
                    all_forecast_data.append(future_predictions_filtered)

            if all_forecast_data:
                all_forecast_data = pd.concat(all_forecast_data)
                capacity_df = capacity_data[(capacity_data['County'] == county) & (capacity_data['Service_Line'] == service_line)]
                capacity_df['Time'] = pd.to_datetime(capacity_df['Time'])
                combined_forecasts = pd.merge(capacity_df, all_forecast_data, left_on='Time', right_on='Time', how='inner')

                # Calculate alerts based on threshold
                alert_df = pd.DataFrame()
                for model_name in selected_models:
                    model_predictions = combined_forecasts[combined_forecasts['Model'] == model_name]
                    model_alerts = calculate_understaffing_alert(model_predictions, threshold, model_name)
                    alert_df = pd.concat([alert_df, model_alerts], ignore_index=True)

                    colors = ['red', 'yellow']
                    cmap = LinearSegmentedColormap.from_list('RedYellow', colors)

                    alert_df_styled = alert_df.style.background_gradient(cmap=cmap, subset=['Difference'])

                # Display the alert table if there are alerts
                if not alert_df.empty:
                    st.write("### Understaffing Alerts")
                    st.dataframe(alert_df_styled)

            # Display the forecast plot
            st.plotly_chart(forecast_fig)
        else:
            st.write("Please select a model to generate the forecast.")

        if selected_models:
            evaluate_models(df, county, service_line)
        else:
            st.write("Please select at least one model to calculate metrics.")

    except Exception as e:
        st.error(f"Error reading files: {e}")
else:
    # Message when files are not uploaded
    st.warning("Please upload both files to proceed.")



