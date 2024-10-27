import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yahoo
import time
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

start_time = time.time()

@st.cache_data
def loadData(ticker, start_date, end_date):
    stockData = yahoo.download(ticker, start=start_date, end=end_date)
    stockData.reset_index(inplace=True)
    stockData.set_index("Date", inplace=True)
    return stockData

st.title("Stock Price Predictor")
st.subheader("Welcome to Stock Predictor")
st.write("Follow the directions below:")

# Session state initialization
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'selectedStock' not in st.session_state:
    st.session_state.selectedStock = None
if 'startDate' not in st.session_state:
    st.session_state.startDate = pd.to_datetime("2010-01-01")
if 'endDate' not in st.session_state:
    st.session_state.endDate = date.today()
if 'yearSlider' not in st.session_state:
    st.session_state.yearSlider = 1

# Step 1: Choose a Stock Ticker
if st.session_state.step == 1:
    st.subheader("Step 1: Choose a Stock Ticker")
    stockTickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
    selectedStock = st.selectbox("Select Stock To Predict", stockTickers)
    if st.button("Next", key="next1"):
        if selectedStock:
            st.session_state.selectedStock = selectedStock
            st.session_state.step += 1
        else:
            st.warning("Select a stock before moving to the next step")

# Step 2: Select Date Range
if st.session_state.step == 2:
    st.subheader("Step 2: Select Date Range for Data")
    startDate = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    endDate = st.date_input("End Date", value=date.today())
    if st.button("Next", key="next2"):
        if startDate < endDate:
            st.session_state.startDate = startDate
            st.session_state.endDate = endDate
            st.session_state.step += 1
        else:
            st.warning("End date must be after start date")

# Step 3: Choose Prediction Period
if st.session_state.step == 3:
    st.subheader("Step 3: Choose Prediction Period")
    yearSlider = st.slider("How many years to predict into the future?", 1, 4)
    if st.button("Next", key="next3"):
        st.session_state.yearSlider = yearSlider
        st.session_state.step += 1

# Step 4: Customize and Run the Prediction Models
if st.session_state.step == 4:
    st.subheader("Step 4: View Predictions and Analysis")
    dataLoadUpdate = st.text("Loading Stock Data...")

    # Load stock data
    stockData = loadData(st.session_state.selectedStock, st.session_state.startDate, st.session_state.endDate)
    dataLoadUpdate.text("Stock Data Loaded.")

    st.subheader("Raw Data")
    st.write(stockData.tail())

    # Prepare data for Prophet model
    sdp = stockData[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize and fit the Prophet model
    # Hyperparameter tuning inputs
    st.subheader("Prophet Model Hyperparameters")
    changepoint_prior_scale = st.slider('Changepoint Prior Scale (Trend Flexibility)', 0.001, 0.5, 0.05)
    seasonality_mode = st.selectbox('Seasonality Mode', ['additive', 'multiplicative'])
    seasonality_prior_scale = st.slider('Seasonality Prior Scale (Seasonality Strength)', 1.0, 10.0, 10.0)

    prophetModel = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    prophetModel.fit(sdp)

    # Cross-validation for Prophet model
    st.subheader("Prophet Model Cross-Validation and Performance Metrics")
    with st.spinner('Performing cross-validation...'):
        # Cross-validation parameters
        initial_days = int(len(sdp) * 0.7)
        initial = f'{initial_days} days'
        period = '180 days'  # Predictions every 180 days
        horizon = '365 days'  # Forecast horizon

        df_cv = cross_validation(prophetModel, initial=initial, period=period, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv)

    st.write("Cross-Validation Metrics:")
    st.write(df_p)

    # Plot cross-validation performance
    from prophet.plot import plot_cross_validation_metric
    fig_cv = plot_cross_validation_metric(df_cv, metric='rmse')
    st.write(fig_cv)

    # Make future dataframe for predictions
    future_periods = st.session_state.yearSlider * 365
    future_dates = prophetModel.make_future_dataframe(periods=future_periods)

    # Predict future prices
    forecast = prophetModel.predict(future_dates)

    # Plot actual vs predicted prices
    st.subheader("Prophet Model: Actual vs Predicted Prices")
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=sdp['ds'],
        y=sdp['y'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    fig_all.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig_all.update_layout(
        title=f"Actual vs Predicted Prices for {st.session_state.selectedStock}",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_all)

    # Plot future predictions
    st.subheader("Prophet Model: Future Predictions")
    fig_future = plot_plotly(prophetModel, forecast)
    fig_future.update_layout(
        title=f"Future Stock Price Predictions for {st.session_state.selectedStock}",
        xaxis_title="Date",
        yaxis_title="Predicted Stock Price",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_future)

    # Display forecasted data
    st.subheader(f"Prophet Model: Forecasted Prices for Next {st.session_state.yearSlider} Year(s)")
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds').tail(future_periods)
    st.write(future_forecast)

    # Calculate profit predictions
    # Assuming we buy at the last available actual price and sell at future predicted prices
    last_actual_price = sdp['y'].iloc[-1]
    future_forecast['Profit'] = future_forecast['yhat'] - last_actual_price
    total_predicted_profit = future_forecast['Profit'].sum()
    st.write(f"Total Predicted Profit over the next {st.session_state.yearSlider} year(s): ${total_predicted_profit:.2f}")

    # Plot predicted profit over time
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Bar(
        x=future_forecast.index,
        y=future_forecast['Profit'],
        name='Predicted Profit per Day'
    ))
    fig_profit.update_layout(
        title=f"Predicted Profit per Day for {st.session_state.selectedStock}",
        xaxis_title="Date",
        yaxis_title="Profit ($)",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_profit)

    # --- Random Forest Classifier for Trend Prediction ---

    st.subheader("Random Forest Classifier for Trend Prediction")

    # Prepare data for Random Forest Classifier
    rf_data = stockData.copy()
    rf_data['Target'] = np.where(rf_data['Close'].shift(-1) > rf_data['Close'], 1, 0)
    rf_data.dropna(inplace=True)

    # Feature Engineering: Adding Technical Indicators
    rf_data['SMA_5'] = rf_data['Close'].rolling(window=5).mean()
    rf_data['SMA_10'] = rf_data['Close'].rolling(window=10).mean()
    rf_data['SMA_15'] = rf_data['Close'].rolling(window=15).mean()
    rf_data['EMA_5'] = rf_data['Close'].ewm(span=5, adjust=False).mean()
    rf_data['EMA_10'] = rf_data['Close'].ewm(span=10, adjust=False).mean()
    rf_data['EMA_15'] = rf_data['Close'].ewm(span=15, adjust=False).mean()
    rf_data['Momentum'] = rf_data['Close'] - rf_data['Close'].shift(5)
    rf_data['Volatility'] = rf_data['Close'].rolling(window=5).std()

    # Drop rows with NaN values after adding indicators
    rf_data.dropna(inplace=True)

    # Define features and target variable
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_15',
                'EMA_5', 'EMA_10', 'EMA_15', 'Momentum', 'Volatility']
    target = 'Target'

    # Split data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(rf_data) * split_ratio)
    X_train = rf_data[features].iloc[:split_index]
    y_train = rf_data[target].iloc[:split_index]
    X_test = rf_data[features].iloc[split_index:]
    y_test = rf_data[target].iloc[split_index:]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    rfcModel = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    rfcModel.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = rfcModel.predict(X_test_scaled)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    st.write(cm_df)

    # Plot predicted trends vs actual trends
    st.subheader("Predicted vs Actual Trends")
    trend_dates = rf_data.index[split_index:]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_dates,
        y=y_test.values,
        mode='lines',
        name='Actual Trend',
        line=dict(color='blue', width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_dates,
        y=y_pred,
        mode='lines',
        name='Predicted Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig_trend.update_layout(
        title=f"Actual vs Predicted Trends for {st.session_state.selectedStock}",
        xaxis_title="Date",
        yaxis_title="Trend (0=Down, 1=Up)",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_trend)

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Execution Time: {elapsed_time:.2f} seconds")
    st.subheader("Restart App")
    if st.button("Restart"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
