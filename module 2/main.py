import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# App title
st.title("Stock Price Predictor App")

# Function to scale data for predictions
def scale_data_for_predictions(google_data):
    # Ensure the 'Close' column exists in google_data
    if 'Close' not in google_data.columns:
        raise ValueError("The 'Close' column is missing in the input data")
    # Scaling data for predictions using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(google_data[['Close']])
    # Return the scaled data and the scaler for inverse scaling later
    return scaled_data, scaler

# User input for stock symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set start and end dates for the stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Button to fetch stock data
fetch_data_button = st.button("Fetch Stock Data")

if fetch_data_button:
    # Error handling for downloading stock data
    try:
        google_data = yf.download(stock, start, end)
        st.success("Stock data fetched successfully!")
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        st.stop()

    # Display the stock data
    st.subheader("Stock Data")
    st.write(google_data)

    # Line Chart for Stock Prices
    st.subheader("Line Chart for Stock Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(google_data.index, google_data['Close'], label='Close Price', color='blue')
    ax.set_title("Line Chart of Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Area Chart for Cumulative Volume
    st.subheader("Area Chart for Cumulative Volume")
    google_data['Cumulative Volume'] = google_data['Volume'].cumsum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(google_data.index, google_data['Cumulative Volume'], color="skyblue", alpha=0.4)
    ax.set_title("Area Chart of Cumulative Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Volume")
    st.pyplot(fig)

    # Scatter Plot for Close vs. Volume
    st.subheader("Scatter Plot for Close Price vs. Volume")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(google_data['Close'], google_data['Volume'], alpha=0.6, color="purple")
    ax.set_title("Scatter Plot: Close Price vs. Volume")
    ax.set_xlabel("Close Price")
    ax.set_ylabel("Volume")
    st.pyplot(fig)

    # Box Plot for Price Distribution
    st.subheader("Box Plot for Closing Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=google_data['Close'], ax=ax, color='orange')
    ax.set_title("Box Plot of Closing Prices")
    ax.set_xlabel("Close Price")
    st.pyplot(fig)

    # Histogram of Closing Prices
    st.subheader("Histogram of Closing Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(google_data['Close'], bins=30, color='blue', alpha=0.7)
    ax.set_title('Histogram of Closing Prices')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Heatmap of Correlations
    st.subheader("Heatmap of Correlations")
    correlation_matrix = google_data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Calculate Moving Averages
    google_data['MA100'] = google_data['Close'].rolling(window=100).mean()
    google_data['MA200'] = google_data['Close'].rolling(window=200).mean()
    google_data['MA250'] = google_data['Close'].rolling(window=250).mean()

    # Plot Combined Moving Averages
    st.subheader("Combined Moving Averages (100, 200, 250 days)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(google_data.index, google_data['Close'], label="Close Price", color='blue')
    ax.plot(google_data.index, google_data['MA100'], label="100-Day MA", color='green', linewidth=2)
    ax.plot(google_data.index, google_data['MA200'], label="200-Day MA", color='orange', linewidth=2)
    ax.plot(google_data.index, google_data['MA250'], label="250-Day MA", color='red', linewidth=2)
    ax.set_title('Stock Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

    # Scaling data for predictions
    scaled_data, scaler = scale_data_for_predictions(google_data)

    # Prepare data for prediction
    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Load model
    try:
        model = load_model(r"C:\Users\DELL\Desktop\module 2\Latest_stock_price_model.keras")
    except FileNotFoundError:
        st.error("Model file not found. Please upload the model file.")
        st.stop()

    # Predict using the model
    try:
        predictions = model.predict(x_data)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        st.stop()

    # Inverse scaling
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Prepare plotting data
    plotting_data = pd.DataFrame({
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    }, index=google_data.index[100:])

    # Display prediction results
    st.subheader("Original values vs Predicted values")
    st.write(plotting_data)

    # Plot results
    st.subheader('Original Close Price vs Predicted Close Price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(google_data.index[:len(google_data['Close'])], google_data['Close'], label='Original Data')
    plt.plot(plotting_data.index, plotting_data['predictions'], label='Predicted Data')
    plt.legend()
    st.pyplot(fig)
