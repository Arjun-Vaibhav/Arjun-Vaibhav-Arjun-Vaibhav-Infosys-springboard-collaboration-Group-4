import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# App title
st.title("Stock Price Predictor App")

# User input for stock symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set start and end dates for the stock data
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

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

    # Error handling for loading model
    try:
        model = load_model(r"C:\Users\DELL\Desktop\module 2\Latest_stock_price_model.keras")
    except FileNotFoundError:
        st.error("Model file not found. Please upload 'Latest_stock_price_model.h5' in the correct directory.")
        st.stop()

    # Display the stock data
    st.subheader("Stock Data")
    st.write(google_data)

    # Splitting data
    splitting_len = int(len(google_data)*0.7)
    x_test = google_data[['Close']].iloc[splitting_len:]  # Corrected this line

    # Plotting helper function
    def plot_graph(figsize, values, full_data, extra_data=None, extra_dataset=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(values, 'Orange')
        plt.plot(full_data.Close, 'b')
        if extra_data is not None:
            plt.plot(extra_dataset)
        return fig

    # Create moving averages
    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean().dropna()
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean().dropna()
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean().dropna()

    # Display moving averages
    st.subheader('Original Close Price and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

    st.subheader('Original Close Price and MA for 200 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

    st.subheader('Original Close Price and MA for 100 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

    st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, extra_data=True, extra_dataset=google_data['MA_for_250_days']))

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    # Prepare data for prediction
    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

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
    ploting_data = pd.DataFrame({
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    }, index=google_data.index[splitting_len + 100:])

    # Display prediction results
    st.subheader("Original values vs Predicted values")
    st.write(ploting_data)

    # Plot results
    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
    plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)
