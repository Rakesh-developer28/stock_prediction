import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Configuration ---
TICKER = 'TATASTEEL.NS'
PREDICTION_DAYS = 60 # Number of days history to look back for prediction

# --- Data Fetch ---
def fetch_data():
    """
    Fetches historical stock data for the defined TICKER.
    Returns a pandas DataFrame with the 'Close' prices.
    """
    print(f"--- Attempting to download stock data for {TICKER} ---")
    try:
        # Fetch 5 years of daily data to ensure enough training samples
        df = yf.download(TICKER, period='5y', interval='1d') 
        
        if df.empty:
            print(f"ERROR: Stock data for {TICKER} could not be downloaded. Returning empty data.")
            return pd.DataFrame({'Close': []}) 
        
        # Use only the 'Close' price
        df = df[['Close']].copy() 
        print(f"Data successfully downloaded. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"An unexpected error occurred during data download: {e}")
        return pd.DataFrame({'Close': []}) 

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Fetch Data
    data_df = fetch_data()

    if data_df.empty or len(data_df) < PREDICTION_DAYS + 1: 
        print(f"Insufficient data ({len(data_df)} samples) to proceed with model training. Requires at least {PREDICTION_DAYS + 1} samples.")
        exit() 

    # 2. Data Preprocessing
    
    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_df['Close'].values.reshape(-1, 1))
    
    # Create the training data set (X_train and y_train)
    x_train = []
    y_train = []

    for x in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-PREDICTION_DAYS:x, 0])
        y_train.append(scaled_data[x, 0])
    
    # Convert to numpy arrays and reshape for LSTM: [samples, time_steps (60), features (1)]
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # 3. Build and Train the LSTM Model
    print("--- Building LSTM Model ---")
    model = Sequential()

    # Model Architecture
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)) 
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output layer

    # Compile and Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("--- Starting model training (25 epochs) ---")
    model.fit(x_train, y_train, epochs=25, batch_size=32) 
    print("--- Model training finished ---")

    # -------------------------------------------------------------
    # 6. Predict the Next Day's Price (Future Prediction)
    # -------------------------------------------------------------

    print("--- Predicting Tomorrow's Closing Price ---")

    # Get the last 60 days of closing price data (the input for the prediction)
    last_60_days = data_df['Close'].values[-PREDICTION_DAYS:]
    
    # Scale the last 60 days of prices using the fitted scaler
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    # Prepare the input for the model: reshape to [1, 60, 1]
    X_test_future = np.array(last_60_days_scaled)
    X_test_future = np.reshape(X_test_future, (1, X_test_future.shape[0], 1))

    # Make the prediction
    predicted_price_scaled = model.predict(X_test_future)

    # Inverse transform the scaled prediction to get the actual rupee price
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # Output the result
    print("---------------------------------------------")
    print(f"Predicted Closing Price for Tomorrow ({TICKER}): ₹{predicted_price[0][0]:.2f}")
    print("---------------------------------------------")

    # -------------------------------------------------------------
    # 5. Model Evaluation and Visualization (Training Performance Plot)
    # -------------------------------------------------------------

    print("--- Starting Model Evaluation and Visualization ---")

    # Make predictions on the training set
    predicted_prices_scaled = model.predict(x_train)

    # Inverse transform the scaled predictions to get actual prices
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
    
    # Get the actual prices for the training period
    actual_prices = data_df['Close'].values[PREDICTION_DAYS:]
    
    # Create a DataFrame for plotting
    predictions_df = pd.DataFrame({
        'Actual': actual_prices.flatten(),
        'Predicted': predicted_prices.flatten()
    }, index=data_df.index[PREDICTION_DAYS:])

    # Plot the results
    plt.figure(figsize=(16, 8))
    plt.title(f'{TICKER} Stock Price Prediction (Training Data Fit)')
    plt.plot(predictions_df['Actual'], label='Actual Closing Price', color='blue')
    plt.plot(predictions_df['Predicted'], label='Predicted Closing Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.show()

    print("--- Visualization complete ---")