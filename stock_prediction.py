import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
TICKER = 'TATASTEEL.NS'
PREDICTION_DAYS = 60      # Look-back window (time steps)
TEST_DATA_SIZE = 0.2     # Use the last 20% of data for testing
MODEL_FILENAME = 'TATASTEEL_Multivariate_LSTM_Model.h5'

# --- Data Fetch ---
def fetch_data():
    """
    Fetches historical stock data (OHLCV) and handles errors.
    """
    print(f"--- Attempting to download stock data for {TICKER} ---")
    try:
        # Fetch 5 years of daily data (includes Open, High, Low, Close, Volume)
        df = yf.download(TICKER, period='5y', interval='1d') 
        
        if df.empty:
            print(f"ERROR: Stock data for {TICKER} could not be downloaded. Returning empty data.")
            return pd.DataFrame() 
        
        # **Multivariate Input** - Select OHLC for prediction
        df = df[['Open', 'High', 'Low', 'Close']].copy() 
        print(f"Data successfully downloaded. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"An unexpected error occurred during data download: {e}")
        print("Note: If the YFRateLimitError persists, you must wait 1-2 hours or change your IP address.")
        return pd.DataFrame() 

# --- Prepare Data ---
def prepare_data(data_df):
    """Scales data and creates time-series sequences (X, y) for train/test sets."""
    
    # Scale all selected features (Open, High, Low, Close)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_df) # Scale the entire DataFrame
    
    # Train/Test Split - Determine split point
    training_data_len = int(len(scaled_data) * (1 - TEST_DATA_SIZE))
    train_data = scaled_data[0:training_data_len, :]
    
    # Function to create sequences (X=60 days, y=61st day's Close price)
    def create_sequences(data):
        x_data = []
        # The target is the 'Close' price, which is the last column (index 3)
        y_data = [] 
        for i in range(PREDICTION_DAYS, len(data)):
            # X: PREDICTION_DAYS time steps of all 4 features (OHLC)
            x_data.append(data[i-PREDICTION_DAYS:i, :]) 
            # y: The target is the 'Close' price (index 3) of the next day
            y_data.append(data[i, 3]) 
        return np.array(x_data), np.array(y_data)

    # Create training set sequences
    x_train, y_train = create_sequences(train_data)

    # Create testing set sequences
    # We include PREDICTION_DAYS data before the split point to form the first test sequence
    test_data = scaled_data[training_data_len - PREDICTION_DAYS:, :]
    x_test, y_test = create_sequences(test_data)
    
    print(f"\nx_train shape: {x_train.shape} | y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} | y_test shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test, scaler, training_data_len

# --- Build and Train Model ---
def build_and_train_model(x_train, y_train):
    print("--- Building Multivariate LSTM Model ---")
    model = Sequential()

    # Model Architecture (Multivariate input_shape = [PREDICTION_DAYS, features=4])
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) 
    model.add(Dropout(0.2)) 
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) 

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("--- Starting model training (50 epochs) ---")
    model.fit(x_train, y_train, epochs=50, batch_size=32) 
    print("--- Model training finished ---")

    # Save the model
    model.save(MODEL_FILENAME)
    print(f"\nModel saved as {MODEL_FILENAME}")
    return model

# --- Main Execution ---
if __name__ == "__main__":
    
    data_df = fetch_data()

    if data_df.empty:
        exit()
    
    # Prepare data (split, scale, sequence creation)
    x_train, y_train, x_test, y_test, scaler, training_data_len = prepare_data(data_df)
    
    # 3. Load or Train Model
    if os.path.exists(MODEL_FILENAME):
        print(f"\n--- Loading saved model from {MODEL_FILENAME} ---")
        model = load_model(MODEL_FILENAME)
    else:
        model = build_and_train_model(x_train, y_train)

    # -------------------------------------------------------------
    # 5. Model Evaluation (on Test Set) and Visualization
    # -------------------------------------------------------------

    print("\n--- Model Evaluation (Test Set) ---")
    
    # Make predictions on the test set
    predictions_scaled = model.predict(x_test)
    
    # Inverse transform the scaled predictions (using dummy array for 4 features)
    dummy_test_array = np.zeros((len(predictions_scaled), data_df.shape[1]))
    dummy_test_array[:, 3] = predictions_scaled.flatten()
    predicted_prices = scaler.inverse_transform(dummy_test_array)[:, 3] # Get only Close price

    # Get the actual unscaled prices for the test set
    test_start_index = training_data_len
    actual_prices = data_df['Close'][test_start_index:].values

    # Create a DataFrame for plotting and calculating metrics
    predictions_df = pd.DataFrame({
        'Actual': actual_prices[PREDICTION_DAYS:],
        'Predicted': predicted_prices
    }, index=data_df.index[test_start_index + PREDICTION_DAYS:])

    # Calculate Root Mean Squared Error (RMSE) for evaluation
    rmse = np.sqrt(np.mean(predictions_df['Predicted'].values - predictions_df['Actual'].values)**2)
    print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")

    # Plot the results
    plt.figure(figsize=(16, 8))
    plt.title(f'{TICKER} Stock Prediction (Test Set Performance)')
    plt.plot(predictions_df['Actual'], label='Actual Closing Price', color='blue')
    plt.plot(predictions_df['Predicted'], label='Predicted Closing Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.show()

    # -------------------------------------------------------------
    # 6. Predict the Next Day's Price (Future Prediction)
    # -------------------------------------------------------------

    print("\n--- Predicting Next Available Day's Closing Price ---")

    # Get the latest data (last 60 days of OHLC) from the original data
    latest_data = data_df.tail(PREDICTION_DAYS) 

    # Scale the latest data
    latest_data_scaled = scaler.transform(latest_data)

    # Prepare input for the model: reshape to [1, 60, 4]
    X_test_future = np.array(latest_data_scaled)
    X_test_future = np.reshape(X_test_future, (1, X_test_future.shape[0], X_test_future.shape[1]))

    # Make the prediction
    predicted_price_scaled = model.predict(X_test_future)

    # Inverse transform the scaled prediction
    dummy_future_array = np.zeros((len(predicted_price_scaled), data_df.shape[1]))
    dummy_future_array[:, 3] = predicted_price_scaled.flatten()
    predicted_price = scaler.inverse_transform(dummy_future_array)[:, 3]

    # Output the result
    print("---------------------------------------------")
    print(f"Predicted Closing Price for {TICKER}: ₹{predicted_price[0]:.2f}")
    print("---------------------------------------------")