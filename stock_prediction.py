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
        # **CORRECTION**: Changed period from '1d' to '5y' to get enough data.
        df = yf.download(TICKER, period='5y', interval='1d') 
        
        if df.empty:
            print(f"ERROR: Stock data for {TICKER} could not be downloaded. Returning empty data.")
            return pd.DataFrame({'Close': []}) 
        
        # We only need the 'Close' price for this prediction
        df = df[['Close']].copy() 
        print(f"Data successfully downloaded. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"An unexpected error occurred during data download: {e}")
        # Note: If rate-limited, wait 5-10 minutes before trying again.
        return pd.DataFrame({'Close': []}) 

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Fetch Data
    data_df = fetch_data()

    # Check if enough data was fetched (we need at least PREDICTION_DAYS + 1 for training)
    if data_df.empty or len(data_df) < PREDICTION_DAYS + 1: 
        print(f"Insufficient data ({len(data_df)} samples) to proceed with model training. Requires at least {PREDICTION_DAYS + 1} samples.")
        exit() 

    # 2. Data Preprocessing
    
    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_df['Close'].values.reshape(-1, 1))
    
    # Create the training data set
    x_train = []
    y_train = []

    for x in range(PREDICTION_DAYS, len(scaled_data)):
        # x_train contains the last 60 scaled prices
        x_train.append(scaled_data[x-PREDICTION_DAYS:x, 0])
        # y_train contains the 61st scaled price (the one to predict)
        y_train.append(scaled_data[x, 0])
    
    # Convert to numpy arrays and reshape for LSTM
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape: [samples, time_steps (60), features (1)]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # 3. Build and Train the LSTM Model
    print("--- Building LSTM Model ---")
    model = Sequential()

    # Layer 1: LSTM layer with 50 units, returns sequences for the next LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # 20% dropout

    # Layer 2: LSTM layer, does NOT return sequences (final LSTM layer)
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Output Layer: Dense layer for the single price prediction
    model.add(Dense(units=1)) 

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("--- Starting model training (25 epochs) ---")
    # Train the model
    model.fit(x_train, y_train, epochs=25, batch_size=32) 
    print("--- Model training finished ---")
    
    # NOTE: To continue this project, you would add sections here for:
    # 1. Loading Test Data
    # 2. Making Predictions
    # 3. Inverse Scaling the Predictions
    # 4. Plotting the results (Actual vs. Predicted)