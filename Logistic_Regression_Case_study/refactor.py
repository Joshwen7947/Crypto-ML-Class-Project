import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


###########################################
# Phase 1

# Load and preprocess the data
def load_data():
    df = pd.read_csv("archive/coin_Bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop(['SNo', 'Name', 'Symbol', 'Volume', 'Marketcap'], axis=1, inplace=True)
    df = df[['Date', 'Close']]
    return df

# Plot Bitcoin price over time
def plot_price_over_time(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], marker='o', linestyle='-')
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()





###########################################
# Phase 2

# Feature engineering: Creating previous close column
def create_previous_close(df):
    df['Previous_Close'] = df['Close'].shift(1)
    df = df.dropna()
    return df

# Split data into features and labels
def split_data(df):
    X = df[['Previous_Close']]
    y = df['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Train Linear Regression model
def train_linear_regression(X_train_scaled, y_train):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    return y_pred

# Plot model predictions
def plot_predictions(X_test, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', label='Actual Prices')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted Prices')
    plt.title('Linear Regression Model: Bitcoin Price Prediction')
    plt.xlabel('Previous Day Close Price (USD)')
    plt.ylabel('Next Day Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()






###########################################
# Phase 3

# Predict future prices
def predict_future_prices(model, scaler, df):
    future_dates = pd.date_range(start=df['Date'].max(), end='2025-01-01', freq='D')
    future_features = pd.DataFrame({'Previous_Close': df['Close'].iloc[-1]}, index=future_dates)
    future_features_scaled = scaler.transform(future_features)
    future_prices = model.predict(future_features_scaled)
    return future_dates, future_prices

# Plot future predictions
def plot_future_predictions(df, model, scaler, future_dates, future_prices, X_train_scaled):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Date'], df['Close'], color='black', label='Actual Prices')
    predicted_train_dates = df['Date'][:len(X_train_scaled)]
    plt.plot(predicted_train_dates, model.predict(X_train_scaled), color='blue', linewidth=3, label='Predicted Prices (Training)')
    plt.plot(future_dates, future_prices, color='green', linestyle='--', linewidth=2, label='Future Prediction')
    plt.title('Linear Regression Model: Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()





def main():
    #######################################
    # Phase 1
    # Load data
    df = load_data()

    # Plot Bitcoin price over time
    plot_price_over_time(df)
    
    
    #######################################
    # Phase 2
    # Feature engineering
    df = create_previous_close(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train Linear Regression model
    model = train_linear_regression(X_train_scaled, y_train)

    # Evaluate model
    y_pred = evaluate_model(model, X_test_scaled, y_test)

    # Plot model predictions
    plot_predictions(X_test, y_test, y_pred)


    #######################################
    # Phase 3
    # Predict future prices
    future_dates, future_prices = predict_future_prices(model, scaler, df)

    # Plot future predictions
    plot_future_predictions(df, model, scaler, future_dates, future_prices, X_train_scaled)

if __name__ == "__main__":
    main()
