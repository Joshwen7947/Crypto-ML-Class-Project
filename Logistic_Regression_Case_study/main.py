import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load data into DataFrame
df = pd.read_csv("archive/coin_Bitcoin.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Drop unnecessary columns
df.drop(['SNo', 'Name', 'Symbol', 'Volume', 'Marketcap'], axis=1, inplace=True)

# Filter only required columns
df = df[['Date', 'Close']]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], marker='o', linestyle='-')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature engineering: We'll use previous prices to predict the next price
# Let's create a new column 'Previous_Close' which contains the previous day's close price
df['Previous_Close'] = df['Close'].shift(1)

# Drop the first row since it will have NaN value for 'Previous_Close'
df = df.dropna()

# Define features (X) and labels (y)
X = df[['Previous_Close']]
y = df['Close']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy (using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the model
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual Prices')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted Prices')
plt.title('Linear Regression Model: Bitcoin Price Prediction')
plt.xlabel('Previous Day Close Price (USD)')
plt.ylabel('Next Day Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Generate future dates for prediction (up to 2025)
future_dates = pd.date_range(start=df['Date'].max(), end='2025-01-01', freq='D')

# Prepare features for future prediction
future_features = pd.DataFrame({'Previous_Close': df['Close'].iloc[-1]}, index=future_dates)
future_features_scaled = scaler.transform(future_features)

# Predict future prices
future_prices = model.predict(future_features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(df['Date'], df['Close'], color='black', label='Actual Prices')

# Plot the predicted prices for the training data
predicted_train_dates = df['Date'][:len(X_train)]
plt.plot(predicted_train_dates, model.predict(X_train_scaled), color='blue', linewidth=3, label='Predicted Prices (Training)')

# Plot the predicted prices for the future dates
plt.plot(future_dates, future_prices, color='green', linestyle='--', linewidth=2, label='Future Prediction')
plt.title('Linear Regression Model: Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
