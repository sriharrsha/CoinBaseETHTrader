import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Load the Ethereum price data
# df = pd.read_csv("eth_price.csv")
# df = df.dropna()

# API endpoint to get Ethereum price data
url = "https://api.coinbase.com/v2/prices/ETH-USD/historic?period=daily"

# Make a request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()

    # Create a pandas DataFrame
    df = pd.DataFrame(data['data'], columns=['time', 'price'])

    # Convert the date column to a datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Set the date column as the index
    df.set_index('time', inplace=True)

    # Convert the price column to a float type
    df['price'] = df['price'].astype(float)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df.values)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df_scaled, test_size=0.2, shuffle=False)

# Convert the data into a 3D format
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data to (samples, time-steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile and fit the model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Make predictions on the test data
inputs = df[len(df) - len(test_data) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot the predictions and the true values
plt.plot(test_data[:, 0], label='True Value')
plt.plot(predictions, label='Prediction')
plt.title('Ethereum Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

