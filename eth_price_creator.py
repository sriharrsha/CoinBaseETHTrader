import csv
import requests

# API endpoint to get Ethereum price data
url = "https://api.coinbase.com/v2/prices/ETH-USD/historic?period=daily"

# Make a request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()

    # Write the data to a CSV file
    with open('eth_price.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'close'])

        for item in data['data']:
            writer.writerow([item['time'], item['price']])

else:
    print("Failed to get Ethereum price data")
