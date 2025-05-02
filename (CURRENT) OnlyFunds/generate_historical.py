from data.historical.historical import save_historical_data

# Fetch and process historical data for a few major pairs
save_historical_data("BTCUSDT", "5m", limit=1000)
save_historical_data("ETHUSDT", "5m", limit=1000)
save_historical_data("LTCUSDT", "5m", limit=1000)

print("✅ Historical data generation complete.")
