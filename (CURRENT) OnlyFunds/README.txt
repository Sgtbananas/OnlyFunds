# CryptoTrader README

CryptoTrader is a sophisticated cryptocurrency trading bot designed to help automate trading strategies, manage risk, and optimize returns across multiple trading pairs. This bot utilizes technical analysis and machine learning for real-time decision-making and trading executions.

## Features

- **Automated Trading**: Execute trades based on a variety of trading signals and strategies.
- **Multi-Pair Trading**: Supports multiple cryptocurrency pairs (e.g., BTC/USDT, ETH/USDT, etc.).
- **Dry Run Mode**: Test your trading strategies in a safe environment before executing live trades.
- **Real-Time Data**: Fetches live market data using APIs for accurate decision-making.
- **Risk Management**: Includes position sizing, stop loss, and take profit configurations.
- **AI Tuning**: Optimize trading strategies using machine learning for predictive modeling.
- **Backtesting**: Run your strategies through historical data to assess performance before going live.
- **Dashboards**: Real-time visual dashboard for tracking trades, profits, and other metrics.

## Requirements

This project requires the following Python packages:

- `pandas`
- `requests`
- `ta`
- `sklearn`
- `streamlit`
- `python-dotenv`
- `ccxt`
- `matplotlib`

### To install the dependencies:


1. Create a .env file
Copy the .env.example to .env and fill in your API keys and other sensitive data.
COINEX_API_KEY=your_coinex_api_key
COINEX_SECRET_KEY=your_coinex_secret_key
3. Configure Trading Pairs
You can specify which trading pairs you want the bot to use. Modify the TRADING_PAIRS variable in core_data.py.

4. Running the Bot
You can run the bot in Dry Run mode (simulation) or Live Mode (actual trading). The default is Dry Run.

To start the bot:

bash
Copy
Edit
python main.py
This will run the bot in Dry Run mode. To switch to live trading, modify the configuration in the .env file to set the dry_run flag to False.

Usage
Dry Run Mode
Dry Run mode simulates trades based on real-time market data, but no actual funds are used. This is useful for testing strategies.

Live Trading
To execute real trades, set the DRY_RUN variable in .env to False. Ensure your API keys have the appropriate permissions to execute trades.

Configuration
Trade Pairs: Modify TRADING_PAIRS in core_data.py to use different trading pairs.

Risk Management: Configure stop-loss, take-profit, and position sizing in core_signals.py.

Backtesting
You can backtest strategies using historical data. To backtest:

Set BACKTESTING_MODE = True in the .env file.

Specify the desired backtest parameters in main.py.

Run the bot and observe the results of the strategy using historical data.

AI Autotuning
The bot features an AI-based autotuning mechanism that can optimize parameters for your trading strategies. This is useful for identifying the best settings for maximum profitability.

To enable autotuning, simply set AUTOTUNE = True in the .env file.

Dashboard
The bot includes a real-time dashboard that displays trading performance, active positions, and other statistics. The dashboard is built using Streamlit. To launch the dashboard:

bash
Copy
Edit
streamlit run dashboard.py
The dashboard will open in your browser, displaying metrics like:

Active positions

Profit/Loss statistics

Trading signals

Risk management parameters