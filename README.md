# DayTrade
Binance automated day trading bot

The bot uses only kline data. It extracts data from a prticular crypto pair and additionally uses BTC price. The raw kline data is processed to the dataset. Bot makes predictions based on TCN-like neural network. Obtained nn can be tested through using the simulation algorithm. Fimally the algorithm for autotrading is also present. Therefore, the process goes through the algorithms in the following order grabber1.py->data_processing.py->TCN.py->simulation.py->AutoTrade.py 
