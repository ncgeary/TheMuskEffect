#%%
import matplotlib.pyplot as plt
import yfinance as yf

# Get the data of the stock Tesla Stock (TSLA)
data = yf.download('TSLA', '2019-01-01', '2020-01-25')
# Import the plotting library
%matplotlib inline
# Plot the close price of the AAPL
data['Adj Close'].plot()
plt.show()


# %%
