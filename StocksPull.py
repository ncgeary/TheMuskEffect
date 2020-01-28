#%%
import matplotlib.pyplot as plt
import yfinance as yf

#%%
# Get the data of the stock Tesla Stock (TSLA)
data = yf.download("TSLA", start="2020-1-1", end="2020-01-25")

#%%
data.head()
