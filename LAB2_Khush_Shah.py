#%% [markdown]
import pandas as pd
from pandas_datareader import data as web
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

#%%
# Override yfinance API
yf.pdr_override()

# Defining stocks and date range
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = '2013-01-01'
end_date = '2023-08-28'

# Retrieve stock data
stock_data = {}
for stock in stocks:
    print(f"Fetching data for {stock}")
    stock_data[stock] = web.get_data_yahoo(stock, start=start_date, end=end_date)

for stock, data in stock_data.items():
    print(f"\nHead for {stock}:")
    print(data.head())

#%%
#Line plot for the stocks
plt.figure(figsize=(16, 8))

def plot_stock_feature(feature):
    """
    The function plots the line plot for the stocks for the selected feature.
    :param feature: The feature to be plot.
    returns the plot.
    """
    plt.figure(figsize=(16, 8))

    for i, (stock, data) in enumerate(stock_data.items()):
        plt.subplot(3, 2, i+1)
        data[feature].plot(title=f'{feature} Prices of {stock}')
        plt.xlabel('Date')
        plt.ylabel(f'{feature} Price (USD)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

features = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
for feature in features:
    plot_stock_feature(feature)

#%%
def plot_stock_feature_histogram(feature):
    """
    The function plots the histogram plot for the stocks for the selected feature.
    :param feature: The feature to be plot.
    returns the plot.
    """
    plt.figure(figsize=(16, 8))

    for i, (stock, data) in enumerate(stock_data.items()):
        plt.subplot(3, 2, i+1)
        data[feature].plot(kind='hist', bins=50, title=f'{feature} Distribution of {stock}')
        plt.xlabel(f'{feature} Price (USD)')
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

for feature in features:
    plot_stock_feature_histogram(feature)


#%%
def calculate_covariance(x, y):
    """
    The function calculates the covariance of the data.
    :param x,y: The features between whom the covariance is to be calculated.
    returns the covariance.
    """
    n = len(x)
    return sum((x - np.mean(x)) * (y - np.mean(y))) / (n - 1)


# Function to calculate the covariance matrix for a given stock
def calculate_stock_covariance(stock_data, stock):
    """
    The function calculates the covariance matrix of the data.
    :param stock_data: The data loaded.
    :param stock: The stock with which the covariance needs to be calculated.
    returns the covariance matrix.
    """
    # Initialize an empty DataFrame for the covariance matrix
    cov_matrix = pd.DataFrame(index=features, columns=features)

    # Calculate covariance for each pair of features
    for i in features:
        for j in features:
            cov_matrix.loc[i, j] = calculate_covariance(stock_data[stock][i], stock_data[stock][j])
    
    return cov_matrix

# Calculating and printing covariance matrices for all stocks
for stock in stocks:
    cov_matrix = calculate_stock_covariance(stock_data, stock)
    print(f"\nCovariance Matrix for {stock}:")
    print(cov_matrix)


#%%
# Function to plot scatter matrix for a stock
def plot_scatter_matrix(stock, stock_data):
    """
    This function plots the scatter_matrix for the selected stock.
    :param stock_data: The data loaded.
    :param stock: The stock with which the covariance needs to be calculated.
    returns the scatter matrix.
    """
    scatter_data = stock_data[stock][['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
    scatter_matrix(scatter_data, alpha=0.5, figsize=(10, 10), diagonal='kde', hist_kwds={'bins': 50}, s=10)
    plt.suptitle(f'Scatter Matrix for {stock}')
    plt.show()

# Plotting scatter matrix for each stock
for stock in stocks:
    plot_scatter_matrix(stock, stock_data)

# %%
