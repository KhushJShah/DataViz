#Import Data
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from prettytable import PrettyTable

# Override yfinance API
yf.pdr_override()

# Defining stocks and date range
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = '2013-01-01'
end_date = '2023-08-28'

# Retrieve stock data
stock_data = {}
for stock in stocks:
    stock_data[stock] = pdr.get_data_yahoo(stock, start=start_date, end=end_date)

#Calculate statistics and add to PrettyTable
def calculate_stats(function, table_title):
    """
    The function 'calculate_stats' calculates the statistics of the loaded dataframe. The output is displayed in the required tabular format using the library prettytables.
    The field names have been added to the respective columns.

    :param function: The stastical function to be calculated.
    :param table_title: Title of the table.
    return the table printed on the console.
    """
    table = PrettyTable()
    table.field_names = ["Feature Name", "High($)", "Low($)", "Open($)", "Close($)", "Volume", "Adj Close($)"]
    table.title = table_title

    # Compute statistics for each stock
    for stock in stocks:
        stats = [stock]
        for feature_name in ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']:
            feature_data = stock_data[stock][feature_name]
            value = function(feature_data)
            stats.append(f"{value:.2f}")
        table.add_row(stats)

    # Compute overall statistics for each feature across all stocks
    for feature_name in ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']:
        feature_data_combined = pd.concat([stock_data[stock][feature_name] for stock in stocks])
        max_val = function(feature_data_combined)
        min_val = function(feature_data_combined)
        max_stock = max(stock_data, key=lambda x: function(stock_data[x][feature_name]))
        min_stock = min(stock_data, key=lambda x: function(stock_data[x][feature_name]))

        # Check for the correct column title
        if feature_name == 'Volume':
            feature_title = feature_name
        else:
            feature_title = f"{feature_name}($)"

        max_index = table.field_names.index(feature_title)
        min_index = table.field_names.index(feature_title)
        max_company_index = table.field_names.index(feature_title)
        min_company_index = table.field_names.index(feature_title)

        # Adding rows for the Maximum and Minimum values and corresponding companies
        table.add_row([f"Maximum {feature_name}"] + [""] * (max_index - 1) + [f"{max_val:.2f}"] + [""] * (len(table.field_names) - max_index - 1))
        table.add_row([f"Minimum {feature_name}"] + [""] * (min_index - 1) + [f"{min_val:.2f}"] + [""] * (len(table.field_names) - min_index - 1))
        table.add_row([f"Company with Max {feature_name}"] + [""] * (max_company_index - 1) + [max_stock] + [""] * (len(table.field_names) - max_company_index - 1))
        table.add_row([f"Company with Min {feature_name}"] + [""] * (min_company_index - 1) + [min_stock] + [""] * (len(table.field_names) - min_company_index - 1))

    print(table)


# Calculate mean values
calculate_stats(pd.DataFrame.mean, "Mean Value Comparison")

# Calculate variance
calculate_stats(pd.DataFrame.var, "Variance Comparison")

# Calculate standard deviation
calculate_stats(pd.DataFrame.std, "Standard Deviation Value Comparison")

# Calculate median
calculate_stats(pd.DataFrame.median, "Median Value Comparison")

# Function to display correlation matrix
def display_correlation(stock):
    """
    The function is utilized to calculate the correlation between the required attributes in the dataframe.
    :param stock: company
    returns the correlation matrix
    """
    print(f"Correlation matrix for {stock}:")
    correlation_matrix = stock_data[stock].corr()
    print(correlation_matrix.round(2))

# Display correlation matrix for each company
for stock in stocks:
    display_correlation(stock)

std_table = PrettyTable()
std_table.field_names = ["Company", "Std Dev"]
std_values = {stock: stock_data[stock]['Close'].std() for stock in stocks}
recommended_stock = min(std_values, key=std_values.get)
for stock, std in std_values.items():
    std_table.add_row([stock, f"{std:.2f}"])

print("Recommendation for Safe Investment (Lower Std Dev):")
print(std_table)
print(f"Based on the data, the recommended stock for a safe investment is {recommended_stock} due to its lowest standard deviation in closing prices.")
