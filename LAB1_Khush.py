import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(6401)
#Part 1
# Function to calculate statistics
def generate_random_variable(mean_x, variance_x, n):
    """
    Calculates the mean and variance of the number.
    :param mean_x: mean of the array
    :param variance_x: variance of the array
    :param n: Number of observations
    :return: array, mean, variance
    """
    x = np.random.normal(mean_x, np.sqrt(variance_x), n)

    mean_x = np.mean(x)
    variance_x = np.var(x)

    return x, mean_x, variance_x

# Function to calculate Pearson's correlation coefficient
def calculate_pearson_coefficient(x, y, mean_x, mean_y):
    """
    Calculate the Pearson correlation coefficient for two arrays using pre-calculated means.

    :param x: NumPy array of numerical values for variable x
    :param y: NumPy array of numerical values for variable y
    :param mean_x: pre-calculated mean of x
    :param mean_y: pre-calculated mean of y
    :return: Pearson correlation coefficient
    """
    # Ensures that x and y have the same size
    if len(x) != len(y):
        return "Error: x and y must have the same number of elements."
    
    # Subtract means
    x_diff = x - mean_x
    y_diff = y - mean_y
    
    # Numerator: sum of product of differences
    numerator = np.sum(x_diff * y_diff)
    
    # Denominator: product of square root of sum of squares of differences
    denominator = np.sqrt(np.sum(x_diff**2) * np.sum(y_diff**2))
    
    # Prevent division by zero
    if denominator == 0:
        return "Error: Division by zero in the calculation of Pearson's coefficient."
    
    # Pearson correlation coefficient
    r = numerator / denominator
    return r

# Function to plot line and histogram plots
def plot_data(x, y):
    # Line plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, label='Random Variable X', color='blue')
    plt.plot(y, label='Random Variable Y', color='orange')
    plt.title('Line Plot of Random Variables X and Y')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)  
    plt.show()

    # Histogram plot
    plt.figure(figsize=(10, 5))
    plt.hist(x, alpha=0.5, bins=30, label='Random Variable X', color='blue')
    plt.hist(y, alpha=0.5, bins=30, label='Random Variable Y', color='orange')
    plt.title('Histogram of Random Variables X and Y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)  
    plt.show()



mean_x = float(input("Enter the mean for variable x: "))
variance_x = float(input("Enter the variance for variable x: "))
n = int(input("Enter the number of observations: "))
x, mean_x, variance_x = generate_random_variable(mean_x, variance_x, n)
mean_y = float(input("Enter the mean for variable y: "))
variance_y = float(input("Enter the variance for variable y: "))
y, mean_y, variance_y = generate_random_variable(mean_y, variance_y, n)

# Display statistics
pearson_coefficient = calculate_pearson_coefficient(x, y, mean_x, mean_y)
if isinstance(pearson_coefficient, str):
    print(pearson_coefficient)  
else:
    # Display statistics
    stats_message = (
        f"The sample mean of random variable x is: {mean_x:.2f}\n"
        f"The sample mean of random variable y is: {mean_y:.2f}\n"
        f"The sample variance of random variable x is: {variance_x:.2f}\n"
        f"The sample variance of random variable y is: {variance_y:.2f}\n"
        f"The sample Pearson’s correlation coefficient between x & y is: {pearson_coefficient:.2f}"
    )
    print(stats_message)

# Plotting the data
plot_data(x,y)


#Part 2
df = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
mean_sales = df['Sales'].mean()
mean_adbudget = df['AdBudget'].mean()
mean_gdp = df['GDP'].mean()

pearson_sales_adbudget = calculate_pearson_coefficient(df['Sales'], df['AdBudget'], mean_sales, mean_adbudget)
pearson_sales_gdp = calculate_pearson_coefficient(df['Sales'], df['GDP'], mean_sales, mean_gdp)
pearson_adbudget_gdp = calculate_pearson_coefficient(df['AdBudget'], df['GDP'], mean_adbudget, mean_gdp)

# Printing the results
print(f"The sample Pearson’s correlation coefficient between Sales & AdBudget is: {pearson_sales_adbudget:.2f}")
print(f"The sample Pearson’s correlation coefficient between Sales & GDP is: {pearson_sales_gdp:.2f}")
print(f"The sample Pearson’s correlation coefficient between AdBudget & GDP is: {pearson_adbudget_gdp:.2f}")

#Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['Sales'], label='Sales', color='blue')
plt.plot(df['AdBudget'], label='AdBudget', color='orange')
plt.plot(df['GDP'], label='GDP', color='green')
plt.title('Line Plot of Sales, AdBudget, and GDP over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['Sales'], bins=30, alpha=0.5, label='Sales', color='blue')
plt.hist(df['AdBudget'], bins=30, alpha=0.5, label='AdBudget', color='orange')
plt.hist(df['GDP'], bins=30, alpha=0.5, label='GDP', color='green')
plt.title('Histogram of Sales, AdBudget, and GDP')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()