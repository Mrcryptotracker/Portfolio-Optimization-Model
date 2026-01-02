
# ==========================================
# PROJECT: Portfolio Optimization (Mean-Variance)
# AUTHOR: Amin Asadi
# DESCRIPTION: Monte Carlo Simulation to find the Efficient Frontier
# ==========================================

# --- STEP 1: LIBRARIES & SETUP ---
# Install yfinance if running in Google Colab (remove the '!' if running locally)
!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 2: DATA EXTRACTION ---
# Define the basket of 5 stocks (Diversified: Tech, Consumer Goods, Defensive)
tickers = ['AAPL', 'MSFT', 'GOOG', 'PG', 'KO']

print("Downloading data...")
# Download 5 years of historical data
# auto_adjust=False ensures we get the specific 'Adj Close' column if available
raw_data = yf.download(tickers, start="2020-01-01", end="2025-01-01", auto_adjust=False)

# CLEANING: Handle yfinance's variable column formats
# We want the "Adjusted Close" price which accounts for dividends and splits.
try:
    # Try to access 'Adj Close' directly (common in older versions)
    data = raw_data['Adj Close']
except KeyError:
    try:
        # Try to access 'Adj Close' as a MultiIndex level (common in newer versions)
        data = raw_data.xs('Adj Close', level=0, axis=1)
    except KeyError:
        # Fallback to 'Close' if Adjusted is unavailable
        data = raw_data['Close']

# Drop any missing values to prevent calculation errors
data = data.dropna()

print("Data successfully loaded.")
print(data.head()) # Preview the first 5 rows

# --- STEP 3: FINANCIAL CALCULATIONS ---
# Calculate Daily Log Returns
# Log returns are preferred in mathematical finance over simple percentage returns
daily_returns = np.log(data / data.shift(1))
daily_returns = daily_returns.dropna() # Drop the first row (NaN)

# Calculate the covariance matrix (measure of how stocks move together)
# We multiply by 252 to annualize it (252 trading days in a year)
cov_matrix = daily_returns.cov() * 252

print("\nCorrelation Matrix (Diversification Check):")
print(daily_returns.corr())

# --- STEP 4: MONTE CARLO SIMULATION ---
print("\nRunning Monte Carlo Simulation (5,000 Iterations)...")

# Arrays to store the results of the simulation
portfolio_returns = []
portfolio_volatility = []
sharpe_ratios = []
portfolio_weights = []

num_assets = len(tickers)
num_portfolios = 5000

# Set seed for reproducibility (so you get the same graph every time)
np.random.seed(42)

for _ in range(num_portfolios):
    # 1. Generate random weights
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Normalize so weights sum to 1 (100%)
    portfolio_weights.append(weights)

    # 2. Calculate Expected Annual Return
    # Formula: Sum(Weight * Average Daily Return) * 252
    ret = np.sum(weights * daily_returns.mean()) * 252
    portfolio_returns.append(ret)

    # 3. Calculate Expected Annual Volatility (Risk)
    # Formula: Sqrt(Weights_Transposed * Covariance_Matrix * Weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_volatility.append(vol)

    # 4. Calculate Sharpe Ratio (Return / Risk)
    # We assume Risk-Free Rate is effectively 0% for simplicity here,
    # or you can subtract roughly 0.04 (4%) if you want to be precise.
    sharpe = ret / vol
    sharpe_ratios.append(sharpe)

# Create a DataFrame to hold all simulation data
portfolios = pd.DataFrame({
    'Return': portfolio_returns,
    'Volatility': portfolio_volatility,
    'Sharpe': sharpe_ratios
})

# --- STEP 5: OPTIMIZATION RESULTS ---
# Find the portfolio with the Max Sharpe Ratio (The "Best" Risk-Adjusted Return)
max_sharpe_idx = portfolios['Sharpe'].idxmax()
max_sharpe_port = portfolios.iloc[max_sharpe_idx]
optimal_weights = portfolio_weights[max_sharpe_idx]

print("\n--- OPTIMAL PORTFOLIO (Max Sharpe) ---")
print(f"Return: {max_sharpe_port['Return']:.2%}")
print(f"Volatility (Risk): {max_sharpe_port['Volatility']:.2%}")
print(f"Sharpe Ratio: {max_sharpe_port['Sharpe']:.2f}")
print("\nOptimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# --- STEP 6: VISUALIZATION ---
plt.figure(figsize=(10, 6))
# Scatter plot of all portfolios
plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe'], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')

# Highlight the Optimal Portfolio with a Red Star
plt.scatter(max_sharpe_port['Volatility'], max_sharpe_port['Return'], marker='*', color='r', s=500, label='Max Sharpe Ratio')

# Labels
plt.title('Efficient Frontier: Monte Carlo Simulation')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Expected Return')
plt.legend(labelspacing=0.8)

plt.show()
