# Portfolio-Optimization-Model
A Python-based Monte Carlo simulation to optimize asset allocation using Modern Portfolio Theory
# Quantitative Portfolio Optimization Model

## Executive Summary
This project utilizes Python to construct an optimal investment portfolio based on Modern Portfolio Theory (MPT). By simulating 5,000 different asset allocations for a 5-stock basket, the model identifies the "Max Sharpe Ratio" portfolio, offering the best risk-adjusted returns.

## Key Features
* **Data Extraction:** Automated retrieval of 5-year historical data using Yahoo Finance API.
* **Financial Modeling:** Calculated annualized volatility, correlation matrices, and expected returns.
* **Optimization:** Implemented a Monte Carlo simulation to generate an Efficient Frontier.
* **Visualization:** Plotted risk vs. return metrics to visualize the optimal trade-off.

## Technologies Used
* Python 3.10
* Pandas, NumPy (Data Manipulation)
* Matplotlib (Visualization)
* yfinance (Market Data)

## Results
--- OPTIMAL PORTFOLIO (Max Sharpe) ---
Return: 21.28%
Volatility (Risk): 26.58%
Sharpe Ratio: 0.80
--- Optimal Weights---
AAPL: 65.68%, 
MSFT: 12.15%, 
GOOG: 0.70%, 
PG: 5.31%, 
KO: 16.16%, 
