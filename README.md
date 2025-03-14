# intro_stats_learning_python
Introduction to Statistical Learning

import numpy as np
import pandas as pd
import statsmodels.api as sm

# 1. Preprocessing: ensure the DataFrame is indexed by date and has no missing values.
df = df.copy()  # work on a copy to avoid modifying original data
df = df.sort_index()               # sort by date if not already sorted
df = df.dropna(how='any')          # drop any rows with NaN values (or use fillna as appropriate)

# 2. Calculate 5-day rolling returns for each series.
returns_5d = df.pct_change(periods=5)  # percentage change over 5 days
returns_5d = returns_5d.dropna()       # drop the initial rows that don't have 5-day data

# 3. Set parameters for rolling regression.
window = 252 * 3   # approximately 3 years of daily data (252 trading days per year -> ~756 days)
decay = 0.94       # exponential decay factor for weighting (adjust this as needed)

pure_residuals = []   # to store the residual for each rolling window end
residual_dates = []   # to store corresponding dates (index) for residuals

# 4. Rolling Exponentially Weighted Regression with Newey-West adjusted errors.
for t in range(window, len(returns_5d)):
    # Define the rolling window slice (inclusive of index t).
    window_data = returns_5d.iloc[t-window : t+1]  
    y = window_data['Commodity']                   # dependent variable (commodity 5-day return)
    X = window_data[['Equity', 'Interest']]        # independent variables (equity & interest 5-day returns)
    X = sm.add_constant(X)                         # add intercept term

    # Create exponential weights for this window (newest = 1.0, oldest ≈ decay^window).
    n = len(window_data)
    weights = np.array([decay ** (n - 1 - j) for j in range(n)])
    
    # Fit Weighted Least Squares regression with HAC (Newey-West) covariance for standard errors.
    model = sm.WLS(y, X, weights=weights).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    
    # Extract the latest residual (the difference between actual and predicted commodity return at time t).
    pure_resid_t = model.resid.iloc[-1]
    pure_residuals.append(pure_resid_t)
    residual_dates.append(window_data.index[-1])

# 5. Create a time series (pandas Series) of the pure commodities risk factor.
pure_commodities_factor = pd.Series(pure_residuals, index=residual_dates)
