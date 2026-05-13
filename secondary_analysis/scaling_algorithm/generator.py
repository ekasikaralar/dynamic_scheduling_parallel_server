#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

# Seed
np.random.seed(73)

high_dim_class_no = 100

arrivals_high_dim = pd.read_csv("lambda_scaled.csv", header=None).values.flatten()
theta_high_dim = pd.read_csv("theta_scaled.csv", header=None).values.flatten()

# Generate arrival cdf
arr_cdf_high_dim = np.cumsum(arrivals_high_dim / np.sum(arrivals_high_dim))
percentage_high_dim = arrivals_high_dim / np.sum(arrivals_high_dim)

# Generate cost parameters
random_numbers = 15 + (35 - 15) * np.random.rand(high_dim_class_no)
holding_cost_rate_high_dim = random_numbers
abandonment_cost_rate_high_dim = holding_cost_rate_high_dim / 15
cost_high_dim = holding_cost_rate_high_dim + abandonment_cost_rate_high_dim * theta_high_dim

np.savetxt("cost_scaled.csv", cost_high_dim, delimiter=",", fmt="%.2f")
np.savetxt("arr_cdf_scaled.csv", arr_cdf_high_dim, delimiter=",", fmt="%.4f")
np.savetxt("percentage_scaled.csv", percentage_high_dim, delimiter=",", fmt="%.3f")
np.savetxt("abandonment_cost_rate_scaled.csv", abandonment_cost_rate_high_dim, delimiter=",", fmt="%.2f")


