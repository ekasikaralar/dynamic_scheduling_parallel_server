import numpy as np
import pandas as pd
import json

with open("config.json", "r") as f:
    config = json.load(f)

np.random.seed(73)

def generate_training_data(config):
    
    # Relevant data
    agents = pd.read_csv(config["agent_file"], header = None).values.T
    optimal_xi = pd.read_csv(config["xi_file"], header = None).values
    mu = pd.read_csv(config["mu_file"], header = None).values
    cost = pd.read_csv(config["cost_file"], header = None).values
    theta = pd.read_csv(config["theta_file"], header = None).values
    r = config["r"]
    
    # x_optimal 
    x_opt = ((optimal_xi * (agents / r)) * (mu > 0)).sum(axis=1)
    
    # State observations
    lower = -np.sqrt(r) * x_opt # The lowest in the prelimit is that there is 0 people
    upper = (config["state_upper_bound"] - r * x_opt) / np.sqrt(r) # The upper bound comes from the FCFS simulation    
    
    state = np.stack([np.random.uniform(l, u, config["num_obs"]) for l, u in zip(lower, upper)], axis=1)
    
    # Gradient observations
    grad_upper = cost / theta
    grad = np.stack([np.random.uniform(0, g, config["num_obs"]) for g in grad_upper], axis=1)

    return state, grad, x_opt

state, grad, x_opt = generate_training_data(config)

# Save to CSV
np.savetxt("state_observations.csv", state, delimiter=",")
np.savetxt("gradient_observations.csv", grad, delimiter=",")
