import csv
import numpy as np
import pandas as pd
from itertools import product
from gurobipy import Model, quicksum, GRB, LinExpr
import json
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from concurrent.futures import ThreadPoolExecutor, as_completed

class PolicyIteration:
    def __init__(self, gamma, c, lambda_, theta, mu, N, state_limits):

        self.cust_dim, self.server_dim = mu.shape  # number of customer classes and service stations

        self.K_set = range(self.cust_dim)  # customer class indices
        self.J_set = range(self.server_dim)  # server indices

        # K[j] = list of customer classes k such that mu[k][j] > 0
        self.K = {j: [k for k in self.K_set if mu[k, j] > 0] for j in self.J_set}

        # J[k] = list of service stations j such that mu[k][j] > 0
        self.J = {k: [j for j in self.J_set if mu[k, j] > 0] for k in self.K_set}
        
        self.gamma = gamma
        self.c = np.array(c) # hourly cost rates
        self.lambda_ = np.array(lambda_) # hourly arrival rates
        self.theta = np.array(theta) # hourly abandonment rates
        self.mu = np.array(mu) # hourly service rates
        self.N = np.array(N) # number of agents in each service station
        self.state_limits = state_limits # boundaries of the state space (K-dimensional)
        self.shape = tuple(lim + 1 for lim in state_limits)
        self.V = np.zeros(self.shape)
        self.policy = {}

        for idx in np.ndindex(self.shape):
            self.policy[idx] = {(k, j): 0.0 for k in self.K_set for j in self.J[k]}

    def delta_minus(self, x, k):
        """Computes Δ⁻ₖ(x) = V(x) - V(x - e_k) with boundary handling."""
        if x[k] == 0:
            return 0.0
        x_prev = list(x)
        x_prev[k] -= 1
        return self.V[x] - self.V[tuple(x_prev)]

    def policy_evaluation(self):
        # Build state index mapping
        all_states = list(np.ndindex(self.shape))
        self.state_index = {idx: i for i, idx in enumerate(all_states)}
        self.index_state = {i: idx for i, idx in enumerate(all_states)}
        num_states = len(self.state_index)

        A = lil_matrix((num_states, num_states))
        b = np.zeros(num_states)

        for i, x in self.index_state.items():
            state = np.array(x)
            psi = self.policy[x]
            row = A.rows[i]
            data = A.data[i]
            
            g_x = np.dot(self.c, state - np.array([sum(psi[k, j] for j in self.J[k]) for k in self.K_set]))

            # Add: x -> x + e_k
            for k in self.K_set:
                if state[k] < self.state_limits[k]:
                    x_plus = list(state)
                    x_plus[k] += 1
                    j = self.state_index[tuple(x_plus)]
                    row.append(j)
                    data.append(-self.lambda_[k])

            # Add: x -> x - e_k
            for k in self.K_set:
                if state[k] > 0:
                    x_minus = list(state)
                    x_minus[k] -= 1
                    j = self.state_index[tuple(x_minus)]

                    service = sum(self.mu[k, j] * psi.get((k, j), 0.0) for j in self.J[k])
                    waiting = state[k] - sum(psi.get((k, j), 0.0) for j in self.J[k])
                    abandonment = self.theta[k] * waiting
                    rate = service + abandonment

                    if rate > 0:
                        row.append(j)
                        data.append(-rate)

            # Diagonal entry
            total_out_rate = 0.0
            for k in self.K_set:
                service = sum(self.mu[k, j] * psi.get((k, j), 0.0) for j in self.J[k])
                waiting = max(state[k] - sum(psi.get((k, j), 0.0) for j in self.J[k]), 0.0)
                abandonment = self.theta[k] * waiting

                if state[k] < self.state_limits[k]:
                    total_out_rate += self.lambda_[k] + service + abandonment
                else:
                    total_out_rate += service + abandonment

            row.append(i)
            data.append(self.gamma + total_out_rate)

            # RHS
            b[i] = g_x

        # Solve the system: (γ I - Q^ψ) V = g
        V_flat = spsolve(A.tocsr(), b)
        self.V = V_flat.reshape(self.shape)
        
        for state in [(0, 0), (0, 10), (10, 0), (20, 20), (100, 100), (200, 200), (300,300), (499,499)]:
            print(f"V{state} = {self.V[state]}")

    def policy_improvement(self):
        stable = True
        for x in np.ndindex(self.shape):
            delta_minus = [self.delta_minus(x, k) for k in self.K_set]
            psi_new = self.solve_lp_for_policy(x, delta_minus)
            if psi_new != self.policy[x]:
                self.policy[x] = psi_new
                stable = False
        return stable
    
    def solve_lp_for_policy(self, x, delta_minus):
        
        model = Model("sup_lp")
        model.setParam("OutputFlag", 0)

        psi = {}
        for k in self.K_set:
            for j in self.J[k]:
                psi[k, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"psi_{k}_{j}")

        # Constraints
        for k in self.K_set:
            model.addConstr(quicksum(psi[k, j] for j in self.J[k]) <= x[k], name=f"class_{k}")

        for j in self.J_set:
            model.addConstr(quicksum(psi[k, j] for k in self.K[j]) <= self.N[j], name=f"server_{j}")
        
        obj_expr = quicksum(
            (self.c[k] + (self.mu[k, j] - self.theta[k]) * delta_minus[k]) * psi[k, j]
            for k in self.K_set for j in self.J[k]
        )

        model.setObjective(obj_expr, GRB.MAXIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"Warning: policy LP not optimal at {x}")
            return self.policy[x]

        return {(k, j): psi[k, j].X for k in self.K_set for j in self.J[k]}


    def solve(self):
        for iteration in range(1000):
            self.policy_evaluation()
            stable = self.policy_improvement()
            print(f"Iteration {iteration}: Policy stable = {stable}")
            if stable:
                break
        return self.V
    
def load_array(path):
    df = pd.read_csv(path, header=None)
    return df.values

def main(config_path):
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    gamma = config['gamma'] # discount factor
    c = load_array(config['c']).flatten() # hourly cost rates
    lambda_ = load_array(config['lambda']).flatten() # hourly arrival rates
    theta = load_array(config['theta']).flatten() # hourly abandonment rates
    mu = load_array(config['mu']) # hourly service rates
    state_limits = load_array(config['state_limits']).flatten()  
    N = load_array(config['agents']).flatten()

    pi = PolicyIteration(gamma=gamma, c=c, lambda_=lambda_, theta=theta, mu=mu, N=N, state_limits=state_limits)

    final_V = pi.solve()

    # Save to .npy file
    np.save("value_function.npy", final_V)
    print("Saved final value function to 'value_function.npy'")

    print("Final Value Function:")
    print(final_V)
    
    with open("policy.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["state", "k", "j", "psi_kj"])
        for state, psi_dict in pi.policy.items():
            for (k, j), val in psi_dict.items():
                writer.writerow([state, k, j, val])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run value iteration with JSON configuration.")
    parser.add_argument("config", type=str, help="Path to configuration JSON file.")
    args = parser.parse_args()
    main(args.config)