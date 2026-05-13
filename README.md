# Dynamic Scheduling of a Parallel-Server Queueing System: A Computational Method for High-Dimensional Problems

**Authors:** Baris Ata, Ebru Kasikaralar

**Paper:** [arXiv:2605.09799](https://arxiv.org/abs/2605.09799)

This repository contains the replication package for the paper. The code implements a neural-network-based approach for computing near-optimal scheduling policies in multi-class, multi-server queueing systems operating in the Halfin-Whitt (many-server) diffusion regime.

---

## Overview

The method solves a discounted infinite-horizon stochastic control problem by reformulating the Hamilton-Jacobi-Bellman (HJB) equation as a backward stochastic differential equation (BSDE) and approximating its solution using deep neural networks. The approach is based on Han et al. (2018) and adapted for the parallel-server control problem.

The loss function is built around the auxiliary function **F(x, v)** (Eq. 46 in the paper), which decomposes as:

> F(x, v) = **D̂(x) · v** − c · x + **Ĥ(x, v)**

where:
- **D̂** (code: `drift_term`) approximates the reference-policy term of F — the drift of the controlled diffusion under a static allocation policy ψ̃
- **Ĥ** (code: `h_term`) approximates the supremum term of F — the optimal value of a parametric LP over the feasible action set

Both networks are trained **offline** before the main solver runs. They are loaded as fixed modules into the main BSDE solver (Algorithm 2 in the paper), which trains:
- **V^ω** (code: `Y_Net`) — neural network approximating the value function V
- **G^ν** (code: `Z_Net`) — neural network approximating the gradient ∇V

---

## Repository Structure

```
.
├── data/
│   ├── arrival_rates.ipynb            # Estimates hourly arrival rates per customer class
│   ├── agents_service_rates.ipynb     # Estimates staffing levels and service rates per station
│   ├── abandonment_times.ipynb        # Estimates customer abandonment (patience) rates
│   └── JulyOct2002/                   # Raw call center records (US Bank dataset, July–Oct 2002)
│       └── D<DDMMYYYY>_<type>.csv     # agent_events, agent_records, cust_subcalls
│
├── neural_network_solver/
│   ├── offline_approximation/
│   │   ├── h_term/
│   │   │   ├── input_data/
│   │   │   │   ├── h_net_input_data.py        # Generates state and gradient observations
│   │   │   │   └── configs/<case>/
│   │   │   │       ├── config.json            # Data generation parameters
│   │   │   │       └── data/                  # System parameters (μ, θ, cost, ξ, agents)
│   │   │   ├── output_data/
│   │   │   │   ├── h_term_offline.jl          # Julia LP solver: computes H-term training targets
│   │   │   │   └── configs/<case>/
│   │   │   │       ├── config.json            # LP solver parameters
│   │   │   │       ├── data/                  # State/gradient observations + system parameters
│   │   │   │       └── h_term_obj_value*.csv  # Computed training targets (LP optimal values)
│   │   │   └── nn_training/
│   │   │       ├── H_Net.py                   # FeedForwardNet model + training loop
│   │   │       ├── main.py                    # Entry point
│   │   │       └── config_hterm_elu_4hidden_5000epochs.json
│   │   │
│   │   └── drift_term/
│   │       ├── training_data/<case>/
│   │       │   ├── config_{cmu,cmu_theta,mu}.json   # Per-policy configs
│   │       │   ├── state_observations.csv           # Sampled state inputs
│   │       │   ├── drift_term_output_*.csv          # Computed drift targets
│   │       │   └── <system_params>.csv              # μ, θ, cost, ξ, agents
│   │       ├── output_data/
│   │       │   └── drift_term_offline.jl            # Julia LP solver: computes drift training targets
│   │       └── nn_training/
│   │           ├── DriftNet.py                      # FeedForwardNet model + training loop
│   │           ├── main.py                          # Entry point
│   │           └── config_drift_leaky_relu_4hidden_1000epochs_{cmu,cmu_theta,mu}.json
│   │
│   └── solver/
│       ├── equation.py        # HJB equation: forward SDE, generator f(x,z)
│       ├── solver.py          # BSDE solver, loss function, training loop
│       ├── Net.py             # Y_Net (value function V) and Z_Net (gradient ∇V)
│       ├── DriftNet.py        # Drift network (loaded as fixed module)
│       ├── H_Net.py           # H-function network (loaded as fixed module)
│       ├── main.py            # Entry point
│       └── configs/<case>/
│           ├── config.json    # Equation + network hyperparameters (includes softplus_output flag)
│           └── data/          # λ, θ, cost, ζ parameter files for the BSDE
│
└── secondary_analysis/
    ├── policy_iteration/
    │   ├── policy_iteration.py          # Exact policy iteration via LP (Gurobi), 2D cases
    │   └── configs/{first_2d,second_2d}/
    │       ├── config.json
    │       └── <system_params>.csv
    ├── static_allocation_problem/
    │   ├── static_allocation_model.jl   # Solves the static staffing problem (SPP) in Julia
    │   └── configs/{main,first_2d,second_2d,variant}/
    │       └── config.json + <system_params>.csv
    ├── scaling_algorithm/
    │   ├── scale_system.py              # Scales system topology and rates to L classes / I stations
    │   ├── generator.py                 # Generates cost and arrival CDF parameters for scaled instances
    │   └── config/config.json
    └── simulation/
        ├── neural_network_simulation/   # C++: simulates the trained NN policy
        │   ├── test.cpp / test.h
        │   ├── CMakeLists.txt
        │   └── configs/{main,first_2d,second_2d,variant,100dim}/
        │       ├── config.json
        │       └── data/
        ├── benchmarks_simulation/       # C++: simulates static priority rules (cμ, cμ/θ, μ, gcμ)
        │   ├── test.cpp / test.h
        │   ├── CMakeLists.txt
        │   └── configs/{main,first_2d,second_2d,variant,100dim}/
        │       ├── config_{policy}.json
        │       └── data/
        └── mdp_simulation/              # C++: simulates the MDP-optimal policy (2D cases only)
            ├── test.cpp / test.h
            ├── CMakeLists.txt
            └── configs/{first_2dim,second_2dim}/
                ├── config.json
                └── data/
```

**Experimental cases:**

| Case | Classes | Stations | Description |
|---|---|---|---|
| `first_2d` | 2 | 2 | First 2D problem |
| `second_2d` | 2 | 2 | Second 2D problem |
| `main` | 13 | 9 | Main problem calibrated to call center data |
| `variant` | 13 | 9 | Variant of the main problem |
| `100dim` | 100 | 70 | High-dimensional scaled instance |

> **Note on 100dim observation files:** The state and gradient observation CSVs for the 100-dimensional case are 2.5 GB each and exceed GitHub's per-file limit. They are provided as a single zip archive via Git LFS:
> `neural_network_solver/offline_approximation/100dim_large_observations.zip`
>
> After downloading, extract and place the files as follows:
> - `state_observations.csv` → `drift_term/training_data/100dim/` **and** `h_term/output_data/configs/100dim/data/` (both directories use the same file)
> - `gradient_observations.csv` → `h_term/output_data/configs/100dim/data/`

---

## Data

The `data/` folder contains call center records from the **US Bank Call Center dataset**, provided by the Technion Service Enterprise Engineering Lab (SEELab). The raw data covers July–October 2002 with three file types per day:

| File suffix | Contents |
|---|---|
| `_agent_events.csv` | Agent-level event log |
| `_agent_records.csv` | Agent shift records |
| `_cust_subcalls.csv` | Customer call records |

Three notebooks estimate the system parameters used throughout the paper:

| Notebook | Output |
|---|---|
| `arrival_rates.ipynb` | Hourly arrival rates λ per customer class |
| `agents_service_rates.ipynb` | Staffing levels N and service rates μ per station |
| `abandonment_times.ipynb` | Customer abandonment rates θ |

---

## Workflow

The pipeline runs in three sequential stages.

### Stage 1 — Offline Pre-training

**Step 1a. Generate H-term training data** (state and gradient observations):
```bash
cd neural_network_solver/offline_approximation/h_term/input_data/configs/<case>/
python ../../h_net_input_data.py
```

**Step 1b. Compute H-term training targets** (LP optimal values, requires Julia + GLPK):
```bash
cd neural_network_solver/offline_approximation/h_term/output_data/configs/<case>/
julia ../../h_term_offline.jl
```

**Step 1c. Train the H-function network:**
```bash
cd neural_network_solver/offline_approximation/h_term/nn_training/
python main.py --config_path config_hterm_elu_4hidden_5000epochs.json --run_name <run_name>
```
Output: `logs_<run_name>/h_network.pth`

**Step 1d. Compute drift-term training targets** (LP solutions, requires Julia + Gurobi):
```bash
cd neural_network_solver/offline_approximation/drift_term/output_data/
julia drift_term_offline.jl <config_path>
```

**Step 1e. Train the drift-term network:**
```bash
cd neural_network_solver/offline_approximation/drift_term/nn_training/
python main.py --config_path <config_path> --run_name <run_name>
```
Available configs: `config_drift_leaky_relu_4hidden_1000epochs_{cmu,cmu_theta,mu}.json`

Output: `logs_<run_name>/drift_network.pth`

### Stage 2 — Main BSDE Solver

With pre-trained H and drift networks in place, set their paths in `configs/<case>/config.json`, then run:
```bash
cd neural_network_solver/solver/
python main.py --config_path configs/<case>/config.json --run_name <run_name>
```

**Key config parameters** (`net_config` block):

| Parameter | Description |
|---|---|
| `num_neurons` | Neurons per hidden layer |
| `num_layers` | Number of hidden layers |
| `activation` | Hidden layer activation (`eLU`, `SiLU`, `LeakyReLU`, etc.) |
| `softplus_output` | If `true`, applies softplus to Y and Z outputs (enforces V ≥ 0, ∇V ≥ 0) |
| `total_iterations` | Number of training iterations |
| `init_learning_rate` | Initial Adam learning rate |
| `milestones` | Steps at which the LR is multiplied by `gamma` |
| `lambd_const` | Weight on the non-negativity penalty in the loss |

Outputs:
- `logs_<run_name>/y_network.pth` — value function network V
- `logs_<run_name>/z_network.pth` — gradient network ∇V
- `cppweights_<run_name>/` — Z-network weights in CSV format for C++ simulation
- `<run_name>_training_history.csv` — training and validation loss history

### Stage 3 — Secondary Analysis

**Policy iteration** (exact solution, 2D cases, requires Gurobi):
```bash
cd secondary_analysis/policy_iteration/
python policy_iteration.py configs/<case>/config.json
```

**Static allocation problem** (requires Julia + GLPK):
```bash
cd secondary_analysis/static_allocation_problem/configs/<case>/
julia ../../static_allocation_model.jl
```

**Scaling algorithm** (construct the 100-dimensional instance):
```bash
cd secondary_analysis/scaling_algorithm/
python scale_system.py config/config.json   # scales topology and rates
python generator.py                          # generates cost and arrival CDF parameters
```

**C++ simulations** (build with CMake, run with a config):
```bash
cd secondary_analysis/simulation/<simulation_type>/
mkdir build && cd build
cmake .. && make
./<executable> ../configs/<case>/config.json
```

Available simulation types: `neural_network_simulation`, `benchmarks_simulation`, `mdp_simulation`.

---

## Dependencies

**Python** (neural network training and policy iteration):
- `torch` (PyTorch)
- `numpy`, `pandas`
- [`munch`](https://github.com/Infinidat/munch)
- `gurobipy`, `scipy` (policy iteration only)

**Julia** (data generation, H-term LP, drift LP, static allocation):
- `JuMP`, `GLPK`, `Gurobi`
- `CSV`, `DataFrames`, `JSON`

**C++** (simulation):
- CMake, OpenMP
- Gurobi C++ API
- [nlohmann/json](https://github.com/nlohmann/json)

---

## Citation

If you use this code, please cite:

```bibtex
@article{ata2025dynamic,
  title   = {Dynamic Scheduling of a Parallel-Server Queueing System:
             A Computational Method for High-Dimensional Problems},
  author  = {Ata, Baris and Kasikaralar, Ebru},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.09799}
}
```

---

## Data Source

The call center data is from the publicly available **US Bank Call Center dataset** provided by the Technion Service Enterprise Engineering Lab (SEELab): https://see-center.iem.technion.ac.il/databases/USBank/
