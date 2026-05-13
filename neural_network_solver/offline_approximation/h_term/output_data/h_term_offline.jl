# Code to generate the output data for offline approximation of the H function
using JuMP, GLPK, CSV, DataFrames, JSON

# ========== Load Config ==========
config = JSON.parsefile("config.json")

cust_dim = config["cust_dim"] # The number of customer classes
server_dim = config["server_dim"] # The number of service stations
scale = config["scale"] # The system parameter 
sqrt_scale = round(sqrt(scale), digits=3)

# ========== Load Data ==========
gradient_obs = CSV.read(config["gradient_file"], DataFrame, header=false) # The gradient observations to use as training data
state_obs = CSV.read(config["state_file"], DataFrame, header=false) # The state observations to use as training data
cost = Matrix(CSV.read(config["cost_file"], DataFrame, header=false)) # Hourly cost rates 
mu = Matrix(CSV.read(config["mu_file"], DataFrame, header=false)) # Hourly service rates
theta = Matrix(CSV.read(config["theta_file"], DataFrame, header=false)) # Hourly abandonment rates
agents = Matrix(CSV.read(config["agents_file"], DataFrame, header=false)) # The number of agent numbers in each service station

optimal_xi = Matrix(CSV.read(config["xi_file"], DataFrame, header=false)) # Optimal solution to the static allocation problem

# ========== Define Sets ==========
K_set = 1:cust_dim # The set of customer classes
J_set = 1:server_dim # The set of service stations

K = Dict(j => findall(x -> x > 0, mu[:, j]) for j in J_set) # K(j) for each j
J = Dict(k => findall(x -> x > 0, mu[k, :]) for k in K_set) # J(k) for each k

# Basic activities, nonbasic activities and edges
act_basic = Dict(k => findall(x -> x > 0, optimal_xi[k, :]) for k in K_set)
B = [(k, j) for k in K_set for j in act_basic[k]] # Basic activities
E = [(k, j) for k in K_set for j in J[k]] # All edges 
N = setdiff(E, B) # Nonbasic activities 

# ========== Define Solve Function ==========
function solve_instance(ins, gradient_row, state_row)
    display(ins)
    gradient = Vector(gradient_row) # gradient observation
    state = Vector(state_row) # state observation

    # Define sets, dicts, and coefficients for each instance inside the loop
    A = Dict((k, j) => cost[k] + (mu[k, j] - theta[k]) * gradient[k] for (k, j) in E)
    x = Dict(k => state[k] for k in K_set)

    model = Model(GLPK.Optimizer)
    @variable(model, z[E]) # decision variables are the number of people in the service along each activity

    # Constraint for nonbasic activities
    for (k, j) in N 
        set_lower_bound(z[(k, j)], 0)  # Ensure z is non-negative only for (k, j) in N
    end
    
    # Constraint for basic activities
    for (k, j) in B 
        set_lower_bound(z[(k, j)], -1 * agents[j]/sqrt_scale * optimal_xi[k,j])  
    end

    # Objective Function
    @objective(model, Max, sum(A[k, j] * z[(k, j)] for (k, j) in E))
    
    # Constraints
    for k in K_set
        @constraint(model, sum(z[(k, j)] for j in J[k]) <= x[k])
    end
    
    for j in J_set
        @constraint(model, sum(z[(k, j)] for k in K[j]) <= 0)
    end

    # Solve the problem
    optimize!(model)
    
    if termination_status(model) != MOI.OPTIMAL
        error("Instance $ins: Non-optimal solution. Status: $(termination_status(model))")
    end
    
    return objective_value(model)
end

# ========== Run All Instances ==========
objective_values = [
    solve_instance(ins, gradient_obs[ins, :], state_obs[ins, :])
    for ins in 1:size(gradient_obs, 1)
]

# Save the objective value observations to a dataframe 
objective_values_df = DataFrame(obj_val = objective_values)
CSV.write(config["output_file"], objective_values_df; header=false)
