using JuMP, Gurobi, CSV, DataFrames, JSON

# ========== Load Config ==========
config_file = length(ARGS) >= 1 ? ARGS[1] : "config_cmu.json"
config = JSON.parsefile(config_file)

cust_dim = config["cust_dim"]
server_dim = config["server_dim"]
scale = config["scale"]
policy = config["policy"]

# ========== Load Data ==========
state_obs = CSV.read(config["state_file"], DataFrame, header=false)
cost = Matrix(CSV.read(config["cost_file"], DataFrame, header=false))
mu = Matrix(CSV.read(config["mu_file"], DataFrame, header=false))
theta = Matrix(CSV.read(config["theta_file"], DataFrame, header=false))
agents = transpose(Matrix(CSV.read(config["agents_file"], DataFrame, header=false)))
optimal_xi = Matrix(CSV.read(config["xi_file"], DataFrame, header=false))

# ========== Precompute Constants ==========
edges = mu .> 0
basic = edges .* (optimal_xi .> 0)
nonbasic = edges .* (optimal_xi .== 0)
lower_bound = -1/sqrt(scale) .* optimal_xi .* agents
theta_minus_mu = theta .- mu

A = if policy == "cmu"
    cost .* mu
elseif policy == "cmu_theta"
    cost .* mu ./ theta
elseif policy == "cost"
    cost
elseif policy == "cost_mu_theta_diff"
    cost .* (mu .- theta)
elseif policy == "mu_theta_diff"
    mu .- theta
elseif policy == "mu"
    mu
else
    error("Unsupported policy")
end
A = A .* edges

# ========== Allocate Results ==========
drift_term_results = zeros(size(state_obs, 1), cust_dim)

# ========== Initialize Model ==========
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "OutputFlag", 0)
set_optimizer_attribute(model, "Method", 2)  # Barrier method for LPs
set_optimizer_attribute(model, "Threads", 4)

# ========== Solve Instances ==========
@variable(model, z[1:cust_dim, 1:server_dim])
@objective(model, Max, sum(A .* z))
con_state = @constraint(model, [k=1:cust_dim], sum(z[k,j] * edges[k,j] for j in 1:server_dim) <= 0)
con_server = @constraint(model, [j=1:server_dim], sum(z[k,j] * edges[k,j] for k in 1:cust_dim) <= 0)
con_nonbasic = @constraint(model, [k=1:cust_dim, j=1:server_dim; nonbasic[k,j] > 0], z[k,j] >= 0)
con_basic = @constraint(model, [k=1:cust_dim, j=1:server_dim; basic[k,j] > 0], z[k,j] >= lower_bound[k,j])

for ins in 1:size(state_obs, 1)
    state = Vector(state_obs[ins, :])
    
    for k in 1:cust_dim
        set_normalized_rhs(con_state[k], state[k])
    end
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        z_val = value.(z)
        drift_term_results[ins, :] .= vec(sum(theta_minus_mu .* z_val, dims = 2))
    else
        @warn "Failed at instance $ins"
    end
    
    if ins % 1000 == 0
        println("Completed instance $ins")
    end
end

# ========== Save Results ==========
CSV.write(config["output_file"], DataFrame(drift_term_results, :auto); writeheader=false)
