using JuMP, GLPK, CSV, DataFrames, JSON

# ========== Load Config ==========
config = JSON.parsefile("config.json")

cust_dim             = config["cust_dim"]
server_dim           = config["server_dim"]
scaling_factor       = config["scaling_factor"]
prelimit_utilization = config["prelimit_utilization"]

# ========== Load Data ==========
N = Matrix(CSV.read(config["agents_file"], DataFrame, header=false)) / scaling_factor
λ = Matrix(CSV.read(config["lambda_file"], DataFrame, header=false)) / scaling_factor
μ = Matrix(CSV.read(config["mu_file"],     DataFrame, header=false))

# ========== Helper: build & solve the SPP for a given λ ==========
function solve_spp(λ, N, μ, cust_dim, server_dim)
    model = Model(GLPK.Optimizer)

    @variable(model, ρ)
    @variable(model, ξ[1:cust_dim, 1:server_dim] >= 0)

    @objective(model, Min, ρ)

    for i in 1:cust_dim
        @constraint(model, sum(N[j] * μ[i,j] * ξ[i,j] for j in 1:server_dim) == λ[i])
    end
    for j in 1:server_dim
        @constraint(model, sum(ξ[i,j] for i in 1:cust_dim) <= ρ)
    end

    optimize!(model)

    optimal_ρ     = objective_value(model)
    optimal_ξ     = value.(ξ)
    reduced_costs = [reduced_cost(ξ[i,j]) for i in 1:cust_dim, j in 1:server_dim]

    return optimal_ρ, optimal_ξ, reduced_costs
end

# ========== First solve: get ρ* with original λ ==========
ρ1, ξ1, rc1 = solve_spp(λ, N, μ, cust_dim, server_dim)
println("First-solve ρ* = ", ρ1)

# ========== Scale λ so the system is critically loaded ==========
λ_scaled = λ ./ ρ1   # equivalent to multiplying by 1/ρ*

# ========== Second solve with scaled λ ==========
ρ2, ξ2, rc2 = solve_spp(λ_scaled, N, μ, cust_dim, server_dim)
println("Second-solve ρ* = ", ρ2, "  (should be ≈ 1.0)")

# ========== Round for readability ==========
rounded_ξ  = round.(ξ2;  digits = 4)
rounded_rc = round.(rc2; digits = 4)
rounded_λ  = round.(λ_scaled; digits = 6)

# ========== Build prelimit λ ==========
prelimit_λ         = rounded_λ .* scaling_factor .* prelimit_utilization
rounded_prelimit_λ = round.(prelimit_λ; digits = 2)

display(rounded_ξ)
display(ρ2)
display(rounded_rc)
display(rounded_prelimit_λ)

# ========== Build utilization tag for filenames ==========
util_tag = string(Int(round(prelimit_utilization * 100)))

# ========== Save CSVs (from the SECOND solve) ==========
CSV.write("lambda_limit.csv",
          DataFrame(rounded_λ, :auto); writeheader=false)

CSV.write("lambda_prelimit_$(util_tag)utilization.csv",
          DataFrame(rounded_prelimit_λ, :auto); writeheader=false)

CSV.write("xi_optimal.csv",
          DataFrame(rounded_ξ, :auto); writeheader=false)

CSV.write("reduced_costs.csv",
          DataFrame(rounded_rc, :auto); writeheader=false)