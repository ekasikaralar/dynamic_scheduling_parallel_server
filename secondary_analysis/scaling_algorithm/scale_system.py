"""
Scaling a queueing system to high dimensions with a guaranteed unique SPP tree.

Algorithm:
  (i)   Grow the tree by leaf addition (random permutation)
  (ii)  Allocate capacity at each station via Dirichlet
  (iii) Back out arrival rates from target utilization rho* = 1
  (iv)  Add non-tree edges with rates below the dual bound
"""

import numpy as np
from collections import defaultdict


def scale_system(
    K, J, L, I,
    mu_orig,        # (K, J) service rates, 0 if no edge
    E_orig,         # set of (k, j) edges in original system
    T_orig,         # set of (k, j) tree edges in original system
    N_orig,         # (J,) staffing levels
    theta_orig,     # (K,) abandonment rates
    N_tot,          # target total staffing
    N_min=1,        # minimum staffing per station
    delta=0.01,     # discount for non-tree rates
    seed=None,
):
    """
    Scale from (K classes, J stations) to (L classes, I stations).

    Returns a dict with all system parameters and the optimal tree.
    """
    rng = np.random.default_rng(seed)

    assert N_tot > I * N_min, f"Need N_tot > I * N_min = {I * N_min}"

    # --- Step (i): Grow the tree ---

    # Assign types to new nodes
    parent_class = list(range(K))  # k(l) = l for original classes
    for _ in range(L - K):
        parent_class.append(rng.integers(0, K))

    template_station = list(range(J))  # t(i) = i for original stations
    for _ in range(I - J):
        template_station.append(rng.integers(0, J))

    # Build adjacency of original system for quick lookup
    orig_stations_for_class = defaultdict(list)  # k -> [j with (k,j) in E_orig]
    orig_classes_for_station = defaultdict(list)  # j -> [k with (k,j) in E_orig]
    for (k, j) in E_orig:
        orig_stations_for_class[k].append(j)
        orig_classes_for_station[j].append(k)

    # Initialize tree with original tree
    tree_edges = set()
    for (k, j) in T_orig:
        tree_edges.add((k, j))

    classes_in_tree = set(range(K))
    stations_in_tree = set(range(J))

    # Create list of new nodes: ('class', index) or ('station', index)
    new_nodes = []
    for l in range(K, L):
        new_nodes.append(('class', l))
    for i in range(J, I):
        new_nodes.append(('station', i))

    # Random permutation
    rng.shuffle(new_nodes)

    # Attach each node
    for node_type, idx in new_nodes:
        if node_type == 'class':
            l = idx
            k_l = parent_class[l]
            # Eligible stations: those in tree whose template connects to k_l
            eligible = [i for i in stations_in_tree
                        if (k_l, template_station[i]) in E_orig]
            assert len(eligible) > 0, (
                f"No eligible station for class {l} with parent {k_l}"
            )
            i = rng.choice(eligible)
            tree_edges.add((l, i))
            classes_in_tree.add(l)

        else:  # 'station'
            i = idx
            t_i = template_station[i]
            # Eligible classes: those in tree whose parent connects to t_i
            eligible = [l for l in classes_in_tree
                        if (parent_class[l], t_i) in E_orig]
            assert len(eligible) > 0, (
                f"No eligible class for station {i} with template {t_i}"
            )
            l = rng.choice(eligible)
            tree_edges.add((l, i))
            stations_in_tree.add(i)

    assert len(tree_edges) == L + I - 1, (
        f"Tree has {len(tree_edges)} edges, expected {L + I - 1}"
    )

    # Service rates on tree edges
    mu_tree = {}
    for (l, i) in tree_edges:
        mu_tree[(l, i)] = mu_orig[parent_class[l], template_station[i]]

    # Staffing levels
    template_sizes = np.array([N_orig[template_station[i]] for i in range(I)],
                              dtype=float)
    weights = template_sizes / template_sizes.sum()
    nu = N_min + (N_tot - I * N_min) * weights
    nu = np.round(nu).astype(int)  # Round to integers
    
    # Adjust to ensure exact total
    diff = N_tot - nu.sum()
    if diff != 0:
        # Add/subtract from largest stations
        idx = np.argsort(nu)[::-1]
        for i in range(abs(diff)):
            nu[idx[i % I]] += np.sign(diff)

    # Abandonment rates
    theta = np.array([theta_orig[parent_class[l]] for l in range(L)])

    # --- Step (ii): Allocate capacity via Dirichlet ---

    # Build tree neighbors for each station
    station_neighbors = defaultdict(list)  # i -> [l, ...]
    for (l, i) in tree_edges:
        station_neighbors[i].append(l)

    xi_star = {}
    for i in range(I):
        neighbors = station_neighbors[i]
        if len(neighbors) == 1:
            xi_star[(neighbors[0], i)] = 1.0
        else:
            alloc = rng.dirichlet(np.ones(len(neighbors)))
            for l, a in zip(neighbors, alloc):
                xi_star[(l, i)] = a

    # --- Step (iii): Arrival rates ---

    lambda_new = np.zeros(L)
    for (l, i) in tree_edges:
        lambda_new[l] += nu[i] * mu_tree[(l, i)] * xi_star[(l, i)]

    assert np.all(lambda_new > 0), "Some arrival rates are non-positive"

    # --- Step (iv): Non-tree edges with safe rates ---

    # Compute dual variables by tree traversal
    alpha = np.full(L, np.nan)
    beta = np.full(I, np.nan)

    # Build tree adjacency for traversal
    class_to_stations = defaultdict(list)
    station_to_classes = defaultdict(list)
    for (l, i) in tree_edges:
        class_to_stations[l].append(i)
        station_to_classes[i].append(l)

    # Start from class 0
    alpha[0] = 1.0
    queue = [('class', 0)]
    visited = set()
    visited.add(('class', 0))

    while queue:
        node_type, idx = queue.pop(0)
        if node_type == 'class':
            l = idx
            for i in class_to_stations[l]:
                if ('station', i) not in visited:
                    beta[i] = nu[i] * mu_tree[(l, i)] * alpha[l]
                    visited.add(('station', i))
                    queue.append(('station', i))
        else:
            i = idx
            for l in station_to_classes[i]:
                if ('class', l) not in visited:
                    alpha[l] = beta[i] / (nu[i] * mu_tree[(l, i)])
                    visited.add(('class', l))
                    queue.append(('class', l))

    assert not np.any(np.isnan(alpha)), "Some alpha not determined"
    assert not np.any(np.isnan(beta)), "Some beta not determined"
    assert np.all(alpha > 0), "Some alpha non-positive"
    assert np.all(beta > 0), "Some beta non-positive"

    # Full edge set and non-tree rates
    E_full = set()
    mu_full = {}

    # Tree edges
    for (l, i) in tree_edges:
        E_full.add((l, i))
        mu_full[(l, i)] = mu_tree[(l, i)]

    # Non-tree edges
    for l in range(L):
        for i in range(I):
            if (l, i) in tree_edges:
                continue
            k_l = parent_class[l]
            t_i = template_station[i]
            if (k_l, t_i) not in E_orig:
                continue
            # This is a feasible non-tree edge
            bound = beta[i] / (nu[i] * alpha[l])
            inherited_rate = mu_orig[k_l, t_i]
            mu_full[(l, i)] = min(inherited_rate, (1 - delta) * bound)
            E_full.add((l, i))

    # --- Verification ---
    # Check reduced costs
    for (l, i) in E_full:
        if (l, i) in tree_edges:
            continue
        rc = beta[i] - nu[i] * mu_full[(l, i)] * alpha[l]
        assert rc > -1e-12, (
            f"Negative reduced cost {rc} on edge ({l},{i})"
        )
        assert rc > 1e-10, (
            f"Zero or near-zero reduced cost {rc} on edge ({l},{i})"
        )

    return {
        'L': L,
        'I': I,
        'parent_class': parent_class,
        'template_station': template_station,
        'tree_edges': tree_edges,
        'E_full': E_full,
        'mu': mu_full,
        'nu': nu,
        'lambda': lambda_new,
        'theta': theta,
        'xi_star': xi_star,
        'alpha': alpha,
        'beta': beta,
    }


def print_system(result):
    """Pretty-print the constructed system."""
    L = result['L']
    I = result['I']

    print(f"System: L = {L} classes, I = {I} stations")
    print(f"Tree edges: {len(result['tree_edges'])}")
    print(f"Total edges: {len(result['E_full'])}")
    print(f"Non-tree edges: {len(result['E_full']) - len(result['tree_edges'])}")
    print()

    print("Tree edges and allocations:")
    print(f"  {'Edge':<10} {'k(l)':<6} {'t(i)':<6} {'μ_li':<8} {'ξ*_li':<8}")
    print(f"  {'-'*42}")
    for (l, i) in sorted(result['tree_edges']):
        k_l = result['parent_class'][l]
        t_i = result['template_station'][i]
        mu = result['mu'][(l, i)]
        xi = result['xi_star'][(l, i)]
        print(f"  ({l},{i}){'':<5} {k_l:<6} {t_i:<6} {mu:<8.3f} {xi:<8.4f}")
    print()

    print("Staffing:")
    for i in range(I):
        print(f"  Station {i}: ν = {result['nu'][i]:.2f} "
              f"(template {result['template_station'][i]})")
    print(f"  Total: {result['nu'].sum():.2f}")
    print()

    print("Arrival rates:")
    for l in range(L):
        print(f"  Class {l}: λ = {result['lambda'][l]:.4f} "
              f"(parent {result['parent_class'][l]})")
    print(f"  Total: {result['lambda'].sum():.4f}")
    print()

    print("Non-tree edges:")
    print(f"  {'Edge':<10} {'μ_li':<8} {'bound':<8} {'red.cost':<10}")
    print(f"  {'-'*38}")
    for (l, i) in sorted(result['E_full'] - result['tree_edges']):
        mu = result['mu'][(l, i)]
        bound = result['beta'][i] / (result['nu'][i] * result['alpha'][l])
        rc = result['beta'][i] - result['nu'][i] * mu * result['alpha'][l]
        print(f"  ({l},{i}){'':<5} {mu:<8.3f} {bound:<8.3f} {rc:<10.4f}")
    print()

    print("Dual variables:")
    print(f"  α = {np.array2string(result['alpha'], precision=4)}")
    print(f"  β = {np.array2string(result['beta'], precision=4)}")


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Scale call center system from config")
    parser.add_argument("config", type=str, help="Path to configuration JSON file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load matrices
    mu_orig = np.loadtxt(config['mu'], delimiter=',')
    xi_orig = np.loadtxt(config['xi'], delimiter=',')
    N_orig = np.loadtxt(config['agents'], delimiter=',')
    theta_orig = np.loadtxt(config['theta'], delimiter=',')

    # Derive K, J from mu matrix
    K, J = mu_orig.shape

    # E_orig: edges where mu > 0
    E_orig = {(k, j) for k in range(K) for j in range(J) if mu_orig[k, j] > 0}

    # T_orig: tree edges where xi > 0
    T_orig = {(k, j) for k in range(K) for j in range(J) if xi_orig[k, j] > 0}

    # Get scaling parameters
    L = config['L']
    I = config['I']
    N_tot = config['N_tot']
    N_min = config['N_min']
    delta = config.get('delta', 0.01)
    seed = config.get('seed', 42)

    result = scale_system(
        K, J, L, I,
        mu_orig, E_orig, T_orig, N_orig, theta_orig,
        N_tot=N_tot, N_min=N_min, delta=delta, seed=seed,
    )

    print("=" * 50)
    print(f"SCALED SYSTEM: K={K}, J={J} -> L={L}, I={I}")
    print("=" * 50)
    print()
    print_system(result)

    # Save outputs
    output_dir = config.get('output_dir', '.')
    
    # Lambda (arrival rates) - 2 decimal points
    np.savetxt(f"{output_dir}/lambda_scaled.csv", result['lambda'], delimiter=',', fmt='%.2f')
    
    # Theta (abandonment rates) - 2 decimal points
    np.savetxt(f"{output_dir}/theta_scaled.csv", result['theta'], delimiter=',', fmt='%.2f')
    
    # Nu (agents per station) - integers
    np.savetxt(f"{output_dir}/agents_scaled.csv", result['nu'], delimiter=',', fmt='%d')
    
    # Mu matrix (service rates) - 2 decimal points
    mu_matrix = np.zeros((L, I))
    for (l, i), rate in result['mu'].items():
        mu_matrix[l, i] = rate
    np.savetxt(f"{output_dir}/mu_scaled.csv", mu_matrix, delimiter=',', fmt='%.2f')
    
    # Xi matrix (allocations) - 4 decimal points
    xi_matrix = np.zeros((L, I))
    for (l, i), alloc in result['xi_star'].items():
        xi_matrix[l, i] = alloc
    np.savetxt(f"{output_dir}/xi_scaled.csv", xi_matrix, delimiter=',', fmt='%.4f')
    
    print(f"\nSaved outputs to {output_dir}/")
