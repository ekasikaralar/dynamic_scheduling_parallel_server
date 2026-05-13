#include "test.h"

namespace simulation {
	
	BasisLRUCache Execute::basis_cache(/*cache_capacity=*/20000);
	std::unordered_map<uint64_t, std::vector<std::vector<double>>> Execute::policy_cache;

	Simulation::Simulation(const std::string& jsonFileName){

		// Create a JSON object
		nlohmann::json config;

		// Open the JSON file
    	std::ifstream file(jsonFileName);
		if (!file.is_open()) {
			throw std::runtime_error("Unable to open config file: " + jsonFileName);
		}

		// Try to parse the JSON file
    	try {
        	file >> config;
    	} catch (const std::exception& e) {
        	// Handle parsing error
        	throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    	}

		// Close the JSON file
    	file.close();

		// Accessing configuration values
		class_no = config["class_no"]; //K
		station_no = config["station_no"]; //J
		num_iterations = config["num_iterations"];
		gamma = config["gamma"]; // interest rate;
		arrival_rate = config["arrival_rate"]; // hourly arrival rate
		policy = config["policy"].get<std::string>(); // "mu", "cmu", "cmu_theta", "gcmu"

		// Validate the policy early so failures are loud and clear
		static const std::set<std::string> valid_policies =
			{"mu", "cmu", "cmu_theta", "gcmu"};
		if (valid_policies.find(policy) == valid_policies.end()) {
			throw std::runtime_error("Unknown policy '" + policy +
				"' (must be one of: mu, cmu, cmu_theta, gcmu)");
		}

		std::string agents_path = config["agents_path"];
		std::string mu_hourly_path = config["mu_hourly_path"];
		std::string theta_hourly_path = config["theta_hourly_path"];
		std::string arr_cdf_path = config["arr_cdf_path"];
		std::string cost_rate_path = config["cost_rate_path"];
		std::string optimal_xi_path = config["xi_path"];

		// Number of agents in each service station (J dimensional)
		std::vector<double> agents = readVectorFromCSV(agents_path);
		no_server.resize(agents.size());
    	std::transform(agents.begin(), agents.end(), no_server.begin(), [](double val) { return static_cast<int>(val); });
		
		// K * J dimensional matrix -- rows customer classes and columns service stations
		mu_hourly = readMatrixFromCSV(mu_hourly_path); // hourly service rate

		// K dimensional hourly abandonment rates
		theta_hourly = readVectorFromCSV(theta_hourly_path);  // hourly abandonment rate

		// Cumulative distribution function for the arrivals
		arr_cdf = readVectorFromCSV(arr_cdf_path); 
		
		// Hourly state costs
		cost_rate = readVectorFromCSV(cost_rate_path); // hourly cost rate

		// The solution of the static allocation problem
		optimal_xi = readMatrixFromCSV(optimal_xi_path);
		
		edges = std::vector<std::vector<double>>(class_no, std::vector<double>(station_no, 0.0)); // The edges between the classes and stations

		for (size_t i = 0; i < mu_hourly.size(); ++i) {
			for (size_t j = 0; j < mu_hourly[i].size(); ++j) {
				edges[i][j] = (mu_hourly[i][j] > 0); // If the corresponding service rate is positive, then there is an edge between the class and the service station
			}
		}
	}
	
	std::vector<std::string> Simulation::splitString(const std::string& input, char delimiter) {
	    std::vector<std::string> tokens;
	    std::string token;
	    std::istringstream tokenStream(input);
	    while (std::getline(tokenStream, token, delimiter)) {
	        tokens.push_back(token);
		}
		return tokens;
	}

	// Function to read matrix from CSV file
	std::vector<std::vector<double> > Simulation::readMatrixFromCSV(const std::string& filename) {
	    std::vector<std::vector<double> > matrix;

	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        std::cerr << "Failed to open the file: " << filename << std::endl;
	        return matrix;
	    }

	    std::string line;
	    while (std::getline(file, line)) {
	        std::vector<std::string> row = splitString(line, ',');
	        std::vector<double> matrixRow;
	        for (const std::string& str : row) {
	            matrixRow.push_back(std::stod(str));
	        }
	        matrix.push_back(matrixRow);
	    }

	    file.close();

	    return matrix;
	}
	
    // Function to read vector from CSV file
	std::vector<double> Simulation::readVectorFromCSV(const std::string& filename) {
	    std::vector<double> vec;
	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        std::cout << "Failed to open the file: " << filename << std::endl;
	        return vec;
	    }

	    std::string line;
	    while (std::getline(file, line)) {
	        std::stringstream ss(line);
	        std::string cell;
	        while (std::getline(ss, cell, ',')) {
	            double value = std::stod(cell);
	            vec.push_back(value);
	        }
	    }

	    file.close();
	    return vec;
	}

	std::vector<int> Simulation::readIndicesFromCSV(const std::string& filename) {
	    std::vector<int> indices;
	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        std::cout << "Failed to open the file: " << filename << std::endl;
	        return indices;
	    }

	    std::string line;
	    std::getline(file, line); // Skip header
	    while (std::getline(file, line)) {
	        if (!line.empty()) {
	            indices.push_back(std::stoi(line));
	        }
	    }

	    file.close();
	    return indices;
	}

	Simulation::~Simulation(){
		std::cout << "Done" << std::endl;
	}


	int Simulation::save(int start_iter, int end_iter, const std::string& record_file){
		int fd = open(record_file.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
		if (fd == -1) {
			perror("open");
			return -1;
		}

		auto write_all = [&](const std::string& s) {
			const char* p = s.c_str();
			size_t len = s.size();
			while (len > 0) {
				ssize_t w = write(fd, p, len);
				if (w <= 0) {
					perror("write");
					close(fd);
					throw std::runtime_error("write failed");
				}
				p   += w;
				len -= w;
			}
		};

		// header
		std::string header = "iter";
		for (int i = 0; i < class_no; ++i) {
			header += ",cost_class_" + std::to_string(i);
		}
		header += ",total_cost,runtime_sec\n";
		write_all(header);

		// loop over iterations
		for (int iter = start_iter; iter < end_iter; ++iter) {
			std::vector<double> cost(class_no + 1, 0.0);

			auto start = std::chrono::steady_clock::now();

			simulation::Execute exec(
				class_no, station_no, arr_cdf, mu_hourly, theta_hourly,
				no_server, cost_rate, optimal_xi, edges,
				arrival_rate, policy,
				iter
			);
			cost = exec.run(gamma, iter);

			auto end = std::chrono::steady_clock::now();
			double elapsed_sec =
				std::chrono::duration<double>(end - start).count();

			std::string line;
			line.reserve(64 + class_no * 32);

			line += std::to_string(iter);
			line += ",";

			for (int i = 0; i < class_no; ++i) {
				line += std::to_string(cost[i]);
				line += ",";
			}
			line += std::to_string(cost[class_no]);  // total cost
			line += ",";
			line += std::to_string(elapsed_sec);
			line += "\n";

			write_all(line);   // write this iteration's result immediately
		}

		close(fd);
		return 0;
	}



	Execute::Execute(int class_no_, int station_no_,
                 const std::vector<double>& arr_cdf_,
                 const std::vector<std::vector<double>>& mu_hourly_,
                 const std::vector<double>& theta_hourly_, 
                 const std::vector<int>& no_server_,
                 const std::vector<double>& cost_rate_,
                 const std::vector<std::vector<double>>& optimal_xi_,
                 const std::vector<std::vector<double>>& edges_,
                 double arrival_rate_,
                 const std::string& policy_,
                 int seed)
		: class_no(class_no_),
		station_no(station_no_),
		arr_cdf(arr_cdf_),
		mu_hourly(mu_hourly_),
		theta_hourly(theta_hourly_),
		no_server(no_server_),
		cost_rate(cost_rate_),
		optimal_xi(optimal_xi_),
		edges(edges_),
		arrival_rate(arrival_rate_),
		policy(policy_),
		generator(seed)
	{
		edge_list.clear();
		edge_list.reserve(class_no * station_no);
		for (int i = 0; i < class_no; ++i) {
			for (int j = 0; j < station_no; ++j) {
				if (edges[i][j] == 1) edge_list.push_back({i, j});
			}
		}

		queue_init();

		// Gurobi tuning for maximum speed
		model.set(GRB_IntParam_OutputFlag, 0);
		model.set(GRB_IntParam_Method, 1);     // dual simplex
		model.set(GRB_IntParam_Presolve, 0);   
		model.set(GRB_IntParam_Threads, 1);
		model.set(GRB_DoubleParam_OptimalityTol, 1e-4); // Looser tolerance
		model.set(GRB_DoubleParam_FeasibilityTol, 1e-4); 

		// ---- RNG distributions init ----
		uniform01 = std::uniform_real_distribution<double>(0.0, 1.0);

		// arrival_rate now comes from the config file via Simulation
		interarrivalDist = std::exponential_distribution<double>(arrival_rate);

		abandonDist.resize(class_no);
		for (int i = 0; i < class_no; ++i) {
			abandonDist[i] = std::exponential_distribution<double>(theta_hourly[i]);
		}

		num_served_buf.resize(class_no);
		optimal_queue_len_buf.resize(class_no);
		diff_buf.resize(class_no);
		
		// Pre-compute the static objective matrix A based on the chosen policy.
		//   mu         : A_ij = mu_ij
		//   cmu        : A_ij = c_i * mu_ij
		//   cmu_theta  : A_ij = c_i * mu_ij / theta_i
		//   gcmu       : A_ij starts as c_i * mu_ij; it is refreshed every
		//                call to linear_program() using the current queue length.
		A_matrix.assign(class_no, std::vector<double>(station_no, 0.0));
		for (int i = 0; i < class_no; ++i){
			for (int j = 0; j < station_no; ++j){
				if (mu_hourly[i][j] == 0.0) continue;  // non-edge stays zero

				if (policy == "mu") {
					A_matrix[i][j] = mu_hourly[i][j];
				} else if (policy == "cmu" || policy == "gcmu") {
					A_matrix[i][j] = cost_rate[i] * mu_hourly[i][j];
				} else if (policy == "cmu_theta") {
					A_matrix[i][j] = cost_rate[i] * mu_hourly[i][j] / theta_hourly[i];
				}
			}
		}
	}


	Execute::~Execute()
	{
	}

    // Initializes the queues for each class in the simulation
	void Execute::queue_init()
	{
		queue_list.clear();
		arr_list.clear();
		abandonment_list.clear();
		next_abandon_time.clear();

		queue_list.resize(class_no);
		arr_list.resize(class_no);            
		abandonment_list.resize(class_no);     
		next_abandon_time.assign(class_no, inf);

		t_abandon   = inf;
		class_abandon = -1;
		cust_abandon  = -1;
	}

	std::vector<std::vector<double>> Execute::initializeModel(std::vector<int>& num_in_system){
		
		class_constraints.resize(class_no);
		station_constraints.resize(station_no);

		// Define variables (d) in the model
		d = std::vector<std::vector<GRBVar>>(class_no, std::vector<GRBVar>(station_no));
        for (int i = 0; i < class_no; ++i) {
			for (int j = 0; j < station_no; ++j) {
				if (edges[i][j] == 1) {
					d[i][j] = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS);
				} else {
					d[i][j] = model.addVar(0.0, 0.0, 0.0, GRB_CONTINUOUS);  // Force d[i][j] == 0
				}
			}
		}

		// Set the objective function using pre-computed matrix
        GRBLinExpr objective;
        for (int i = 0; i < class_no; ++i){
            for (int j = 0; j < station_no; ++j){
                objective += A_matrix[i][j] * d[i][j];
            }
        }
    
		model.setObjective(objective, GRB_MAXIMIZE);

		// Add prelimit constraints
		for (int i = 0; i < class_no; ++i) {
            GRBLinExpr lhs;
            for (int j = 0; j < station_no; ++j) {
                lhs += d[i][j] * edges[i][j]; // Assuming d is a flattened 1D vector of GRBVar
            }
            class_constraints[i] = model.addConstr(lhs <= num_in_system[i + 1]);
        }

		for (int i = 0; i < station_no; ++i) {
            GRBLinExpr lhs;
            for (int j = 0; j < class_no; ++j) {
                lhs += d[j][i] * edges[j][i];
            }
            station_constraints[i] = model.addConstr(lhs <= no_server[i]);
        }

		// Optimize the model
        model.optimize();

        // Retrieve the optimal solution
		optimal_d = std::vector<std::vector<double>>(class_no, std::vector<double>(station_no));

		for (int i = 0; i < class_no; ++i) {
            for (int j = 0; j < station_no; ++j) {
            	optimal_d[i][j] = d[i][j].get(GRB_DoubleAttr_X);
			}
        }

		return optimal_d; // first index is class, second index is service station
	}

	const std::vector<std::vector<double>> Execute::linear_program(std::vector<int>& num_in_system) {

		// Only the generalized cμ policy (gcmu) has a state-dependent
		// objective that must be refreshed every iteration.
		// For mu / cmu / cmu_theta, A_matrix was set once in the constructor
		// and does not change, so we only need to update the RHS.
		if (policy == "gcmu") {
			for (int i = 0; i < class_no; ++i) {
				for (int j = 0; j < station_no; ++j) {
					if (mu_hourly[i][j] != 0.0) {
						A_matrix[i][j] = cost_rate[i] * mu_hourly[i][j]
						               * (num_in_queue[i + 1] + 0.1);
					}
				}
			}

			GRBLinExpr objective;
			for (int i = 0; i < class_no; ++i) {
				for (int j = 0; j < station_no; ++j) {
					objective += A_matrix[i][j] * d[i][j];
				}
			}
			model.setObjective(objective, GRB_MAXIMIZE);
		}

		// update RHS of class constraints
		for (int i = 0; i < class_no; ++i)
			class_constraints[i].set(GRB_DoubleAttr_RHS, num_in_system[i + 1]);
		
		model.optimize();

		// read back solution (zero non-edges!)
		for (int i = 0; i < class_no; ++i) {
			for (int j = 0; j < station_no; ++j) {
				optimal_d[i][j] = (edges[i][j] == 1) ? d[i][j].get(GRB_DoubleAttr_X) : 0.0;
			}
		}
		return optimal_d;
	}

    /**
 		* Generates a random interarrival time based on an exponential distribution.
 		* 
 		* @return A randomly generated interarrival time.
 	*/ 
	double Execute::generate_interarrival(){	
		// just draw from the pre-built distribution
		return interarrivalDist(generator);   
	}

    /**
		* Generates a random abandonment time based on an exponential distribution.
 		* 
 		* @param cls The class index used to select the abandonment rate from 'theta_hourly'.
 		* @return A randomly generated abandonment time.
 	*/ 
	double Execute::generate_abandon(int& cls) {
		// reuse the per-class exponential distribution
		return abandonDist[cls](generator);
	}

	/**
 		* Generates a random service time based on an exponential distribution.
 		*
 		* The service time is determined by calculating the service rate as a sumproduct of the number
 		* of services in each class ('num_in_service') and the hourly service rate for each class ('mu_hourly').
 		*
 		* @return A randomly generated service time for the indicated station.
 	*/ 
	double Execute::generate_service(int& station){
		double service_rate = 0.0;				
		for (int i = 0; i < class_no; ++i){
			service_rate += num_in_service[station + 1][i + 1] * mu_hourly[i][station];
		}

		if (service_rate <= 0.0) {
			// No one in service or zero rate: no departure from this station
			return inf;
		}

		std::exponential_distribution<double> serviceTimeDistribution(service_rate);
		return serviceTimeDistribution(generator);
	}

	void Execute::system_arrangement(std::vector<int>& num_in_system)
	{
		// Re-solve LP and get optimal number in service per (class, station)
		std::vector<std::vector<double>> optimal_policy = linear_program(num_in_system);

		for (int i = 0; i < class_no; ++i) {

			int served_i = 0;
			for (int j = 0; j < station_no; ++j) {
				int x_ij = static_cast<int>(std::lround(optimal_policy[i][j]));
				served_i += x_ij;
			}
			num_served_buf[i] = served_i;

			// optimal queue length for class i
			optimal_queue_len_buf[i] = num_in_system[i + 1] - served_i;

			// current queue vs optimal queue length
			int d = num_in_queue[i + 1] - optimal_queue_len_buf[i];
			diff_buf[i] = d;

			// keep aggregate queue size consistent
			num_in_queue[0]     -= d;
			num_in_queue[i + 1] -= d;

			if (d > 0) {
				// Remove d customers from queue (send to service)
				for (int k = 0; k < d; ++k) {
					remove_queue(i);
				}
			} else if (d < 0) {
				// Add |d| customers back to the queue
				int add_back = -d;
				for (int k = 0; k < add_back; ++k) {
					add_queue(t_event, i);
				}
			}
		}

		// reset aggregates
		num_in_service[0][0] = 0;
		for (int j = 0; j < station_no; ++j) {
			num_in_service[j + 1][0] = 0; // total at station j
		}
		for (int i = 0; i < class_no; ++i) {
			num_in_service[0][i + 1] = 0; // total for class i
		}

		// fill matrix and row/column totals in a single nested loop
		for (int i = 0; i < class_no; ++i) {
			for (int j = 0; j < station_no; ++j) {
				int x_ij = static_cast<int>(std::lround(optimal_policy[i][j]));

				num_in_service[j + 1][i + 1] = x_ij;  // (station j, class i)

				num_in_service[j + 1][0]    += x_ij;  // total at station j
				num_in_service[0][i + 1]    += x_ij;  // total of class i
				num_in_service[0][0]        += x_ij;  // total in service overall
			}
		}
	}
		

	void Execute::rearrange_system() {

		// How many servers are currently busy 
		int busy_servers = num_in_service[0][0];

		bool has_idle_server   = (busy_servers < total_servers);
		bool has_waiting_jobs  = (num_in_queue[0] > 0); // Only solve when queue > 5

		// Only re-solve the LP if there is both idle capacity and waiting work
		if (has_idle_server && has_waiting_jobs) {
			system_arrangement(num_in_system);
		}

		// If the system becomes empty, make sure no departures are scheduled
		if (num_in_service[0][0] > 0){
			for (int i = 0; i < station_no; ++i) {
				t_depart[i] = sim_clock + generate_service(i); // schedule the next departure event 
			} 
		} else{
			for (int i = 0; i < station_no; ++i) {
				t_depart[i] = inf; // schedule the next departure event 
			} 
		}
	}

	int Execute::choose_station_for_arrival(int cls) {
		int best_station = -1;
		double best_mu = -1.0;

		for (int j = 0; j < station_no; ++j) {
			// Station must be able to serve this class
			if (edges[cls][j] == 0) continue;

			// Must have idle capacity at this station
			int busy_at_j = num_in_service[j + 1][0];
			if (busy_at_j >= no_server[j]) continue;

			double mu = mu_hourly[cls][j];

			// pick the highest μ; tie-breaker by lower load or index if you like
			if (mu > best_mu) {
				best_mu = mu;
				best_station = j;
			}
		}
		return best_station;  // -1 means: no station can take this job directly
	}

    /**
 		* Handles the arrival event in a queueing system simulation.
 		* 
 		* This function updates the state of the system upon the arrival of a new entity (person, job, etc.) and decides 
		* whether to serve the new arrival immediately or queue it, based on the current system state and queueing discipline.
 		*
 		* @param cls Class of the arriving entity.
 	*/ 
	void Execute::handle_arrival_event(int& cls){ 

		// --- bookkeeping ---
		num_in_system[0]     += 1;
		num_in_system[cls+1] += 1;

		num_arrivals[0]      += 1; 
		num_arrivals[cls+1]  += 1; 
			
		num_in_queue[0]      += 1; 
		num_in_queue[cls+1]  += 1; 
			
		// add to queue
		add_queue(t_event, cls); 
		
		// schedule next arrival
		t_arrival = sim_clock + generate_interarrival();

		int busy_servers = num_in_service[0][0];

		bool has_idle_server  = (busy_servers < total_servers);
		bool is_single_in_q   = (num_in_queue[0] == 1); // the only waiting job is this arrival

		if (has_idle_server && is_single_in_q) {
			// then send this job to the station with the highest service rate if there is an idle capacity
			int best_station = choose_station_for_arrival(cls);

			if (best_station != -1) {
				// remove this job from queue
				num_in_queue[0]     -= 1;
				num_in_queue[cls+1] -= 1;
				remove_queue(cls);

				// put into service
				num_in_service[0][0]               += 1;
				num_in_service[0][cls+1]           += 1;
				num_in_service[best_station+1][0]  += 1;
				num_in_service[best_station+1][cls+1] += 1;

				// schedule departure time for that station
				t_depart[best_station] = sim_clock + generate_service(best_station);
				// done: no LP, no rearrange_system
				return;
			}
			// else: no station could take it then fall through to LP
		}

		// ---- general case: now we really have a queue or no idle capacity → use LP ----
		rearrange_system();
	}


    /**
 		* This function updates the state of the system when an entity (person, job, etc.) departs from service. It adjusts the 
 		* counts of entities in the system and in service, and determines whether to bring new entities into service based on the 
 		* current state and the queueing discipline.
 		*
 		* @param cls Class of the departing entity.
 		* @param station The service station that customer leaves from.
 	*/ 
	void Execute::handle_depart_event(int& station, int& cls){ 
		
		// number of people in the system has decreased as someone left the system
		num_in_system[0] -= 1; 
		num_in_system[cls + 1] -=1;
		
		// number of departures has increased by 1
		num_departures[0] += 1;
		num_departures[cls + 1] += 1;

		num_in_service[0][0] -= 1; 
		num_in_service[0][cls + 1] -= 1; // total number of people in service from class k has decreased
		
		num_in_service[station + 1][0] -= 1; 
		num_in_service[station + 1][cls + 1] -= 1; // number of people in service from class k at station j has decreased	

		// else: no station could take it then fall through to LP
		rearrange_system();

	}

    /**
		* This function is called when a customer decides to leave the queue before getting the service. It updates the system's
 		* state by decrementing the count of people in the system and in the specific class of the customer who abandoned. 
 
 		* If the condition of system overload or interval change is met, the function calls 'optimal_policy_calculation' 
 		* to recalibrate the service strategy.
 		*
 	*/ 
	void Execute::handle_abandon_event() {
		// removing the person who has abandoned from the system
		num_in_system[0]          -= 1;             // total in system
		num_in_system[class_abandon + 1] -= 1;      // in the abandoned class

 		// increasing the cumulative number of abandons
		num_abandons[0]               += 1;
		num_abandons[class_abandon + 1] += 1;

		// decreasing the number of people in the queue
		num_in_queue[0]          -= 1;
		num_in_queue[class_abandon + 1] -= 1;

		// Remove abandoned customer from queue and abandonment list
		if (cust_abandon >= 0 &&
			cust_abandon < static_cast<int>(queue_list[class_abandon].size())) {
			queue_list[class_abandon].erase(
				queue_list[class_abandon].begin() + cust_abandon
			);
		}

		if (cust_abandon >= 0 &&
			cust_abandon < static_cast<int>(abandonment_list[class_abandon].size())) {
			abandonment_list[class_abandon].erase(
				abandonment_list[class_abandon].begin() + cust_abandon
			);
		}

		// Update per-class next abandonment time for this class
		if (abandonment_list[class_abandon].empty()) {
			next_abandon_time[class_abandon] = inf;
		} else {
			next_abandon_time[class_abandon] = *std::min_element(
				abandonment_list[class_abandon].begin(),
				abandonment_list[class_abandon].end()
			);
		}

		// Determine the next abandonment time globally
		if (num_in_queue[0] > 0) {
			double best_time  = inf;
			int    best_class = -1;

			for (int i = 0; i < class_no; ++i) {
				if (next_abandon_time[i] < best_time) {
					best_time  = next_abandon_time[i];
					best_class = i;
				}
			}

			if (best_class == -1 || best_time == inf) {
				t_abandon     = inf;
				class_abandon = -1;
				cust_abandon  = -1;
			} else {
				t_abandon     = best_time;
				class_abandon = best_class;

				// find the customer index within that class
				auto it = std::min_element(
					abandonment_list[best_class].begin(),
					abandonment_list[best_class].end()
				);
				cust_abandon = static_cast<int>(
					std::distance(abandonment_list[best_class].begin(), it)
				);
			}
		} else {
			t_abandon     = inf;
			class_abandon = -1;
			cust_abandon  = -1;
		}
	}
	
	
	void Execute::add_queue(double& arr_time, int& cls) {
		// Add arriving person
		queue_list[cls].push_back(arr_time);
		double abandon_time = arr_time + generate_abandon(cls);
		abandonment_list[cls].push_back(abandon_time);

		// Update per-class next abandonment time
		if (abandon_time < next_abandon_time[cls]) {
			next_abandon_time[cls] = abandon_time;
		}

		// Find global earliest abandonment among classes
		double best_time  = inf;
		int    best_class = -1;
		for (int i = 0; i < class_no; ++i) {
			if (next_abandon_time[i] < best_time) {
				best_time  = next_abandon_time[i];
				best_class = i;
			}
		}

		if (best_class == -1) {
			t_abandon     = inf;
			class_abandon = -1;
			cust_abandon  = -1;
			return;
		}

		t_abandon     = best_time;
		class_abandon = best_class;

		// Find which customer in that class has the earliest abandonment time
		auto it = std::min_element(
			abandonment_list[best_class].begin(),
			abandonment_list[best_class].end()
		);
		cust_abandon = static_cast<int>(
			std::distance(abandonment_list[best_class].begin(), it)
		);
	}
	
	void Execute::remove_queue(int& cls) {

		if (!queue_list[cls].empty()) {
			queue_list[cls].pop_front();
		}
		if (!abandonment_list[cls].empty()) {
			abandonment_list[cls].erase(abandonment_list[cls].begin());
		}

		// Update per-class next abandonment time for this class
		if (abandonment_list[cls].empty()) {
			next_abandon_time[cls] = inf;
		} else {
			next_abandon_time[cls] = *std::min_element(
				abandonment_list[cls].begin(),
				abandonment_list[cls].end()
			);
		}

		// If no one is in queue globally, no abandonment
		if (num_in_queue[0] == 0) {
			t_abandon     = inf;
			class_abandon = -1;
			cust_abandon  = -1;
			return;
		}

		// Find global earliest abandonment among classes
		double best_time  = inf;
		int    best_class = -1;
		for (int i = 0; i < class_no; ++i) {
			if (next_abandon_time[i] < best_time) {
				best_time  = next_abandon_time[i];
				best_class = i;
			}
		}

		if (best_class == -1 || abandonment_list[best_class].empty()) {
			t_abandon     = inf;
			class_abandon = -1;
			cust_abandon  = -1;
			return;
		}

		t_abandon     = best_time;
		class_abandon = best_class;

		auto it = std::min_element(
			abandonment_list[best_class].begin(),
			abandonment_list[best_class].end()
		);
		cust_abandon = static_cast<int>(
			std::distance(abandonment_list[best_class].begin(), it)
		);
	}


	void Execute::save_monitoring_data(const std::string& filename){
		std::ofstream file(filename);

		if (!file.is_open()) {
			std::cerr << "Failed to open monitoring output file!" << std::endl;
			return;
		}

		// Write headers
		file << "Time,NumInSystem,NumInQueue,CumulativeCost\n";

		// Write recorded data
		for (size_t i = 0; i < time_stamps.size(); ++i){
			file << time_stamps[i] << ","
				 << system_sizes[i] << ","
				 << queue_sizes[i] << ","
				 << cumulative_costs[i] << "\n";
		}

		file.close();
	}

	std::vector<double> Execute::run(double gamma, int iter)
	{
		total_servers = 0;
		for (int j = 0; j < station_no; ++j) {
			total_servers += no_server[j];
		}

        constexpr double MaxTime = std::numeric_limits<double>::max();
		double discount = 1.0;
		
		sim_clock = 0.0;

		time_stamps.clear();
		system_sizes.clear();
		queue_sizes.clear();
		cumulative_costs.clear();

		num_in_system.assign(class_no + 1, 0);
		num_in_service.assign(station_no + 1, std::vector<int>(class_no + 1, 0));

		double total_time = 24 * 1.25; // we simulate 30 hrs
		double steady_state_time = 24 * 0.75; // the 18 hrs are transient period
		double time_delta = total_time - steady_state_time; 

		num_arrivals.assign(class_no + 1, 0); // 0th index is used for the sum
		num_in_queue.assign(class_no + 1, 0); // 0th index is used for the sum
		num_abandons.assign(class_no + 1, 0); // 0th index is used for the sum
		num_departures.assign(class_no + 1, 0); // 0th index is used for the sum

		queue_integral.assign(class_no + 1, 0); // 0th index is used for the sum
		system_integral.assign(class_no + 1, 0); // 0th index is used for the sum
		waiting_cost.assign(class_no, 0);
		discounted_waiting_cost.assign(class_no, 0); 
		
        total_cost = 0;
		t_arrival = generate_interarrival(); // first event should be an arrival
		t_depart.assign(station_no, MaxTime);
		t_abandon = MaxTime;

		numerator_tmp.assign(class_no, 0.0);
		prob_tmp.assign(class_no, 0.0);
		ser_cdf_tmp.assign(class_no, 0.0);
		
		// initialization of the number of people in system and number of people in service		
		for (int i = 0; i < class_no; i++){
			for (int j = 0; j < station_no; j++){
				num_in_service[j + 1][i + 1] = static_cast<int>(std::round(optimal_xi[i][j] * no_server[j]));
				num_in_service[0][0] += num_in_service[j + 1][i + 1];
			}
		}

		for (int j = 0; j < station_no; j++){
			for (int i = 0; i < class_no; i++){
				num_in_service[j + 1][0] += num_in_service[j + 1][i + 1];
			}
		}

		for (int i = 0; i < class_no; i++){
			for (int j = 0; j < station_no; j++){
				num_in_service[0][i + 1] += num_in_service[j + 1][i + 1];
			}
			num_arrivals[i + 1] = num_in_service[0][i+1];
		}

		num_arrivals[0] = num_in_service[0][0];

		for (int i = 0; i < class_no; i++){
			num_in_system[i + 1] = num_in_service[0][i + 1];
			num_in_system[0] += num_in_system[i + 1];
		}
		
		initializeModel(num_in_system); // Initialize the LP model
		
		double recording_interval = 5.0 / 60.0; // 5 minutes = 0.083333... hours
		double next_recording_time = recording_interval;
		bool waiting_cost_reset_done = false;

		// Main part of the simulation
		while (sim_clock < total_time){
			
            int min_depart_station = -1;
			double min_depart_time = MaxTime;

			for (int j = 0; j < station_no; ++j) {
				if (t_depart[j] < min_depart_time) {
					min_depart_time = t_depart[j];
					min_depart_station = j;
				}
			}

			t_event = std::min({t_arrival, t_abandon, min_depart_time});
			double dt = t_event - sim_clock;

			discount *= std::exp(-gamma * dt);

			for (int i = 0; i < class_no + 1; i++) {
				queue_integral[i]  += num_in_queue[i]  * dt;
				system_integral[i] += num_in_system[i] * dt;
			}

			for (int i = 0; i < class_no; ++i){
				double incr = num_in_queue[i + 1] * cost_rate[i] * dt;
				waiting_cost[i]            += incr;
				discounted_waiting_cost[i] += incr * discount;
				total_cost                 += incr * discount;
			}

			if (!waiting_cost_reset_done && sim_clock >= steady_state_time) {
				for (int i = 0; i < class_no; ++i){
					waiting_cost[i] = 0.0;
				}
				waiting_cost_reset_done = true;
			}

			// Save metrics to determine the steady state 	
			if (sim_clock >= next_recording_time){

				double avg_queue_length = queue_integral[0] / t_event;
				double avg_system_length = system_integral[0] / t_event;
				system_sizes.push_back(avg_system_length);
				queue_sizes.push_back(avg_queue_length);
				cumulative_costs.push_back(total_cost);
				time_stamps.push_back(sim_clock);
				next_recording_time += recording_interval; // move to the next 5-minute checkpoint
			}
			
			// advance the time
			sim_clock = t_event; // time unit is hourly

            // if the current event is an arrival
			if (t_event == t_arrival) {
				double arrival_seed = uniform01(generator);
				auto low = std::lower_bound(arr_cdf.begin(), arr_cdf.end(), arrival_seed);
				int arrival_ind = static_cast<int>(low - arr_cdf.begin());
                // handle arrival event
            	handle_arrival_event(arrival_ind);
			} 

            else if (min_depart_station != -1 && t_event == t_depart[min_depart_station]) {
				int station_index = min_depart_station;
                double departure_seed = uniform01(generator);

				for (int i = 0; i < class_no; ++i) {
					numerator_tmp[i] = num_in_service[station_index + 1][i + 1] * mu_hourly[i][station_index];
				}
				double service_rate = std::accumulate(numerator_tmp.begin(), numerator_tmp.end(), 0.0);

				double s = 0.0;
				for (int i = 0; i < class_no; ++i) {
					prob_tmp[i] = numerator_tmp[i] / service_rate;
					s += prob_tmp[i];
					ser_cdf_tmp[i] = s;
				}

				auto low = std::lower_bound(ser_cdf_tmp.begin(), ser_cdf_tmp.end(), departure_seed);
				int class_index = low - ser_cdf_tmp.begin();

                handle_depart_event(station_index, class_index); // Adjusted to handle depart based on index
            }
			
			else if (t_event == t_abandon) {
				handle_abandon_event();
			} else {std::cout << "Something is Wrong" << std::endl;}
						
 		}

		std::vector<double> steady_state_expected_cost;
		steady_state_expected_cost.assign(class_no, 0);

		for (int i = 0; i < class_no; i++){
			steady_state_expected_cost[i] = waiting_cost[i] / time_delta;  // c_bar
		}

		std::vector<double> res;
		res.assign(class_no + 1, 0);

		for (int i = 0; i < class_no; i ++){
			res[i] = discounted_waiting_cost[i] + steady_state_expected_cost[i] * std::exp(-gamma * total_time) / gamma;
		}
		
		for (int i = 0; i < class_no; i++) {
			res[class_no] += res[i];
		}
		return res;
	}
}

int main(int argc, char** argv){
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " config.json start_iter end_iter output.csv\n";
		return 1;
    }

    std::string jsonFileName = argv[1];
    int start_iter = std::stoi(argv[2]);
    int end_iter   = std::stoi(argv[3]);   // end is exclusive
    std::string record_file = argv[4];

    simulation::Simulation simObj(jsonFileName);
    simObj.save(start_iter, end_iter, record_file);

    return 0;
}
