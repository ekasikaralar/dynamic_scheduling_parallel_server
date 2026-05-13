#ifndef STATIC_H
#define STATIC_H

#include <set>
#include <iostream>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <numeric>
#include <limits>
#include <list>
#include <vector>
#include <deque>
#include <stdlib.h>
#include <stdio.h>
#include <array>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <cstdio>
#include <sstream>
#include <map>
#include <chrono>
#include <omp.h>
#include <random>
#include <iomanip>
#include "gurobi_c++.h"
#include <unordered_map>
#include <functional> // For std::hash
#include </project/Call_Center_Control/problem_main/preemptive/benchmark_comparison/new_approach/json.hpp>


namespace simulation
{
	class Simulation
	{
		public: 
			Simulation(const std::string& jsonFileName);
			~Simulation();

			int save(int start_iter, int end_iter, const std::string& record_file);

			std::vector<std::string> splitString(const std::string& input, char delimiter);
			std::vector<std::vector<double>> readMatrixFromCSV(const std::string& filename); 
			std::vector<double> readVectorFromCSV(const std::string& filename);
			
		private:

			int class_no;
			int station_no;
			int num_iterations;
			double scaling_factor;
			int num_hidden_layers;
			double leaky_relu_slope;
			double gamma;
			double arrival_rate;   // hourly arrival rate (pulled from config)
			std::string activation_function;
			bool apply_softplus;

			std::vector<int> no_server;
			std::vector<std::vector<double>> mu_hourly;
			std::vector<double> theta_hourly;
			std::vector<double> cost_rate;
			std::vector<double> arr_cdf; 
			std::vector<std::vector<double>> optimal_xi;
			std::vector<double> x_optimal;
			std::vector<std::vector<double>> edges;
			std::string neural_network_folder_name;

	};

	struct BasisEntry {
		// VBasis for edge variables only (size E)
		std::vector<int8_t> v_basis;

		// CBasis for (K class constraints + J station constraints)
		std::vector<int8_t> c_basis;

		// LRU pointer
		std::list<uint64_t>::iterator it;
	};

	class BasisLRUCache {
	public:
		explicit BasisLRUCache(size_t capacity = 10000) : cap_(capacity) {}

		bool get(uint64_t key, std::vector<int8_t>& v_out, std::vector<int8_t>& c_out) {
			auto it = map_.find(key);
			if (it == map_.end()) return false;
			// move to front
			lru_.splice(lru_.begin(), lru_, it->second.it);
			v_out = it->second.v_basis;
			c_out = it->second.c_basis;
			return true;
		}

		void put(uint64_t key, std::vector<int8_t>&& v_in, std::vector<int8_t>&& c_in) {
			auto it = map_.find(key);
			if (it != map_.end()) {
				// update existing, move to front
				it->second.v_basis = std::move(v_in);
				it->second.c_basis = std::move(c_in);
				lru_.splice(lru_.begin(), lru_, it->second.it);
				return;
			}

			// evict if needed
			if (map_.size() >= cap_) {
				uint64_t old = lru_.back();
				lru_.pop_back();
				map_.erase(old);
			}

			lru_.push_front(key);
			BasisEntry e;
			e.v_basis = std::move(v_in);
			e.c_basis = std::move(c_in);
			e.it = lru_.begin();
			map_.emplace(key, std::move(e));
		}

		size_t size() const { return map_.size(); }
		size_t capacity() const { return cap_; }

	private:
		size_t cap_;
		std::list<uint64_t> lru_;
		std::unordered_map<uint64_t, BasisEntry> map_;
	};

	class MyNetwork
	{
	  
	  public:
			MyNetwork() : num_hidden_layers(0), apply_softplus(false) {}
			MyNetwork(int num_hidden_layers_, const std::string& activation_function_, double leaky_relu_slope_, bool apply_softplus_)
				: num_hidden_layers(num_hidden_layers_), activation_function(activation_function_), leaky_relu_slope(leaky_relu_slope_), apply_softplus(apply_softplus_) {
				biases.resize(num_hidden_layers_);
				weights.resize(num_hidden_layers_ + 1);
			}

			~MyNetwork(); 
			
			void load(const std::string& folder_name, const std::string& filename);
			std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix);
				
		private:
	
			int num_hidden_layers;
			std::string activation_function;
			double leaky_relu_slope;
			bool apply_softplus;
		
			std::vector<std::vector<std::vector<float>>> weights;
			std::vector<std::vector<float>> biases;
		
			std::vector<std::vector<float>> add(const std::vector<std::vector<float>>& x, const std::vector<float>& y);
			std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2);
			std::vector<std::vector<float>> divide(const std::vector<std::vector<float> >& x, const std::vector<float>& y);
			std::vector<std::vector<float>> subtract(const std::vector<std::vector<float> >& x, const std::vector<float>& y);
			std::vector<std::vector<float>> multiply(const std::vector<std::vector<float> >& x, const std::vector<float>& y);
			std::vector<std::vector<float>> leaky_relu(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> silu(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> relu(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> gelu(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> selu(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> elu(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> softplus(const std::vector<std::vector<float>>& x);
			std::vector<std::vector<float>> apply_activation(const std::vector<std::vector<float>>& x);

			std::vector<std::vector<float>> readMatrix(const std::string& filename);
			std::vector<float> readVector(const std::string& filename);
	};


	class Execute
	{
		public: 
			
			Execute(int& class_no_,int& station_no_,
							std::vector<double>& arr_cdf_,
							std::vector<std::vector<double>>& mu_hourly_,
							std::vector<double>& theta_hourly_, 
							std::vector<int>& no_server_,
							std::vector<double>& cost_rate_,
							std::vector<double>& x_optimal_,
							std::vector<std::vector<double>>& optimal_xi_,
							std::vector<std::vector<double>>& edges_,
							int& num_hidden_layers_,
							std::string activation_function_,
							double& leaky_relu_slope_,
							bool apply_softplus_,
							int& seed,
							double& scaling_factor_,
							double arrival_rate_);
			~Execute();

			std::vector<double> run(std::string neural_network_folder_name, double gamma, int iter);

			
			std::map<std::vector<int>, std::vector<float>> globalStateMap;
			std::map<std::vector<int>, int> visits;

		private:
			
			GRBEnv env = GRBEnv();
        	GRBModel model = GRBModel(env);
    		std::vector<std::vector<GRBVar>> d;
    		std::vector<GRBConstr> class_constraints;
			std::vector<GRBConstr> station_constraints;

			void queue_init();
			double generate_interarrival();
			double generate_abandon(int& cls);
			double generate_service(int& station);

			void add_queue(double& arr_time, int& cls);
			void remove_queue(int& cls);

			void handle_arrival_event(int& cls);
			void handle_depart_event(int& station, int& cls);
			void handle_abandon_event();
			std::vector<float> load_and_predict(std::vector<std::vector<float> >& input_tensor);
			std::vector<std::vector<double>> linear_program(std::vector<int>& num_in_system, std::vector<float>& gradient);
			std::vector<std::vector<double>> optimal_policy_calculation(std::vector<int>& num_in_system);
			void system_arrangement(std::vector<int>& num_in_system);
			std::vector<std::vector<double>> initializeModel(std::vector<int>& num_in_system);
			void save_monitoring_data(const std::string& filename);
			void rearrange_system();
			
			int choose_station_for_arrival(int cls);
			
			// --- basis cache / edge var flattening ---
			std::vector<std::pair<int,int>> edge_list; // (i,j) with edges==1
			std::vector<GRBVar> edge_vars;             // references copied from d[i][j] after init
			static BasisLRUCache basis_cache;
			static std::unordered_map<uint64_t, std::vector<std::vector<double>>> policy_cache; // policy cache
			size_t cache_capacity = 10000;             // tune

			double t_arrival;
			std::vector<double> t_depart;
			double t_abandon;
			double t_event;
			double sim_clock = 0;
            double T;
            double total_cost;
			double scaling_factor;

			int class_abandon;
			int cust_abandon;
			int station_departure;
			int class_departure;
			int class_no;
			int station_no;
			int total_servers;

			std::vector<double> system_sizes;
			std::vector<double> queue_sizes;
			std::vector<double> time_stamps;
			std::vector<double> cumulative_costs;
			std::vector<int> num_in_system;
			std::vector<std::vector<int>> num_in_service;
			std::vector<int> num_in_queue;
			std::vector<int> num_abandons;
			std::vector<int> num_arrivals;
			std::vector<int> num_departures;
			std::vector<int> server_avail;
			std::vector<double> queue_integral;
			std::vector<double> service_integral;
			std::vector<double> system_integral;
			std::vector<double> waiting_cost;
			std::vector<double> discounted_waiting_cost;

			std::vector<std::deque<double>> queue_list;
			std::vector<std::vector<double>> abandonment_list;

			std::deque<double> empty_queue;
			std::vector<std::deque<double> > arr_list;
			std::vector<double> next_abandon_time; // size = class_no, next abandon per class

			
			double inf = std::numeric_limits<double>::infinity();
			std::mt19937 generator;
			// Reusable distributions
			std::uniform_real_distribution<double> uniform01;
			double arrival_rate; // store this instead of hardcoding in generate_interarrival
			std::exponential_distribution<double> interarrivalDist;
			std::vector<std::exponential_distribution<double>> abandonDist;  // one per class


			std::vector<std::vector<double>> mu_hourly;
			std::vector<double> theta_hourly;
			std::vector<int> no_server;
			std::vector<double> cost_rate;
			std::vector<double> arr_cdf; 
			std::vector<std::vector<double>> optimal_xi;
			std::vector<double> x_optimal;
			std::vector<std::vector<double>> edges;
			std::vector<std::vector<double>> optimal_d;
			std::vector<std::vector<double>> A_matrix;
			std::string activation_function;
			int num_hidden_layers;
			double leaky_relu_slope;
			bool apply_softplus;
			std::vector<double> numerator_tmp;
			std::vector<double> prob_tmp;
			std::vector<double> ser_cdf_tmp;
			
			std::vector<int> num_served_buf;
			std::vector<int> optimal_queue_len_buf;
			std::vector<int> diff_buf;
			std::vector<float> gradient;
			std::vector<float> current_gradient;
			std::vector<float> scaled_x_buf;
			std::vector<std::vector<float>> input_tensor_buf;
			simulation::MyNetwork* nn_zs = nullptr; // Declare as a pointer

			static inline uint64_t fp_state(const std::vector<int>& num_in_system, int K) {
				uint64_t h = 1469598103934665603ULL;
				for (int i = 0; i < K; ++i) {
					uint8_t b = static_cast<uint8_t>(num_in_system[i+1]); // Back to exact states
					h ^= (uint64_t)b;
					h *= 1099511628211ULL;
				}
				return h;
			}

			
	};
};
#endif