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
#include "gurobi_c++.h"
#include <unordered_map>
#include <functional> // For std::hash
#include </project/Call_Center_Control/problem_main/preemptive/benchmark_comparison/new_approach/json.hpp>
#include <cstdint>


namespace simulation
{
	class Simulation
	{
		public: 
			Simulation(const std::string& jsonFileName);
			~Simulation();

			int save(int start_iter, int end_iter, const std::string& record_file);
			std::vector<int> readIndicesFromCSV(const std::string& filename);
			std::vector<std::string> splitString(const std::string& input, char delimiter);
			std::vector<std::vector<double>> readMatrixFromCSV(const std::string& filename); 
			std::vector<double> readVectorFromCSV(const std::string& filename);
			
		private:

			int class_no;
			int station_no;
			int num_iterations;
			double gamma;
			double arrival_rate;   // hourly arrival rate (pulled from config)
			std::string policy;    // "mu", "cmu", "cmu_theta", or "gcmu"

			std::vector<int> no_server;
			std::vector<std::vector<double>> mu_hourly;
			std::vector<double> theta_hourly;
			std::vector<double> cost_rate;
			std::vector<double> arr_cdf; 
			std::vector<std::vector<double>> optimal_xi;
			std::vector<std::vector<double>> edges;

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

	class Execute
	{
		public: 
			
			Execute(int class_no_, int station_no_,
				const std::vector<double>& arr_cdf_,
				const std::vector<std::vector<double>>& mu_hourly_,
				const std::vector<double>& theta_hourly_, 
				const std::vector<int>& no_server_,
				const std::vector<double>& cost_rate_,
				const std::vector<std::vector<double>>& optimal_xi_,
				const std::vector<std::vector<double>>& edges_,
				double arrival_rate_,
				const std::string& policy_,
				int seed);

			~Execute();

			std::vector<double> run(double gamma, int iter);
			const std::vector<std::vector<double>> linear_program(std::vector<int>& num_in_system);


		private:
			
		  	static simulation::BasisLRUCache basis_cache;
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
			void system_arrangement(std::vector<int>& num_in_system);
			std::vector<std::vector<double>> initializeModel(std::vector<int>& num_in_system);
			void save_monitoring_data(const std::string& filename);
			void rearrange_system();

			int choose_station_for_arrival(int cls);
			
			// --- basis cache / edge var flattening ---
			std::vector<std::pair<int,int>> edge_list; // (i,j) with edges==1
			std::vector<GRBVar> edge_vars;             // references copied from d[i][j] after init
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
			
			std::vector<double> arrivals_minus_departures_abandons;
			std::vector<double> system_sizes;
			std::vector<double> queue_sizes;
			std::vector<double> time_stamps;
			std::vector<double> cumulative_costs;
			std::vector<int> server_avail;
			std::vector<int> num_in_system;
			std::vector<std::vector<int>> num_in_service;
			std::vector<int> num_in_queue;
			std::vector<int> num_abandons;
			std::vector<int> num_arrivals;
			std::vector<int> num_departures;
			std::vector<double> queue_integral;
			std::vector<double> service_integral;
			std::vector<double> system_integral;
			std::vector<double> waiting_cost;
			std::vector<double> discounted_waiting_cost;

			std::vector<std::deque<double>> queue_list;
			std::vector<std::vector<double>> abandonment_list;

			std::deque<double> empty_queue;
			std::vector<std::deque<double> > arr_list;
			
			double inf = std::numeric_limits<double>::infinity();
			std::mt19937 generator;

			// Reusable distributions
			std::uniform_real_distribution<double> uniform01;
			double arrival_rate; // stored, coming from Simulation (config file)
			std::string policy;  // "mu", "cmu", "cmu_theta", or "gcmu"
			std::exponential_distribution<double> interarrivalDist;
			std::vector<std::exponential_distribution<double>> abandonDist;  // one per class

			const std::vector<std::vector<double>> mu_hourly;
			const std::vector<double> theta_hourly;
			const std::vector<int> no_server;
			const std::vector<double> cost_rate;
			const std::vector<double> arr_cdf; 
			const std::vector<std::vector<double>> optimal_xi;
			const std::vector<std::vector<double>> edges;
			std::vector<std::vector<double>> optimal_d;		
			std::vector<double> numerator_tmp;
			std::vector<double> prob_tmp;
			std::vector<double> ser_cdf_tmp;
			std::vector<double> next_abandon_time; // size = class_no, next abandon per class

			std::vector<int> num_served_buf;
			std::vector<int> optimal_queue_len_buf;
			std::vector<int> diff_buf;
			std::vector<std::vector<double>> A_matrix; // Pre-computed objective matrix

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
