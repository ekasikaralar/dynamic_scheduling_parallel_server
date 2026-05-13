import json
import munch
import os
import argparse

import torch 
import numpy as np
from torch import nn
import pandas as pd

import equation as eqn
from solver import BSDE_solver
from Net import Y_Net
from Net import Z_Net

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type = str, help = "The path to load json file")
    parser.add_argument('--run_name', type = str, help = 'The name of numerical experiments')
    
    args = parser.parse_args()

    return args


def load_config(args):

    with open(args.config_path) as json_data_file:
        config = json.load(json_data_file)

    return munch.munchify(config)

def prepare_directories(args):

    os.makedirs(f"logs_{args.run_name}", exist_ok = True)
    os.makedirs(f"cppweights_{args.run_name}", exist_ok = True)


def main():
    
    print(torch.__version__)
    args = parse_arguments()
    
    config = load_config(args)
    prepare_directories(args)

    h_net_path = config.eqn_config.h_net_path + "h_network.pth" # The path to get the pretrained network for the supremum term (H function)
    drift_net_path = config.eqn_config.drift_net_path + "drift_network.pth" # The path to get the pretrained network for the drift term

    # Load pretrained networks using paths from the configuration file
    h_net = torch.load(h_net_path, weights_only=False)
    drift_net = torch.load(drift_net_path, weights_only=False)

    # Initialize equation with pretrained networks
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config, drift_net = drift_net, h_net = h_net)
    bsde_solver = BSDE_solver(config, bsde)

    save_path_logs = f"logs_{args.run_name}/"
    save_path_cppweights = f"cppweights_{args.run_name}/"

    training_history = bsde_solver.train(save_path_logs, save_path_cppweights)
    
    np.savetxt(f"{args.run_name}_training_history.csv", training_history, delimiter = ',')

if __name__ == '__main__':
    main()