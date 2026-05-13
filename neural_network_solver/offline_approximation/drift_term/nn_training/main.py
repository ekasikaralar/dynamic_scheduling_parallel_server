import json
import munch
import os
import argparse

import torch 
import numpy as np
from torch import nn
import pandas as pd

from DriftNet import DriftNet

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

def main():
    
    print(torch.__version__)
    args = parse_arguments()
    
    config = load_config(args)
    prepare_directories(args)
    
    drift_net_model = DriftNet(config)

    save_path_logs = f"logs_{args.run_name}/"

    training_history = drift_net_model.train(save_path_logs)
    
    np.savetxt(f"{args.run_name}_training_history.csv", training_history, delimiter = ',')

if __name__ == '__main__':
    main()