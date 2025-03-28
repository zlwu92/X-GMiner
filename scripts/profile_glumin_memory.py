#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import re
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")[:-3]
print(f"[{current_time}]")

benchmark_dir = "/mnt/microndisk/home/zlwu/graphmine_bench/glumin_data/datasets/dataset2/"
datasets = [
    "mico/graph",
    "orkut/graph",
    "livej/graph",
    "youtube/graph",
    "com-dblp/graph",
    "cit-Patents/graph",
]

patterns = [
    # "P1",
    "P2",
    # "P3",
    # "P4",
    # "P5",
    # "P6",
]


def test_glumin_memory():
    
    cmd = f"cd ../codes/GLUMIN && make all && cd ../../scripts/"
    os.system(cmd)
            
    for pattern in patterns:
        for dataset in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
            with open("../results/g2miner_glumin_memory_profiling.csv", "a") as f:
                f.write(f"{dataset},")
                
            
            cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM {dataset_path} {pattern}"
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)
            
            cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM_LUT {dataset_path} {pattern}"
            subprocess.run(cmd, shell=True)
            
            with open("../results/g2miner_glumin_memory_profiling.csv", "a") as f:
                f.write("\n")
            
            
            
    


if __name__ == "__main__":
    
    test_glumin_memory()

