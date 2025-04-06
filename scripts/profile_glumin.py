#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import re
from datetime import datetime
import utils

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")[:-3]
print(f"[{current_time}]")

# benchmark_dir = "/mnt/microndisk/home/zlwu/graphmine_bench/glumin_data/datasets/dataset2/"
benchmark_dir = "/data-ssd/home/zhenlin/workspace/graphmining/graphmine_bench/glumin_data/datasets/dataset2/"
datasets = [
    ("mico/graph", "mico"),
    ("youtube/graph", "youtube"),
    ("com-dblp/graph", "com-dblp"),
    ("cit-Patents/graph", "cit-Patents"),
    ("livej/graph", "livej"),
    ("orkut/graph", "orkut"),
    
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
        # for dataset in datasets:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
            with open("../results/g2miner_glumin_memory_profiling.csv", "a") as f:
                f.write(f"{dataset_name},")
                
            
            cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM {dataset_path} {pattern}"
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)
            
            cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM_LUT {dataset_path} {pattern}"
            subprocess.run(cmd, shell=True)
            
            with open("../results/g2miner_glumin_memory_profiling.csv", "a") as f:
                f.write("\n")
            


def test_glumin_workload_distribution():

    cmd = f"cd ../codes/GLUMIN && make all && cd ../../scripts/"
    # os.system(cmd)
            
    for pattern in patterns:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
            with open("../results/work_depth_per_warp_glumin_g2miner.csv", "a") as f:
                f.write(f"{dataset_name},")
                
            cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM {dataset_path} {pattern}"
            subprocess.run(cmd, shell=True)
            
            with open("../results/work_depth_per_warp_glumin_g2miner_lut.csv", "a") as f:
                f.write(f"{dataset_name},")

            cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM_LUT {dataset_path} {pattern}"
            # subprocess.run(cmd, shell=True)
            
            # with open("../results/g2miner_glumin_memory_profiling.csv", "a") as f:
            #     f.write("\n")


def parse_args():            
    """显示交互式菜单"""
    print(f"{utils.Colors.OKBLUE}>> Choose experiment:{utils.Colors.ENDC}")
    print("0 -- test_glumin_memory")
    print("1 -- test_glumin_workload_distribution")
    
    choice = input("Enter Exp. ID: ").strip()
    if (choice == "0"):
        return test_glumin_memory()
    if (choice == "1"):
        return test_glumin_workload_distribution()


if __name__ == "__main__":
    
    args = parse_args()

