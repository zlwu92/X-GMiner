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
    # ("../testgr1/", "TestGr1"),
    ("../testgr2/", "TestGr2"),
    # ("mico/", "mico"),
    # ("youtube/", "YT"),
    # ("com-dblp/", "dblp"),
    # ("cit-Patents/", "cp"),
    # ("livej/", "livej"),
    # ("orkut/", "orkut"),
    
]

patterns = [
    # ("P1", 17),
    ("P2", 18),
    # ("P3", 19),
    # ("P4", 20),
    # ("P5", 21),
    # ("P6", 22),
]


def test_bitmap_opt1():
    
    cmd = f"cd ../build && cmake .. && make xgminer && cd ../scripts/"
    os.system(cmd)
            
    for pattern, pattern_id in patterns:
        # for dataset in datasets:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
                
            
            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm bitmap_opt1 "
            cmd += f"--patternID {pattern_id} "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)



def parse_args():            
    """显示交互式菜单"""
    print(f"{utils.Colors.OKBLUE}>> Choose experiment:{utils.Colors.ENDC}")
    print("0 -- test_bitmap_opt1: bitmap bucket compression")
    
    choice = input("Enter Exp. ID: ").strip()
    if (choice == "0"):
        return test_bitmap_opt1()


if __name__ == "__main__":
    
    args = parse_args()

