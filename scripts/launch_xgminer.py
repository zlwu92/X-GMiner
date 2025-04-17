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
from scipy.stats import skew
import numpy as np
import csv

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")[:-3]
print(f"[{current_time}]")

# benchmark_dir = "/mnt/microndisk/home/zlwu/graphmine_bench/glumin_data/datasets/dataset2/"
benchmark_dir = "/data-ssd/home/zhenlin/workspace/graphmining/graphmine_bench/glumin_data/datasets/dataset2/"
datasets = [
    # ("../testgr1/", "TestGr1"),
    # ("../testgr2/", "TestGr2"),
    # ("../testgr3/", "TestGr3"),
    # ("../testgr4/", "TestGr4"),
    # ("../testgr5/", "TestGr5"),
    ("ba_1k_150k/", "ba_1k"),
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

bucket_num = [
    # 4,
    8,
    # 16,
    # 32,
    # 64,
    # 128,
    # 256,
]


def test_bitmap_opt1():
    
    cmd = f"cd ../build && cmake .. -DBMAP_BUCKET_NUM=8 && make xgminer && cd ../scripts/"
    # cmd = f"cd ../build && make xgminer && cd ../scripts/"
    # os.system(cmd)
            
    for pattern, pattern_id in patterns:
        # for dataset in datasets:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
                
            for bucket_k in bucket_num:
                cmd = f"cd ../build && cmake .. -DBMAP_BUCKET_NUM={bucket_k} && make xgminer -j && cd ../scripts/"
                os.system(cmd)

                cmd = f"../xgminer/bin/xgminer "
                cmd += f"--graph {dataset_path} "
                cmd += f"--dataname {dataset_name} "
                cmd += f"--run-xgminer 1 "
                # cmd += f"--algorithm bitmap_bigset_opt "
                cmd += f"--algorithm ideal_bitmap_test "
                cmd += f"--patternID {pattern_id} "
                cmd += f"--vert-induced 1 "
                # cmd += f"--do-validation 1 "
                print(f"Command: {cmd}")
                subprocess.run(cmd, shell=True)


def test_glumin_workload():
    cmd = f"cd ../build && make xgminer -j && cd ../scripts/"
    os.system(cmd)

    # with open(f"../results/prof_glumin_kernel_workload.csv", "a") as f:
    #     f.write(f"benchmarks,pattern,g2miner+LUT_total,max/min,")
    #     f.write(f"g2miner_edge+total,max/min,g2miner_vert+total,max/min\n")
    for pattern, pattern_id in patterns:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")

            with open(f"../results/prof_glumin_kernel_workload.csv", "a") as f:
                f.write(f"{dataset_name},{pattern},")
            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm glumin_g2miner_lut "
            cmd += f"--patternID {pattern_id} "
            cmd += f"--prof-workload 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)
# '''
            # print("G2Miner+LUT:", end="")
            # with open(f"../results/prof_glumin_kernel_workload_{dataset_name}.csv", "r") as file:
            #     # read every line to a list
            #     lines = file.readlines()
            #     # 去除每行末尾的换行符
            #     lines = [int(line.strip()) for line in lines]
            #     print("max:", max(lines), " min:", min(lines))
            #     with open(f"../results/prof_glumin_kernel_workload.csv", "a") as f:
            #         f.write(f"{sum(lines):.2f},{max(lines):.2f},{min(lines):.2f},")
            #         f.write(f"{(max(lines)/min(lines)):.2f},{np.std(lines):.2f},{skew(lines):.2f},")
                
                

            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm glumin_g2miner "
            cmd += f"--patternID {pattern_id} "
            cmd += f"--prof-workload 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)

            # print("G2Miner:", end="")
            # with open(f"../results/prof_glumin_kernel_workload_{dataset_name}.csv", "r") as file:
            #     # read every line to a list
            #     lines = file.readlines()
            #     # 去除每行末尾的换行符
            #     lines = [int(line.strip()) for line in lines]
            #     print("max:", max(lines), " min:", min(lines))
            #     with open(f"../results/prof_glumin_kernel_workload.csv", "a") as f:
            #         f.write(f"{sum(lines):.2f},{max(lines):.2f},{min(lines):.2f},")
            #         f.write(f"{(max(lines)/min(lines)):.2f},{np.std(lines):.2f},{skew(lines):.2f}\n")


            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm glumin_g2miner "
            cmd += f"--patternID {pattern_id} "
            cmd += f"--prof-workload 1 "
            cmd += f"--use-vert-para 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)
            with open(f"../results/prof_glumin_kernel_workload.csv", "a") as f:
                f.write("\n")
# '''

def test_glumin_kernel_profile():
    cmd = f"cd ../build && make xgminer -j && cd ../scripts/"
    os.system(cmd)

    for pattern, pattern_id in patterns:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")

            # with open(f"../results/prof_glumin_kernel_edgecheck.csv", "a") as f:
            #     f.write(f"{dataset_name},{pattern},")
            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm glumin_g2miner_lut "
            cmd += f"--patternID {pattern_id} "
            # cmd += f"--use-vert-para 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)

            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm glumin_g2miner "
            cmd += f"--patternID {pattern_id} "
            # cmd += f"--use-vert-para 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)


def test_glumin_kernel_edgecheck_redundancy():
    cmd = f"cd ../build && make xgminer -j && cd ../scripts/"
    os.system(cmd)

    for pattern, pattern_id in patterns:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
            
            cmd = f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--algorithm glumin_g2miner_lut "
            cmd += f"--patternID {pattern_id} "
            cmd += f"--prof-edgecheck 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)


def ncu_profile_xgminer_kernel():
    cmd = f"cd ../build && make xgminer -j && cd ../scripts/"
    os.system(cmd)

    for pattern, pattern_id in patterns:
        for dataset, dataset_name in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
            # mkdir dir
            output_dir = f"../results/profile_xgminer/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cmd = f"ncu --set full --import-source on -f -o {output_dir}{dataset_name}_{pattern}  "
            cmd += f"--target-processes all "
            cmd += f"../xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} "
            cmd += f"--dataname {dataset_name} "
            cmd += f"--run-xgminer 1 "
            cmd += f"--algorithm ideal_bitmap_test "
            cmd += f"--patternID {pattern_id} "
            cmd += f"--vert-induced 1 "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)
            
            



def parse_args():            
    """显示交互式菜单"""
    print(f"{utils.Colors.OKBLUE}>> Choose experiment:{utils.Colors.ENDC}")
    print("0 -- test_bitmap_opt1: bigset bitmap optimization")
    print("1 -- test_glumin_workload: glumin workload")
    print("2 -- test_glumin_kernel_profile: glumin kernel profile")
    print("3 -- test_glumin_kernel_edgecheck_redundancy: ")
    print("4 -- ncu_profile_xgminer_kernel: ")

    choice = input("Enter Exp. ID: ").strip()
    if (choice == "0"):
        return test_bitmap_opt1()
    if (choice == "1"):
        return test_glumin_workload()
    if (choice == "2"):
        return test_glumin_kernel_profile()
    if (choice == "3"):
        return test_glumin_kernel_edgecheck_redundancy()
    if (choice == "4"):
        return ncu_profile_xgminer_kernel()


if __name__ == "__main__":
    
    args = parse_args()
    # test_bitmap_opt1()
    # test_glumin_workload()
    # test_glumin_kernel_profile()
    # test_glumin_kernel_edgecheck_redundancy()
    # ncu_profile_xgminer_kernel()

