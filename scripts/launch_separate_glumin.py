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
    
benchmark_dir = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/dataset2/"
datasets = [
    "mico/graph",
    "orkut/graph",
    "livej/graph",
    "youtube/graph",
    "com-dblp/graph",
    "cit-Patents/graph",
]

# from P1 to P24
patterns = [
    # "P1",
    # "P2",
    # "P3",
    # "P4",
    # "P5",
    # "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
    "P13",
    "P14",
    "P15",
    "P16",
    "P17",
    "P18",
    "P19",
    "P20",
    "P21",
    "P22",
    "P23",
    "P24",
]



def launch_glumin_test():
    with open("../results/g2miner_glumin_intersection_time_percentage.csv", "a") as f:
        f.write(f"============ {current_time} ============\n")
    for pattern in patterns:
        with open("../results/g2miner_glumin_intersection_time_percentage.csv", "a") as f:
            f.write(f"{pattern},")
        for dataset in datasets:
            dataset_path = benchmark_dir + dataset
            print(f"Dataset: {dataset_path}")
            with open("../results/g2miner_glumin_intersection_time_percentage.csv", "a") as f:
                f.write(f"{dataset},")
            for i in range(1):
                cmd = f"cd ../codes/GLUMIN && make -j && cd ../../scripts/"
                os.system(cmd)
                cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM {dataset_path} {pattern}"
                print(f"Command: {cmd}")
                result = subprocess.run(
                            cmd, 
                            shell=True, 
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True
                        )
                output = result.stdout
                pattern1 = r"runtime \[[^\]]+\] = ([0-9.]+) sec"
                match = re.search(pattern1, output)
                if match:
                    runtime = float(match.group(1))
                    print(f"Runtime: {runtime} sec")
                else:
                    print("未找到匹配的运行时信息。")
                
                runtime_no = 0
                # cmd = f"cd ../codes/GLUMIN && make CUSTOM_FLAGS=\"-DINTERSECTION\" -j"
                # os.system(cmd)
                # cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM {dataset_path} {pattern}"
                # print(f"Command: {cmd}")
                # result = subprocess.run(
                #             cmd, 
                #             shell=True, 
                #             check=True, 
                #             stdout=subprocess.PIPE, 
                #             stderr=subprocess.PIPE, 
                #             text=True
                #         )
                # output = result.stdout
                # pattern1 = r"runtime \[[^\]]+\] = ([0-9.]+) sec"
                # match = re.search(pattern1, output)
                # if match:
                #     runtime_no = float(match.group(1))
                #     print(f"Runtime: {runtime_no} sec")
                # else:
                #     print("未找到匹配的运行时信息。")
                    
                
                runtime_lut = 0
                runtime_lut_no = 0
                cmd = f"cd ../codes/GLUMIN && make -j"
                os.system(cmd)
                cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM_LUT {dataset_path} {pattern}"
                print(f"Command: {cmd}")
                # os.system(cmd)
                result = subprocess.run(
                            cmd, 
                            shell=True, 
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True
                        )
                output = result.stdout
                pattern1 = r"runtime \[[^\]]+\] = ([0-9.]+) sec"
                match = re.search(pattern1, output)
                if match:
                    runtime_lut = float(match.group(1))
                    print(f"Runtime: {runtime_lut} sec")
                else:
                    print("未找到匹配的运行时信息。")
                
                
                # cmd = f"cd ../codes/GLUMIN && make CUSTOM_FLAGS=\"-DINTERSECTION\" -j"
                # os.system(cmd)
                # cmd = f"../codes/GLUMIN/bin/pattern_gpu_GM_LUT {dataset_path} {pattern}"
                # print(f"Command: {cmd}")
                # result = subprocess.run(
                #             cmd, 
                #             shell=True, 
                #             check=True, 
                #             stdout=subprocess.PIPE, 
                #             stderr=subprocess.PIPE, 
                #             text=True
                #         )
                # output = result.stdout
                # pattern1 = r"runtime \[[^\]]+\] = ([0-9.]+) sec"
                # match = re.search(pattern1, output)
                # if match:
                #     runtime_lut_no = float(match.group(1))
                #     print(f"Runtime: {runtime_lut_no} sec")
                # else:
                #     print("未找到匹配的运行时信息。")

                with open("../results/g2miner_glumin_intersection_time_percentage.csv", "a") as f:
                    f.write(f"{runtime},{runtime_no},{runtime_lut},{runtime_lut_no},")
        with open("../results/g2miner_glumin_intersection_time_percentage.csv", "a") as f:
            f.write("\n")


if __name__ == "__main__":
    
    launch_glumin_test()