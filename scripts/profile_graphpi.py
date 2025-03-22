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

benchmark_dir = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/"

datasets = [
    # "mico.txt",
    ("wiki-Vote.txt", "Wiki-Vote"),
    ("cit-Patents.txt", "Patents"),
    ("com-dblp.ungraph.txt", "DBLP"),
    ("com-youtube.ungraph.txt", "YouTube"),
    # ("com-lj.ungraph.txt", "LiveJournal"),
    # ("com-orkut.ungraph.txt", "Orkut"),
]

# Patterns in GLUMIN
patterns = [
    ("P0", 4, "0110100110010110"),
    # ("P1", 4, "0111100010001000")
    ("P2", 4, "0111101111001100"),
    ("P3", 4, "0111101011001000"),
    ("P4", 4, "0111101111011110"),
    ("P5", 5, "0111110111110111110111110"),
    ("P6", 5, "0111110010100011100010100"),

]

VTUNE = "/opt/intel/oneapi/vtune/2025.0/bin64/vtune"

def profile_intersection_in_graphpi():
    # for pattern in patterns:
    with open("../results/prof_graphpi_intersection_time_percentage.csv", "a") as f:
        f.write("Dataset,Pattern,Int,Diff,\n")
    for pattern, pattern_size, pattern_adj_mat in patterns:
        # with open("../results/prof_graphpi_intersection_time_percentage.csv", "a") as f:
        #     f.write(f"{pattern},")
        # f.close()
        # for dataset in datasets:
        for dataset, dataname in datasets:
            dataset_path = benchmark_dir + dataset
            
            print(f"Dataset: {dataset_path}")
            with open("../results/prof_graphpi_intersection_time_percentage.csv", "a") as f:
                f.write(f"{pattern},{dataset},")
            f.close()
            
            report = "graphpi_" + dataname + "_" + pattern
            
            os.system(f"rm -rf {report}/")
            
            cmd = f"{VTUNE} -collect hotspots -knob sampling-mode=hw -start-paused -r {report} --app-working-dir=../results/ "
            cmd += f"/home/wuzhenlin/workspace/2-graphmining/X-GMiner/xgminer/bin/xgminer "
            cmd += f"--graph {dataset_path} --dataname {dataname} "
            cmd += f"--algorithm cpu_baseline --run-graphpi 1 "
            cmd += f"--pattern-size {pattern_size} --pattern-adj-mat {pattern_adj_mat} "
            print(f"Command: {cmd}")
            subprocess.run(cmd, shell=True)
            
            cmd = f"{VTUNE} -R top-down -r {report} > ../results/{report}.txt"
            subprocess.run(cmd, shell=True)
            
            int_time = 0
            diff_time = 0
            with open(f"../results/{report}.txt", "r") as f:
                for line in f:
                    res = re.findall(r"VertexSet::unorderd_subtraction_size\s+([\d\.]+)\%", line)
                    if len(res) > 0:
                        diff_time = float(res[0])
                    res = re.findall(r"VertexSet::intersection\s+([\d\.]+)\%", line)
                    if len(res) > 0:
                        int_time = float(res[0])
                        
            with open("../results/prof_graphpi_intersection_time_percentage.csv", "a") as file:
                file.write(f"{int_time},{diff_time},\n")
        
                

if __name__ == "__main__":
    
    profile_intersection_in_graphpi()