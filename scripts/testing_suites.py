#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import utils

# 定义测试套件
TEST_SUITES = {
    "0": {
        "name": "Run GraphPi Baseline",
        "params": {
            "algorithm": "cpu_baseline",
            # "use-graphpi-sched": 1
            "run-graphpi": 1
        }
    },
    "1": {
        "name": "Our CPU Baseline (GraphPi sched)",
        "params": {
            "algorithm": "cpu_baseline",
            "use-graphpi-sched": 1,
            "run-our-baseline": 1
        }
    },
    "2": {
        "name": "Our CPU Baseline",
        "params": {
            "algorithm": "cpu_baseline",
            "use-graphpi-sched": 0,
            "run-our-baseline": 1
        }
    },
    "3": {
        "name": "GPU Baseline (GraphPi sched)",
        "params": {
            "algorithm": "gpu_optimized",
            "use-graphpi-sched": 1,
        }
    },
    "4": {
        "name": "GPU Baseline",
        "params": {
            "algorithm": "gpu_optimized",
            "use-graphpi-sched": 0,
        }
    },
}

# def load_config() -> dict:
#     """加载默认配置文件（如果存在）"""
#     default_config = Path("default_config.json")
#     if default_config.exists():
#         with open(default_config, "r") as f:
#             return json.load(f)
#     return {
#         "executable": "build/xgminer",
#         "input_dir": "data/",
#         "output_dir": "results/",
#         "params": {}
#     }


def RUN_TEST0():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "cpu_baseline", "--run-graphpi", "1"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 0")

# def RUN_TEST1(config: dict):
def RUN_TEST1():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "cpu_baseline", 
           "--use-graphpi-sched", "1", "--run-our-baseline", "1"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 1")

def RUN_TEST2():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "cpu_baseline", 
           "--use-graphpi-sched", "0", "--run-our-baseline", "1"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 2")
    
def RUN_TEST3():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "gpu_baseline", "--use-graphpi-sched", "1"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 3")

def RUN_TEST4():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "gpu_baseline", "--use-graphpi-sched", "0"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 4")


def show_menu():
    """显示交互式菜单"""
    print(f"{utils.Colors.OKBLUE}Which experiment do you want to launch?{utils.Colors.ENDC}")
    for key, suite in TEST_SUITES.items():
        print(f"{key} -- {suite['name']}")
    choice = input("Enter Testing ID: ").strip()
    return choice


def testing_suites():
    # 显示菜单并获取选择
    choice = show_menu()
    while choice not in TEST_SUITES:
        print(f"Invalid choice: {choice}")
        sys.exit(1)

    # 加载默认配置
    # config = load_config()
    # config_file = input("Enter config file path (optional): ").strip()
    # if config_file:
    #     config["config_file"] = config_file
    
    # 根据选择执行测试
    try:
        if choice == "0":
            RUN_TEST0()
        if choice == "1":
            RUN_TEST1()
        elif choice == "2":
            RUN_TEST2()
        elif choice == "3":
            RUN_TEST3()
        elif choice == "4":
            RUN_TEST4()
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with code {e.returncode}")
        sys.exit(e.returncode)
    

if __name__ == "__main__":
    testing_suites()