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
        "name": "Run GraphPi CPU",
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
        "name": "GLUMIN + G2Miner",
        "params": {
            "algorithm": "glumin_g2miner",
            # "use-graphpi-sched": 1,
        }
    },
    "4": {
        "name": "GLUMIN + G2Miner with LUT",
        "params": {
            "algorithm": "glumin_g2miner_lut",
            # "use-graphpi-sched": 0,
        }
    },
    "5": {
        "name": "GLUMIN + GraphFold",
        "params": {
            "algorithm": "glumin_gf",
        }
    },
    "6": {
        "name": "GLUMIN + GraphFold with LUT",
        "params": {
            "algorithm": "glumin_gf_lut",
        }
    },
    "7": {
        "name": "GLUMIN + AutoMine",
        "params": {
            "algorithm": "glumin_automine",
        }
    },
    "8": {
        "name": "GLUMIN + AutoMine with LUT",
        "params": {
            "algorithm": "glumin_automine_lut",
        }
    },
    "glumin": {
        "name": "Run all in original GLUMIN",
    }
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
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "glumin_g2miner"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 3")

def RUN_TEST4():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "glumin_g2miner_lut"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 4")

def RUN_TEST5():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "glumin_gf"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 5")
    
def RUN_TEST6():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "glumin_gf_lut"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 6")
    
def RUN_TEST7():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "glumin_automine"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 7")
    
def RUN_TEST8():
    cmd = ["python", "scripts/launch_exp.py", "--algorithm", "glumin_automine_lut"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run test 8")


def RUN_TEST_ALL_IN_ORIGINAL_GLUMIN():
    cmd = ["python", "scripts/launch_separate_glumin.py"]
    utils.run_command(cmd, shell=False, error_msg="Failed to run all in original GLUMIN")


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
        elif choice == "5":
            RUN_TEST5()
        elif choice == "6":
            RUN_TEST6()
        elif choice == "7":
            RUN_TEST7()
        elif choice == "8":
            RUN_TEST8()
        elif choice == "glumin":
            RUN_TEST_ALL_IN_ORIGINAL_GLUMIN()
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with code {e.returncode}")
        sys.exit(e.returncode)
    

if __name__ == "__main__":
    testing_suites()
