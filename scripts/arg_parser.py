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
from argparse import ArgumentParser, Namespace
import utils

PROJECT_ROOT = Path(os.environ.get("XGMINER_ROOT")).resolve()
print(f"XGMiner root: {PROJECT_ROOT}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run XGMiner Experiments")
    # parser.add_argument(
    #     "-c", "--config",
        # required=True,
    #     type=Path,
    #     help="Path to JSON config file"
    # )
    parser.add_argument(
        "-e", "--executable",
        type=Path,
        help="Override executable path (if not in default build location)"
    )
    parser.add_argument(
        "--graph",
        type=Path,
        # default=Path("/home/wuzhenlin/workspace/2-graphmining/X-GMiner/codes/GraphPi/dataset/wiki-vote_input"),
        default=Path("/home/wuzhenlin/workspace/2-graphmining/X-GMiner/scripts/test_gr2.txt"),
        # required=True,
        help="Override input directory"
    )
    # parser.add_argument(
    #     "--output-dir",
    #     type=Path,
    #     default=Path("results/"),
    #     help="Override output directory (default: ./results)"
    # )
    # parser.add_argument(
    #     "--timeout",
    #     type=int,
    #     default=300,
    #     help="Experiment timeout in seconds (default: 300)"
    # )
    parser.add_argument(
        "--dataname",
        type=str,
        default="Wiki-Vote",
        # required=True,
        help="Dataset name (default: Wiki-Vote)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="cpu_baseline",
        # required=True,
        help="Algorithm type (default: cpu_baseline)"
    )
    parser.add_argument(
        "--use-graphpi-sched",
        type=int,
        default=0,
        dest="use-graphpi-sched",
        choices=[0, 1],  # 限制取值为 0 或 1
        help="Enable GraphPi scheduler (1=on, 0=off, default: 1)"
    )
    parser.add_argument(
        "--run-our-baseline",
        type=int,
        default=0,
        dest="run-our-baseline",
        choices=[0, 1],  # 限制取值为 0 或 1
        help="Run our baseline (1=on, 0=off, default: 1)"
    )
    parser.add_argument(
        "--run-graphpi",
        type=int,
        default=0,
        dest="run-graphpi",
        choices=[0, 1],  # 限制取值为 0 或 1
        help="Run GraphPi (1=on, 0=off, default: 1)"
    )
    parser.add_argument(
        "--pattern-size",
        type=int,
        default=3,
        dest="pattern-size",
        help="Pattern size (default: 3 for triangle)"
    )
    parser.add_argument(
        "--pattern-adj-mat",
        type=str,
        default="011101110",
        dest="pattern-adj-mat",
        help="Pattern adjacency matrix (default: 011101110 for triangle)"
    )
    parser.add_argument(
        "--patternID",
        type=int,
        default=1,
        dest="patternID",
        help="Pattern ID (default: 1 for triangle)"
    )
    parser.add_argument(
        "--do-validation",
        type=int,
        default=1,
        # required=True,
        dest="do-validation",
        help="Pattern ID (default: 1 for triangle)"
    )
    
    """显示交互式菜单"""
    print(f"{utils.Colors.OKBLUE}>> Choose input dataset:{utils.Colors.ENDC}")
    print("0 -- TestGr1")
    print("1 -- TestGr2")
    print("2 -- Wiki-Vote")
    print("3 -- Mico-Bin")
    print("4 -- TestGr1_bin")
    print("5 -- TestGr2_bin")
    print("6 -- MiCo")
    print("7 -- Patents")
    print("8 -- DBLP")
    print("9 -- LiveJournal")
    print("10 -- YouTube")
    choice = input("Enter Dataset ID: ").strip()
    if (choice == "0"):
        parser.set_defaults(graph=Path(f"{utils.datasets['TestGr1']}"))
        parser.set_defaults(dataname="TestGr1")
    if (choice == "1"):
        parser.set_defaults(graph=Path(f"{utils.datasets['TestGr2']}"))
        parser.set_defaults(dataname="TestGr2")
    if (choice == "2"):
        parser.set_defaults(graph=Path(f"{utils.datasets['Wiki-Vote']}"))
        parser.set_defaults(dataname="Wiki-Vote")
    if (choice == "3"):
        parser.set_defaults(graph=Path(f"{utils.datasets['mico_bin']}"))
        parser.set_defaults(dataname="Mico-Bin")
    if (choice == "4"):
        parser.set_defaults(graph=Path(f"{utils.datasets['TestGr1_bin']}"))
        parser.set_defaults(dataname="TestGr1_bin")
    if (choice == "5"):
        parser.set_defaults(graph=Path(f"{utils.datasets['TestGr2_bin']}"))
        parser.set_defaults(dataname="TestGr2_bin")
    if (choice == "6"):
        parser.set_defaults(graph=Path(f"{utils.datasets['MiCo']}"))
        parser.set_defaults(dataname="MiCo")
    if (choice == "7"):
        parser.set_defaults(graph=Path(f"{utils.datasets['Patents']}"))
        parser.set_defaults(dataname="Patents")
    if (choice == "8"):
        parser.set_defaults(graph=Path(f"{utils.datasets['DBLP']}"))
        parser.set_defaults(dataname="DBLP")
    if (choice == "9"):
        parser.set_defaults(graph=Path(f"{utils.datasets['LiveJournal']}"))
        parser.set_defaults(dataname="LiveJournal")
    if (choice == "10"):
        parser.set_defaults(graph=Path(f"{utils.datasets['YouTube']}"))
        parser.set_defaults(dataname="YouTube")
        
    print(f"{utils.Colors.OKBLUE}>> Choose input pattern:{utils.Colors.ENDC}")
    print("1 -- Triangle (size = 3)")
    print("2 -- Rectangle (size = 4)")
    print("3 -- Diamond (size = 4)")
    print(" For GLUMIN (>16)")
    print("17 -- P1 in GLUMIN (size = 4)")
    print("19 -- P3 in GLUMIN (size = 4)")
    choice = input("Enter Pattern ID: ").strip()
    args = parser.parse_args()
    if (choice == "1"):
        setattr(args, "pattern-size", 3)  # 动态设置 pattern-size
        setattr(args, "pattern-adj-mat", "011101110")
        setattr(args, "patternID", 1)
    if (choice == "2"):
        setattr(args, "pattern-size", 4)  # 动态设置 pattern-size
        custom_adj_mat = input("Define a pattern-adj-mat (Default is permitted): ").strip()
        if custom_adj_mat == "":
            setattr(args, "pattern-adj-mat", "0110100110010110")
        else:
            setattr(args, "pattern-adj-mat", custom_adj_mat)
        setattr(args, "patternID", 2)
    if (choice == "17"):
        setattr(args, "pattern-size", 4)  # 动态设置 pattern-size
        setattr(args, "pattern-adj-mat", "0111100010001000")
        setattr(args, "patternID", 17)
    if (choice == "19"):
        setattr(args, "pattern-size", 4)
        setattr(args, "pattern-adj-mat", "0111101011001000")
        setattr(args, "patternID", 19)
    # return parser.parse_args()
    return args
