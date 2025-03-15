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
        help="Dataset name (default: Wiki-Vote)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="cpu_baseline",
        help="Algorithm type (default: cpu_baseline)"
    )
    parser.add_argument(
        "--use-graphpi-sched",
        type=int,
        default=1,
        dest="use-graphpi-sched",
        choices=[0, 1],  # 限制取值为 0 或 1
        help="Enable GraphPi scheduler (1=on, 0=off, default: 1)"
    )
    return parser.parse_args()