#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import re

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


datasets = {
    "Wiki-Vote": "/home/wuzhenlin/workspace/2-graphmining/X-GMiner/codes/GraphPi/dataset/wiki-vote_input",
    "TestGr1": "/home/wuzhenlin/workspace/2-graphmining/X-GMiner/scripts/test_gr1.txt",
    "TestGr2": "/home/wuzhenlin/workspace/2-graphmining/X-GMiner/scripts/test_gr2.txt",
    "TestGr2_bin": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/testgr2/",
    "TestGr1_bin": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/testgr1/",
    "mico_bin" : "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/mico/",
    "MiCo": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/mico.txt",
    "Patents": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/cit-Patents.txt",
    "DBLP": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/com-dblp.ungraph.txt",
    "LiveJournal": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/com-lj.ungraph.txt",
    "YouTube": "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/com-youtube.ungraph.txt",
}


patterns = {
    "P1", # Triangle
    "P2", # Rectangle
}



def run_command(cmd, shell=True, cwd=None, env=None, error_msg="Command failed"):
    """执行 shell 命令并检查错误"""
    print(f"{Colors.OKBLUE}>> {cmd}{Colors.ENDC}")
    try:
        subprocess.run(
            cmd,
            shell=shell,
            check=True,
            cwd=cwd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
        sys.exit(e.returncode)