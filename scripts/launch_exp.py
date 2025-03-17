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
import itertools
import arg_parser as ap

PROJECT_ROOT = Path(os.environ.get("XGMINER_ROOT")).resolve()
xgminer_bin = PROJECT_ROOT / "bin" / "xgminer"
print(f"XGMiner binary: {xgminer_bin}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class ExperimentRunner:
    """实验执行核心类"""
    def __init__(self, config: Dict, args: Namespace):
        # 验证配置
        # required_fields = ["executable", "input_dir", "output_dir", "params"]
        required_fields = ["executable"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        self.executable = Path(config["executable"]).resolve()
        
        # self.input_dir = Path(config["input_dir"]).resolve()
        if "input_dir" in config:
            self.input_dir = Path(config["input_dir"]).resolve()
        else:
            self.input_dir = ""
        
        if "output_dir" in config:
            self.output_dir = Path(config["output_dir"]).resolve()
        else:
            self.output_dir = PROJECT_ROOT / "../results"
        
        # self.params = config["params"]
        # self.params = config.get("params", {})  # 允许空参数
        # assign key and value in args to self.params
        self.params = {k: v for k, v in vars(args).items() if k not in ["executable", "input_dir", "output_dir"]}
        self.timeout = 3600 # config.get("timeout", 300)  # 默认超时300秒
        # print("self.params:", self.params)
        # 检查可执行文件是否存在
        if not self.executable.exists():
            raise FileNotFoundError(f"Executable not found: {self.executable}")
        
        # 自动创建输出目录
        # self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
    def generate_param_combinations(self) -> list:
        """生成所有参数组合（笛卡尔积）"""
        if not self.params:
            return [{}]  # 返回空参数集
        
        return [self.params]
    
    def run_experiment(self, params: Dict[str, Any]) -> Path:
        """运行单个实验"""
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = self.output_dir / f"run_{timestamp}"
        # output_subdir.mkdir(parents=True, exist_ok=True)
        print(output_subdir)
        
        # 构建命令行参数
        args = " ".join([f"--{k} {v}" for k, v in params.items()])
        if self.input_dir:
            input_files = " ".join([str(f) for f in self.input_dir.glob("*") if f.is_file()])
        else:
            input_files = ""
        # 执行命令
        log_file = output_subdir / "output.log"
        # cmd = f"{self.executable} {input_files} {args} > {log_file} 2>&1"
        cmd = f"{self.executable} {input_files} {args} "
        logging.info(f"Executing: {cmd}")
        
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=self.output_dir, timeout=self.timeout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Experiment failed with code {e.returncode}")
            sys.exit(e.returncode)
        except subprocess.TimeoutExpired as e:
            logging.error(f"Experiment timed out after {self.timeout} seconds")
            # 强制终止进程组（防止子进程残留）
            if e.cmd:
                logging.warning("Terminating child processes...")
                subprocess.run(f"pkill -P {e.pid}", shell=True, check=False)
            sys.exit(1)
        
        return output_subdir
    
    
    def process_results(self, output_dir: Path):
        """结果处理（示例）"""
        # 这里可添加结果解析逻辑
        logging.info(f"Results saved to: {output_dir}")
        

def launch_exp():
    args = ap.parse_args()
    
    # 加载配置文件
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except Exception as e:
        # logging.error(f"Failed to load config: {e}")
        # sys.exit(1)
        config = {}

    # print args key and value
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")
    # 覆盖配置（如果命令行指定了参数）
    if args.executable:
        config["executable"] = str(args.executable.resolve())
    else:
        config["executable"] = str(xgminer_bin.resolve())
    # if args.input_dir:
    #     config["input_dir"] = str(args.input_dir.resolve())
    # config["output_dir"] = str(args.output_dir.resolve())
    
    # args.__dict__["pattern-size"] = 3
    # del args.__dict__["pattern_size"]
    print(args)
    # 初始化运行器
    runner = ExperimentRunner(config, args=args)

    # print(runner.generate_param_combinations())
    # 运行所有参数组合
    for param_set in runner.generate_param_combinations():
        output_dir = runner.run_experiment(param_set)
        # runner.process_results(output_dir)

if __name__ == "__main__":
    launch_exp()
