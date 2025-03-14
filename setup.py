#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import re

# 项目根目录（根据 env.sh 动态获取）
env_sh_path = os.path.abspath("env.sh")

PROJECT_ROOT = Path(__file__).resolve().parent
print(f"Project root: {PROJECT_ROOT}")

# 配置参数
BUILD_DIR = PROJECT_ROOT / "build"
THIRD_PARTY_DIR = PROJECT_ROOT / "codes"
XGMINER_DIR = PROJECT_ROOT / "xgminer"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def run_command(cmd, cwd=None, env=None, error_msg="Command failed"):
    """执行 shell 命令并检查错误"""
    print(f"{Colors.OKBLUE}>> {cmd}{Colors.ENDC}")
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=cwd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
        sys.exit(e.returncode)

def setup_environment():
    """设置环境变量"""
    # env_script = PROJECT_ROOT / "../env.sh"
    # if not env_script.exists():
    #     print(f"{Colors.FAIL}env.sh not found in {PROJECT_ROOT}{Colors.ENDC}")
    #     sys.exit(1)
    
    print(f"{Colors.OKGREEN}Setting up environment...{Colors.ENDC}")
    
    # 通过 Bash 子进程加载 env.sh 并导出环境变量
    # 使用 bash -c 执行，确保正确处理路径和命令
    command = ['bash', '-c', f'source "{env_sh_path}" && env']
    output = subprocess.check_output(command, text=True)

    # 将环境变量逐行加载到当前 Python 进程
    for line in output.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value

    # 验证 XGMINER_ROOT 是否设置成功
    print("XGMINER_ROOT:", os.environ.get("XGMINER_ROOT"))

def build_project():
    """编译第三方库 3rd_party"""
    print(f"{Colors.HEADER}Building Third-Party Libraries and XGMiner...{Colors.ENDC}")
    
    cmake_build_root = PROJECT_ROOT / "build"
    if not cmake_build_root.exists():
        cmake_build_root.mkdir(exist_ok=True)
    run_command("cmake ..", cwd=cmake_build_root, error_msg="CMake configure failed")
    
    """编译主项目"""
    print(f"{Colors.HEADER}Building main project...{Colors.ENDC}")
    run_command("make -j", cwd=cmake_build_root, error_msg="Failed to build")

def configure_cmake(build_type="Release"):
    """配置 CMake 项目"""
    print(f"{Colors.HEADER}Configuring CMake project...{Colors.ENDC}")
    BUILD_DIR.mkdir(exist_ok=True)
    
    cmake_args = [
        "cmake",
        # f"-DCMAKE_BUILD_TYPE={build_type}",
        # "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        str(XGMINER_DIR)
    ]
    print(f"{Colors.OKBLUE}{' '.join(cmake_args)}{Colors.ENDC}")
    
    # run_command(" ".join(cmake_args), cwd=BUILD_DIR, error_msg="CMake configure failed")


def run_tests():
    """运行测试"""
    print(f"{Colors.HEADER}Running tests...{Colors.ENDC}")
    test_dir = BUILD_DIR / "test"
    if not test_dir.exists():
        print(f"{Colors.FAIL}Test directory not found!{Colors.ENDC}")
        return
    
    for test_bin in test_dir.iterdir():
        if test_bin.is_file() and test_bin.name.startswith("test_"):
            print(f"{Colors.OKGREEN}Running {test_bin.name}...{Colors.ENDC}")
            run_command(str(test_bin), cwd=test_dir, error_msg=f"{test_bin.name} failed")


def run_xgminer():
    """运行 xgminer"""
    print(f"{Colors.HEADER}Running XGMiner...{Colors.ENDC}")
    # xgminer_bin = XGMINER_DIR / "bin" / "xgminer"
    # if not xgminer_bin.exists():
    #     print(f"{Colors.FAIL}xgminer binary not found!{Colors.ENDC}")
    #     return
    
    # cmd = [str(xgminer_bin),
    #     ]
    cmd = "python scripts/launch_exp.py"
    run_command(cmd, error_msg="xgminer failed")


def clean():
    """清理构建文件"""
    print(f"{Colors.HEADER}Cleaning project...{Colors.ENDC}")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    print(f"{Colors.OKGREEN}Clean completed.{Colors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-GMiner Build System")
    subparsers = parser.add_subparsers(dest="command")

    # 子命令: setup
    parser_setup = subparsers.add_parser("setup", help="Setup environment and dependencies")
    parser_setup.add_argument("--build", action="store_true", help="Build the project after setup env")
    parser_setup.add_argument("--xgminer", action="store_true", help="Run xgminer after build")
    
    # 子命令: build
    parser_build = subparsers.add_parser("build", help="Build the project")
    parser_build.add_argument("-j", "--jobs", type=int, default=4, help="Number of parallel jobs")
    parser_build.add_argument("--build-type", default="Release", choices=["Debug", "Release"],
                             help="Build type (default: Release)")
    parser_build.add_argument("--test", action="store_true", help="Run tests after build")

    # 子命令: clean
    parser_clean = subparsers.add_parser("clean", help="Clean build artifacts")

    # 子命令: test
    parser_test = subparsers.add_parser("test", help="Run tests")

    args = parser.parse_args()

    if args.command == "setup":
        setup_environment()
        # build_project()
        if args.build:
            build_project()
            if args.xgminer:
                run_xgminer()
    # elif args.command == "build":
    #     if args.test:
    #         run_tests()
    elif args.command == "clean":
        clean()
    elif args.command == "test":
        run_tests()
    else:
        parser.print_help()
        sys.exit(1)