#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import re
from datetime import datetime
from easypyplot import pdf, barchart
from easypyplot import format as fmt
import torch
from sympy import symbols
from sympy.printing.pretty import pretty
from IPython.display import display, Markdown
from itertools import combinations

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算 utils.py 的绝对路径
scripts_dir = os.path.abspath(os.path.join(current_dir, "../../../scripts"))

# 将 scripts 目录添加到 sys.path
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# 导入 utils 模块
import utils


current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")[:-3]
print(f"[{current_time}]")

benchmark_dir = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/"

datasets = [
    # "mico.txt",
    ("synthetic/test_gr1.txt", "TestGr1"),
    ("synthetic/test_gr2.txt", "TestGr2"),
    # ("wiki-Vote.txt", "Wiki-Vote"),
    # ("cit-Patents.txt", "Patents"),
    # ("com-dblp.ungraph.txt", "DBLP"),
    # ("com-youtube.ungraph.txt", "YouTube"),
    # ("com-lj.ungraph.txt", "LiveJournal"),
    # ("com-orkut.ungraph.txt", "Orkut"),
]

patterns = [
    ("P0", 4, "0110100110010110"),
    # ("P1", 4, "0111100010001000"),
    # ("P2", 4, "0111101111001100"),
    # ("P3", 4, "0111101011001000"),
    # ("P4", 4, "0111101111011110"),
    # ("P5", 5, "0111110111110111110111110"),
    # ("P6", 5, "0111110010100011100010100"),

]


class MatrixTest:
    test_pattern_dir = Path("../input_patterns/")
    test_dataset_dir = Path("../input_datasets/")
    datasets = []
    patterns = []
    restrictions = []
    reference_res = []
    
    def __init__(self, *args, **kwargs):
        # super(CLASS_NAME, self).__init__(*args, **kwargs)
        # self.dataset = kwargs.get("dataset")
        # self.pattern = kwargs.get("pattern")
        pass
    
    
    def dump_pattern_schedule_and_restrictions_from_graphpi(self):
        cmd = "cd ../../../build/ && make xgminer -j "
        os.system(cmd)
        
        for dataset, dataname in datasets:
            dataset_path = benchmark_dir + dataset
            for pattern, pattern_size, pattern_adj_mat in patterns:
                print(f"Dataset: {dataname}")
                
                cmd = f"/home/wuzhenlin/workspace/2-graphmining/X-GMiner/xgminer/bin/xgminer "
                cmd += f"--graph {dataset_path} --dataname {dataname} "
                cmd += f"--algorithm cpu_baseline --run-graphpi 1 "
                cmd += f"--pattern-size {pattern_size} --pattern-adj-mat {pattern_adj_mat} "
                print(f"Command: {cmd}")
                # subprocess.run(cmd, shell=True)
                result = subprocess.run(
                            cmd, 
                            shell=True, 
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True
                        )
                output = result.stdout
                pat = r"Current schedule:\s*([01]+)"
                matches = re.findall(pat, output)
                # pattern.append(matches[0])
                print(f"Pattern: {matches}")
                self.patterns.append(matches[0])
                
                # 定义正则表达式，匹配形如 "数字 (x,y)(x,y)..." 的行
                pat = r"^\s*(\d+)\s*((?:\(\d+,\d+\))+)"
                lines = output.splitlines()  # 按行分割输出
            
                # 遍历每一行，提取 pairs 并存入 res 列表
                for line in lines:
                    match = re.match(pat, line)
                    if match:
                        # 提取数字部分（可选）
                        number = int(match.group(1))
                        
                        # 提取所有 (x,y) 对
                        pairs_str = match.group(2)  # 如 "(1,2)(0,3)(0,1)"
                        pairs_pattern = r"\((\d+),(\d+)\)"  # 匹配单个 (x,y)
                        pairs = [(int(x), int(y)) for x, y in re.findall(pairs_pattern, pairs_str)]
                        
                        print(f"Pairs: {pairs}")
                        self.restrictions.append(pairs)

                match = re.search(r'ans (\d+)', output)
                if match:
                    self.reference_res.append(int(match.group(1)))
                    print(f"{utils.Colors.OKGREEN}Reference result: {match.group(1)}{utils.Colors.ENDC}")
    
    
    def test_different_patterns(self):
        print("Testing different patterns")
    #     print(f"Dataset: {self.dataset}")
    #     print(f"Pattern: {self.pattern}")
    #     print(f"Restriction: {self.restriction}")
        count = 0
        for dataset, dataname in datasets:
            dataset_path = benchmark_dir + dataset
            for pattern, pattern_size, pattern_adj_mat in patterns:
                print(f"{utils.Colors.OKBLUE}Dataset: {dataname}{utils.Colors.ENDC}")
                print(f"{utils.Colors.OKBLUE}Pattern: {pattern}{utils.Colors.ENDC}")
                if pattern == "P0":
                    self.TEST_P0(dataset_path, count)
                count += 1


    def build_adjacency_matrix(self, file_path):
        """
        从文件中读取图的连接信息并构建邻接矩阵。
        
        :param file_path: 文件路径
        :return: 邻接矩阵 (torch.Tensor)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 第一行包含顶点数和边数
        n_vertices, n_edges = map(int, lines[0].strip().split())

        # 初始化邻接矩阵 (n_vertices x n_vertices)，初始值为 0
        self.adj_matrix = torch.zeros((n_vertices, n_vertices), dtype=torch.float32)

        # 遍历后续行，解析每条边
        for line in lines[1:]:
            u, v = map(int, line.strip().split())
            self.adj_matrix[u][v] = 1  # 设置边 u -> v
            self.adj_matrix[v][u] = 1  # 如果是无向图，设置边 v -> u

        return self.adj_matrix


    def superscript(self, num):
        sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return str(num).translate(sup)

    # 使用 Unicode 上标字符
    superscript_map = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    }
    
    def validation(self, res, idx):
        if res == self.reference_res[idx]:
            print(f"{utils.Colors.OKGREEN}Validation passed!{utils.Colors.ENDC}")
        else:
            print(f"{utils.Colors.FAIL}Validation failed!{utils.Colors.ENDC}")
            print(f"Expected: {self.reference_res[idx]}, Got: {res}")
            sys.exit(1)
            

    def TEST_P0(self, dataset_path, idx):
        print(f"Loading input graph: {dataset_path}")

        self.adj_matrix = self.build_adjacency_matrix(dataset_path)
        print(f"Adjacency matrix: {self.adj_matrix}")
        
        # matrix multiplication
        M_super2 = torch.matmul(self.adj_matrix, self.adj_matrix)
        # M_super2 = self_adj_matrix @ self_adj_matrix
        # elementwise_result = self_adj_matrix * self_adj_matrix
        
        print(f"M{self.superscript(2)} = {M_super2}")
        
        # 找到所有非对角元素中大于1的元素
        non_diagonal_elements = M_super2.flatten()[torch.eye(M_super2.size(0)).flatten() == 0]  # 获取非对角元素
        non_diagonal_elements_gt1 = non_diagonal_elements[non_diagonal_elements > 1]  # 获取大于1的元素
        
        # 计算大于1的元素的组合数 C(k, 2) 并求和
        # sum_combinations = 0
        # for element in non_diagonal_elements_gt1:
        #     k = element.item()
        #     # print(f"Element: {k}")
        #     combinations_k2 = len(list(combinations(range(int(k)), 2)))
        #     # combinations_k2 = k * (k - 1) // 2
        #     sum_combinations += combinations_k2
        sum_combinations = sum(len(list(combinations(range(int(val.item())), 2))) for val in non_diagonal_elements_gt1)
        print(f"Sum of C(k, 2): {sum_combinations}")
        
        symmetric_redundancy_num = len(set(num for pair in self.restrictions[idx] for num in pair))
        res = int(sum_combinations / 1)
        print(f"{utils.Colors.OKGREEN}Result: {res}{utils.Colors.ENDC}")
        self.validation(res, idx)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Matrix Operation Tests")
    # parser.add_argument("

    
    mat_test = MatrixTest()
    mat_test.dump_pattern_schedule_and_restrictions_from_graphpi()
    mat_test.test_different_patterns()