import struct
import os
import sys

def read_graph_from_txt(file_path):
    """
    读取图的txt文件, 返回顶点数、边数和边列表。
    """
    edges = []
    with open(file_path, 'r') as f:
        # 第一行是顶点数和边数
        n_vertices, n_edges = map(int, f.readline().strip().split())
        # 剩余行是边信息
        for line in f:
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return n_vertices, n_edges, edges


def compute_csr_format(n_vertices, edges):
    """
    计算CSR格式的offset数组和edge数组。
    """
    # 初始化邻接表
    adjacency_list = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)  # 如果是无向图，需要双向添加

    # 按照升序排序每个顶点的邻居
    for neighbors in adjacency_list:
        neighbors.sort()

    # 构造CSR格式
    offset = [0]  # offset数组
    edge_array = []  # edge数组
    for neighbors in adjacency_list:
        offset.append(offset[-1] + len(neighbors))
        edge_array.extend(neighbors)

    # 最大度数
    max_degree = max(len(neighbors) for neighbors in adjacency_list)

    return offset, edge_array, max_degree


def write_bin_files(vertex_offsets, edge_array, output_dir):
    """
    写入vertex.bin和edge.bin文件到指定目录。
    """
    vertex_file = os.path.join(output_dir, "graph.vertex.bin")
    edge_file = os.path.join(output_dir, "graph.edge.bin")

    # 写入vertex.bin (int64)
    with open(vertex_file, 'wb') as f:
        for offset in vertex_offsets:
            # print("offset:", offset)
            f.write(struct.pack('<q', offset))  # 使用小端格式写入int64

    # 写入edge.bin (int32)
    with open(edge_file, 'wb') as f:
        for edge in edge_array:
            # print("edge:", edge)
            f.write(struct.pack('<i', edge))  # 使用小端格式写入int32


def write_meta_file(n_vertices, n_edges, max_degree, output_dir):
    """
    写入meta.txt文件到指定目录。
    """
    meta_file = os.path.join(output_dir, "graph.meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"{n_vertices}\n")
        f.write(f"{n_edges*2}\n")
        f.write("4 8 1 2\n")
        f.write(f"{max_degree}\n")
        f.write("0\n")
        f.write("0\n")
        f.write("0\n")


def convert_graph(input_file, output_dir):
    """
    主函数: 将输入的txt文件转换为所需的bin和txt文件, 并保存到指定目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取图数据
    n_vertices, n_edges, edges = read_graph_from_txt(input_file)

    # 计算CSR格式
    vertex_offsets, edge_array, max_degree = compute_csr_format(n_vertices, edges)

    # 写入bin文件
    write_bin_files(vertex_offsets, edge_array, output_dir)

    # 写入meta文件
    write_meta_file(n_vertices, n_edges, max_degree, output_dir)



def read_meta_txt(meta_file):
    """
    读取 meta.txt 文件，返回顶点数、边数和最大度数。
    """
    # with open(meta_file, 'r') as f:
    #     n_vertices = int(f.readline().strip())
    #     n_edges = int(f.readline().strip())
    #     max_degree = int(f.readline().strip())
    max_degree = 1359
    n_vertices = 100000
    n_edges = 2160312
    return n_vertices, n_edges, max_degree


def read_vertex_bin(vertex_bin_file, n_vertices):
    """
    读取 graph.vertex.bin 文件，返回 offset 数组。
    """
    offsets = []
    with open(vertex_bin_file, 'rb') as f:
        for _ in range(n_vertices + 1):  # 读取 n_vertices + 1 个 int64 值
            offset = struct.unpack('<q', f.read(8))[0]  # 每个值为 int64
            offsets.append(offset)
    return offsets


def read_edge_bin(edge_bin_file, n_edges):
    """
    读取 graph.edge.bin 文件，返回 edge 数组。
    """
    edges = []
    with open(edge_bin_file, 'rb') as f:
        for _ in range(n_edges):  # 读取 n_edges 个 int32 值
            edge = struct.unpack('<i', f.read(4))[0]  # 每个值为 int32
            edges.append(edge)
    return edges


def convert_bin_to_txt(vertex_bin_file, edge_bin_file, meta_file, output_txt_file):
    """
    将 graph.vertex.bin、graph.edge.bin 和 graph.meta.txt 转换为原始的 TXT 格式。
    """
    # 读取 meta.txt 文件
    n_vertices, n_edges, _ = read_meta_txt(meta_file)

    # 读取 graph.vertex.bin 文件
    vertex_offsets = read_vertex_bin(vertex_bin_file, n_vertices)

    # 读取 graph.edge.bin 文件
    edge_array = read_edge_bin(edge_bin_file, vertex_offsets[-1])

    # 使用集合去重，确保每条边只保留一个方向
    unique_edges = set()

    # 根据 CSR 格式解析边
    for src in range(n_vertices):
        start = vertex_offsets[src]
        end = vertex_offsets[src + 1]
        for idx in range(start, end):
            dst = edge_array[idx]
            if src < dst:  # 确保只保留 (src, dst) 且 src < dst
                unique_edges.add((src, dst))

    # 写入输出文件
    with open(output_txt_file, 'w') as f:
        # 第一行写入顶点数和边数
        f.write(f"{n_vertices} {len(unique_edges)}\n")
        # 写入每条边
        for u, v in sorted(unique_edges):  # 按字典序排序
            f.write(f"{u} {v}\n")



if __name__ == "__main__":
    # 检查命令行参数数量
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <input_file>")
    #     sys.exit(1)

    # # 获取第一个参数作为输入文件路径
    # input_file = sys.argv[1]
    input_file = "test_gr3.txt"
    # # 输出目录路径（硬编码）
    output_dir = "/mnt/data-ssd/home/zhenlin/workspace/graphmining/graphmine_bench/glumin_data/datasets/testgr3/"

    # # 执行转换
    convert_graph(input_file, output_dir)
    
    # 输入文件路径
    # vertex_bin_file = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/mico/graph.vertex.bin"
    # edge_bin_file = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/mico/graph.edge.bin"
    # meta_file = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/glumin_data/datasets/mico/graph.meta.txt"

    # # 输出文件路径
    # output_txt_file = "/home/wuzhenlin/workspace/2-graphmining/graphmine_bench/graphpi_data/datasets/mico.txt"

    # # 执行转换
    # convert_bin_to_txt(vertex_bin_file, edge_bin_file, meta_file, output_txt_file)

    # print(f"Conversion complete. Output written to {output_txt_file}")