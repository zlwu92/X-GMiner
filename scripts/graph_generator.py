import networkx as nx
import numpy as np
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt

def generate_powerlaw_graph(v: int, e: int, alpha: float = 2.5) -> nx.Graph:
    """
    生成无向图，度数分布近似幂律
    :param v: 顶点数
    :param e: 边数
    :param alpha: 幂律指数（默认 2.5）
    :return: NetworkX 图对象
    """
    # 生成度序列（幂律分布）
    degrees = []
    while sum(degrees) != 2 * e:
        degrees = [round(d) for d in nx.utils.powerlaw_sequence(v, alpha)]
        degrees = [max(1, int(d * (2 * e / sum(degrees)))) for d in degrees]  # 缩放至总边数
        degrees[-1] = 2 * e - sum(degrees[:-1])  # 调整最后一个节点度数确保总和为偶数
    
    # 创建配置模型图
    G = nx.configuration_model(degrees, create_using=nx.Graph())
    G = nx.Graph(G)  # 移除多重边
    G.remove_edges_from(nx.selfloop_edges(G))  # 移除自环
    
    # 调整边数至目标值
    current_e = G.number_of_edges()
    if current_e < e:
        while current_e < e:
            u, v = random.sample(range(v), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                current_e += 1
    elif current_e > e:
        edges = list(G.edges())
        random.shuffle(edges)
        for edge in edges[:current_e - e]:
            G.remove_edge(*edge)
    
    return G

def save_graph(G: nx.Graph, output_path: Path):
    """保存图到 TXT 文件"""
    edges = list(G.edges())
    with open(output_path, "w") as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        # f.write(f"{G.number_of_edges()}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic power-law graph")
    parser.add_argument("--vertices", "-v", 
                        type=int, 
                        default=20,
                        # required=True, 
                        help="Number of vertices")
    parser.add_argument("--edges", "-e", 
                        type=int, 
                        default=50,
                        # required=True, 
                        help="Number of edges")
    parser.add_argument("--output", "-o", type=Path, default=Path("synthetic_graph.txt"), help="Output file")
    parser.add_argument("--alpha", "-a", type=float, default=2.5, help="Power-law exponent (default=2.5)")
    
    args = parser.parse_args()
    
    # 验证边数合理性
    max_edges = args.vertices * (args.vertices - 1) // 2
    if args.edges > max_edges:
        raise ValueError(f"Edges cannot exceed {max_edges} for {args.vertices} vertices (undirected, no self-loops)")
    
    # 生成并保存图
    G = generate_powerlaw_graph(args.vertices, args.edges, args.alpha)
    save_graph(G, args.output)
    print(f"✅ Graph saved to {args.output} (V={G.number_of_nodes()}, E={G.number_of_edges()})")


def generate_synthetic_graph_file(vertex_count, edge_count, output_file):
    edges = set()  # 用集合来存储唯一的边

    with open(output_file, 'w') as file:
        file.write(f"{vertex_count} {edge_count}\n")

        while len(edges) < edge_count:
            source = random.randint(0, vertex_count - 1)
            target = random.randint(0, vertex_count - 1)

            if source != target:  # 确保不是自环
                edges.add((min(source, target), max(source, target)) )  # (1, 3) 和 (3, 1) 视为同一条边

        for edge in edges:
            file.write(f"{edge[0]} {edge[1]}\n")
            
    return edges


def visualize_graph_from_file(file_path):
    G = nx.Graph()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        vertex_count, edge_count = map(int, lines[0].split())

        for line in lines[1:]:
            source, target = map(int, line.split())
            G.add_edge(source, target)

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_color='black')
    plt.axis('off')

    plt.savefig("graph_visualization.pdf", format="pdf")

    print("Graph visualization saved as: graph_visualization.pdf")


def find_rectangle_subgraphs(graph):
    rectangle_count = 0
    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if node1 == node2:
                continue
            if not graph.has_edge(node1, node2):
                continue

            for node3 in graph.nodes():
                if node3 in [node1, node2] or not graph.has_edge(node1, node3) or not graph.has_edge(node2, node3):
                    continue

                for node4 in graph.nodes():
                    if node4 in [node1, node2, node3] or not graph.has_edge(node1, node4) or not graph.has_edge(node2, node4) or not graph.has_edge(node3, node4):
                        continue

                    rectangle_count += 1
                    print(f"Rectangle pattern found: {node1}-{node2}-{node3}-{node4}")

    return rectangle_count


# 从文件中读取图信息
def read_graph_from_file(file_path):
    G = nx.Graph()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        _, _ = map(int, lines[0].split())

        for line in lines[1:]:
            source, target = map(int, line.split())
            print(f"Adding edge: {source}-{target}")
            G.add_node(source)
            G.add_node(target)
            G.add_edge(source, target)
            # G.add_edge(target, source)

    return G


def find_triangles(file_path):
    # 读取图数据并构建邻接表
    adj = {}
    with open(file_path, 'r') as f:
        n, m = map(int, f.readline().split())
        for _ in range(m):
            u, v = map(int, f.readline().split())
            if u > v:  # 保持邻接表有序
                u, v = v, u
            for node in [u, v]:
                if node not in adj:
                    adj[node] = set()
            adj[u].add(v)
            adj[v].add(u)
    
    # 查找所有三角形
    triangles = []
    nodes = sorted(adj.keys())
    for u in nodes:
        neighbors = sorted([v for v in adj[u] if v > u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i+1, len(neighbors)):
                w = neighbors[j]
                if w in adj[v]:
                    triangles.append((u, v, w))
    
    print(f"Total triangles: {len(triangles)}")
    print("Unique triangles:")
    for t in sorted(triangles):
        print(f"{t[0]} {t[1]} {t[2]}")
        
    return triangles



if __name__ == "__main__":
    # main()
    

    # 输入顶点数和边数
    vertex_count = int(input("Enter the number of vertices: "))
    edge_count = int(input("Enter the number of edges: "))

    output_file = "test_gr2.txt"

    edges = generate_synthetic_graph_file(vertex_count, edge_count, output_file)

    # print(f"Synthetic graph file generated: {output_file}")
    
    visualize_graph_from_file(output_file)
    
    # graph = read_graph_from_file(output_file)
    
    # 计算图中的三角形数量
    # triangle_count = sum(nx.triangles(graph).values()) // 6  # 每个三角形被计算了6次，所以要除以6
    # print(f"Number of triangles in the graph: {triangle_count}")
    
    # 示例用法
    triangles = find_triangles(output_file)
    
