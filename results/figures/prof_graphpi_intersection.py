import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
# sys.path.append(".")
# import myplot
import matplotlib
import os
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
from scipy.stats import gmean
import datetime
import matplotlib.patches as patches

current_date = datetime.date.today()

# 读取 CSV 文件
# data = pd.read_csv('res.csv')
data = pd.read_csv('../prof_graphpi_intersection_time_percentage.csv')


colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", 
          "#800080", "#ffc0cb", "#ffa500", "#808080", 
          "#007fff", "#8e52dc", "#ff7f50", "#ff4500", 
          "#396e04", "#ff00ff", "#ff69b4", "#489177",
          "#CC99FF", "#FFCC99", "#99CCFF", "#CCFF99",
          "#a98ec6", "#f9dd7e"]

mycolors = ['#014F9C', '#3C79B4', '#78A3CC', '#B3CDE4', '#EEF7FC']

colorsuite = ["#3376b0", "#498dbd", "#68a4cd", 
              "#88bddb", "#aed7e9", "#d1eff7",
              "#8ab9dd", "#b0cfe9", "#d6e5f5"]


App_abbr = [
    ("cit-Patents", "Patents"),
    ("com-dblp", "DBLP"),
    ("com-youtube", "YouTube"),
    ("com-lj", "LJ"),
    ("com-orkut", "Orkut"),
    ("wiki-Vote", "Wiki-Vote"),
]

# 替换 Dataset 标签为缩写
def get_abbreviated_label(dataset):
    for full_name, abbrev in App_abbr:
        if full_name in dataset:  # 检查是否为子字符串
            return abbrev
    return dataset  # 如果未找到匹配项，返回原始名称


width = .7
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 36
plt.rcParams['legend.frameon'] =  True


def prof_graphpi_intersection():
    # 获取所有唯一的 Pattern 值
    patterns = data['Pattern'].unique()

    # 设置绘图参数
    n_datasets = len(data['Dataset'].unique())  # 数据集数量
    bar_width = 0.35  # 柱状图宽度
    index = np.arange(n_datasets)  # x 轴索引
    
    # 创建画布和子图布局 (上三下三)
    fig, axes = plt.subplots(2, 3, figsize=(20, 7), constrained_layout=True)

    # 遍历每个 Pattern 并绘制子图
    for i, pattern in enumerate(patterns):
        # 获取当前 Pattern 的数据
        subset = data[data['Pattern'] == pattern]
        
        # 确定子图位置
        row = i // 3  # 行号
        col = i % 3   # 列号
        ax = axes[row, col]
        
        ax.set_ylim(0, 1.01)
        # ax1.yaxis.set_ticks([.1, 1, 10, 100, 1e3, 1e4])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        # ax.yaxis.set_ticks([0, .2, .4, .6, .8, 1])
        
        # 绘制柱状图
        # bars1 = ax.bar(index, subset['Int'], bar_width, label='Int', color='skyblue')
        # bars2 = ax.bar(index + bar_width, subset['Diff'], bar_width, label='Diff', color='orange')
        # 绘制堆叠柱状图
        bars1 = ax.bar(index, subset['Int'] / 100, bar_width, label='Intersection', color=mycolors[0])
        bars2 = ax.bar(index, subset['Diff'] / 100, bar_width, bottom=subset['Int']/100, label='Difference', color=mycolors[2])
        
        abbreviated_labels = [get_abbreviated_label(ds) for ds in subset['Dataset']]
        print(abbreviated_labels)
        # 设置标题和标签
        # ax.set_title(f'Pattern: {pattern}', fontsize=14)
        # ax.set_xticks(index + bar_width / 2)
        ax.set_xticks(index)
        ax.set_xticklabels(abbreviated_labels, rotation=0, fontsize=10)
        # ax.set_xlabel('Dataset', fontsize=12)
        ax.set_xlabel(f'Pattern: {pattern}', fontsize=14)
        ax.set_ylabel('Time Percentage (%)', fontsize=12)

        ax1 = ax
        ax1.yaxis.grid(True, color='gray', linestyle = '--', linewidth = 0.5)  # horizontal grid, another method
        # 设置y轴刻度标签加粗
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')

        # set rotation of the last xtick
        for label in ax1.get_xticklabels():
            if label.get_text() == 'GMean':
                label.set_rotation(30)
                label.set_fontsize(30)
                label.set_fontweight('bold')
                
        handles, labels = ax1.get_legend_handles_labels()
        # legend = ax1.legend(handles, labels, bbox_to_anchor=(0.31, 0.92), loc='center', ncol=3, fontsize=35, frameon=True)
        legend = ax1.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), loc='center', ncol=3, fontsize=16, frameon=True)
        frame = legend.get_frame()
        frame.set_edgecolor('#808080')  # 设置边框颜色
        frame.set_linewidth(2)  # 设置边框粗细
        frame.set_alpha(1)  # 设置边框透明度

        spines1 = ax1.spines
        spines1['top'].set_linewidth(5)
        spines1['bottom'].set_linewidth(5)
        spines1['left'].set_linewidth(5)
        spines1['right'].set_linewidth(5)  

        # ax.legend()
        # ax.grid(axis='y', linestyle='--', alpha=0.9)

    # 调整布局并保存图像
    # plt.suptitle('Bar Chart Group by Pattern', fontsize=16)
    print("prepare to save figure")
    plt.savefig(f'prof_graphpi_intersection_{current_date}.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    prof_graphpi_intersection()