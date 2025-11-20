import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体


def friedman_nemenyi_test(data, alpha=0.1):  # 执行Friedman检验和Nemenyi事后检验
    # 参数: data: 形状为(num_datasets, num_algorithms) alpha: 显著性水平
    num_data, num_algo = data.shape
    ranks = np.zeros_like(data, dtype=float)  # 在每个数据集上对算法进行排名
    for i in range(num_data):  # 对每一行进行排名，处理并列情况
        ranks[i] = stats.rankdata(data[i])

    avg_rank = np.mean(ranks, axis=0)  # 计算每个算法的平均排名
    sum_sq_ranks = np.sum(avg_rank ** 2)
    friedman = (12 * num_data / (num_algo * (num_algo + 1))) * \
               (sum_sq_ranks - (num_algo * (num_algo + 1) ** 2) / 4)  # 计算Friedman统计量
    chi2 = stats.chi2.ppf(1 - alpha, num_algo - 1)  # 获取卡方分布的临界值
    q_alpha = get_q_alpha(num_algo, alpha)  # 获取q_alpha
    cd = q_alpha * np.sqrt(num_algo * (num_algo + 1) / (6 * num_data))  # 计算临界差异CD
    return friedman, chi2, cd, avg_rank  # 返回: friedman: Friedman统计量 cd: 临界差异值 rankings: 各算法的平均排名


def get_q_alpha(k, alpha=0.1):  # 获取q_alpha
    q_table = {
        2: {0.1: 1.960, 0.05: 2.343},
        3: {0.1: 2.343, 0.05: 2.569},
        4: {0.1: 2.569, 0.05: 2.728},
        5: {0.1: 2.728, 0.05: 2.850},
        6: {0.1: 2.850, 0.05: 2.948},
        7: {0.1: 2.948, 0.05: 3.031},
        8: {0.1: 3.031, 0.05: 3.102},
        9: {0.1: 3.102, 0.05: 3.164},
        10: {0.1: 3.164, 0.05: 3.219}
    }  # 常见k值的q_\alpha的近似值
    if k in q_table and alpha in q_table[k]:  # k和alpha在表中
        return q_table[k][alpha]
    else:  # k和alpha不在表中，使用近似公式
        return 2.5 + 0.1 * k  # 这只是一个粗略的近似


def plot_critical_difference_diagram(avg_rank, cd, algorithm_names): # 绘制临界差异图
    # 参数: avg_rank: 各算法的平均排名 cd: 临界差异值 algorithm: 算法名称列表
    num_algo = len(avg_rank)
    sorted_indices = np.argsort(avg_rank)
    sorted_ranks = avg_rank[sorted_indices]
    sorted_names = np.array(algorithm_names)[sorted_indices]
    ranks_np = np.array(avg_rank)
    # 查找所有最大团（maximal cliques），即不具有显著差异的最大算法组
    cliques = []
    for i in range(num_algo):
        for j in range(i + 1, num_algo):
            if sorted_ranks[j] - sorted_ranks[i] > cd:
                break
            is_maximal = (i == 0 or sorted_ranks[j] - sorted_ranks[i - 1] > cd) and \
                         (j == num_algo - 1 or sorted_ranks[j + 1] - sorted_ranks[i] > cd)
            if is_maximal:
                cliques.append((i, j))

    # 为每个clique分配层级，避免重叠
    cliques.sort(key=lambda x: x[0])
    level_intervals = [[] for _ in range(num_algo)]
    level_dict = {}
    for clique in cliques:
        x_start = sorted_ranks[clique[0]]
        x_end = sorted_ranks[clique[1]]
        for lv in range(len(level_intervals)):
            conflict = False
            for s, e in level_intervals[lv]:
                if max(x_start, s) < min(x_end, e):
                    conflict = True
                    break
            if not conflict:
                level_intervals[lv].append((x_start, x_end))
                level_dict[clique] = lv
                break

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 4))
    min_rank, max_rank = sorted_ranks[0], sorted_ranks[-1]
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    max_lv = max(level_dict.values()) if level_dict else 0
    ax.set_ylim(-1, 0.5 + max_lv * 0.3 + 0.5)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 绘制排名轴
    ax.hlines(0, min_rank - 0.5, max_rank + 0.5, color='black', lw=1)
    # 算法名称
    for r, name in zip(sorted_ranks, sorted_names):
        ax.text(r, -0.1, name, ha='center', va='top', rotation=45)
        ax.plot(r, 0, 'o', color='red', markersize=10, zorder=5)  # 红点
        ax.text(r, 0.15, f'{r:.2f}', ha='center', va='bottom', fontsize=11, color='#1f77b4', fontweight='bold', rotation=35)

    # 绘制clique线条（标注不显著差异组）
    for clique, lv in level_dict.items():
        x_start = sorted_ranks[clique[0]]
        x_end = sorted_ranks[clique[1]]
        y = 0.6 + lv * 0.3
        ax.hlines(y, x_start, x_end, color='black', lw=4)
        ax.vlines(x_start, y - 0.1, y + 0.1, color='black', lw=2)
        ax.vlines(x_end, y - 0.1, y + 0.1, color='black', lw=2)
    # 添加CD值
    ax.text(max_rank + 0.2, 0.6, f'CD = {cd:.2f}', fontsize=13, va='center', ha='left')
    ax.set_xlabel('Average ranking (lower is better)')
    ax.set_title('Critical Difference Diagram')
    plt.tight_layout()
    plt.show()

#------------------------#

np.random.seed(42)

# 5 种算法在 10 个数据集上的误差（越小越好）
error = np.random.rand(10, 5) * 15 + np.array([8, 7, 5, 9, 6])   # 人为制造差距
algorithms = ['SVM', '随机森林', 'XGBoost', 'KNN', '神经网络']

# 执行检验
f_stat, chi_crit, cd, avg_rank = friedman_nemenyi_test(error, alpha=0.1)

print(f'Friedman χ² = {f_stat:.3f}  >  χ²临界值 {chi_crit:.3f} → '
      f'{"存在" if f_stat>chi_crit else "不存在"}显著差异')
print(f'CD = {cd:.3f}')
print('平均排名：')
for name, r in zip(algorithms, avg_rank):
    print(f'  {name:<8} {r:>.3f}')

# 画 CD 图（粗蓝线 = 无显著差异）
plot_critical_difference_diagram(avg_rank, cd, algorithms)