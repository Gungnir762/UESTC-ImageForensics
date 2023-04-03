import random
import numpy as np
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics  # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图
import time


def get_feature(matrix: np.ndarray, Q1: float, Q2: float, Q3: int):
    """
    :param matrix:
    :param Q1: Q1取[0,1]，代表两个copy-move图像间的最小距离，Q1取0代表距离最小为0，取1代表距离为整个图像的对角线，一般取0.2
    :param Q2: Q2取(0,+inf)，代表对判断两个特征向量相似性的严格程度，越小越严格，一般取10
    :param Q3: Q3取正整数，一般在[5,10]，越小，判断为疑似copy-move的块数越多，相应地准确度也会变低
    :return: list[tuple]，有效点构成的列表
    """
    vec_arr = []
    n, m = matrix.shape[:2]
    x, y = matrix[0][0].shape
    for i in range(n):
        for j in range(m):
            vec_arr.append({"vec": get_zigzag(matrix[i][j], (x * y) // 3), "x": i, "y": j})
    vec_arr.sort(key=cmp)
    print("特征向量排序完毕")

    min_dis = pow(n ** 2 + m ** 2, 0.5) * Q1
    res = {}
    dis_vec_list = []
    start = time.perf_counter()
    for i in range(len(vec_arr))[1:]:
        dis_vec = cal_dis_vec(vec_arr[i - 1], vec_arr[i])
        if cal_module(dis_vec) < min_dis:
            continue
        if dif_of_vec(vec_arr[i - 1]["vec"], vec_arr[i]["vec"]) > Q2:
            continue
        # if dif_of_vec(vec_arr[i - 1]["vec"], vec_arr[i]["vec"])<10:
        #     print(dif_of_vec(vec_arr[i - 1]["vec"], vec_arr[i]["vec"]), dis_vec)
        dis_vec_list.append([dis_vec[0], dis_vec[1]])
        if not res.__contains__(dis_vec):
            k = ((vec_arr[i]["x"], vec_arr[i]["y"]), (vec_arr[i - 1]["x"], vec_arr[i - 1]["y"]))
            s = {k}
            res[dis_vec] = s
        else:
            res[dis_vec].add(((vec_arr[i]["x"], vec_arr[i]["y"]), (vec_arr[i - 1]["x"], vec_arr[i - 1]["y"])))
    end = time.perf_counter()

    useful_dis_vec_array = get_biggest_cluster(dis_vec_list, Q3)
    useful_x_points = []
    usefel_points_pairs = {}
    usefel_points = []
    for i in range(len(useful_dis_vec_array)):
        useful_dis_vec = useful_dis_vec_array[i]
        for points_pair in res[useful_dis_vec]:
            usefel_points_pairs[points_pair[0]] = points_pair[1]
            useful_x_points.append(list(points_pair[0]))
    # show_data(useful_x_points,3)
    # exit(0)
    useful_x_points = get_biggest_cluster(useful_x_points, 3)
    print("可疑点寻找完毕，正在统计...")
    for x_point in useful_x_points:
        usefel_points.append(x_point)
        usefel_points.append(usefel_points_pairs[x_point])
    return usefel_points


# 获取z字形展开，取前len位
def get_zigzag(matrix: np.ndarray, len: int) -> list:
    """
    :param matrix: 8*8的DCT变换后的小矩阵
    :param len: 保留的长度
    :return: z字形排列后得到的列表，长度为len
    """
    res = []
    f = 0
    loc = {"x": 0, "y": 0}
    n, m = matrix.shape
    n -= 1
    m -= 1
    while loc["x"] <= n and loc["y"] <= m:
        if f == 0:
            while loc["x"] < n and loc["y"] > 0:
                res.append(matrix[loc["x"]][loc["y"]])
                loc["x"] += 1
                loc["y"] -= 1
            res.append(matrix[loc["x"]][loc["y"]])
            if loc["x"] < n:
                loc["x"] += 1
            else:
                loc["y"] += 1
            f ^= 1
        else:
            while loc["x"] > 0 and loc["y"] < m:
                res.append(matrix[loc["x"]][loc["y"]])
                loc["x"] -= 1
                loc["y"] += 1
            res.append(matrix[loc["x"]][loc["y"]])
            if loc["y"] < m:
                loc["y"] += 1
            else:
                loc["x"] += 1
            f ^= 1
    return res[:len]


def cmp(a: dict) -> list:
    return a["vec"]


# 计算位移向量
def cal_dis_vec(a: dict, b: dict) -> tuple:
    x = a["x"] - b["x"]
    y = a["y"] - b["y"]
    if x < 0:
        x = -x
        y = -y
    elif x == 0 and y < 0:
        y = -y
    return x, y


# 计算两个特征向量的相似程度
def dif_of_vec(a: list, b: list) -> float:
    l = min(len(a), len(b))
    all = 0.0
    for i in range(l):
        all += ((a[i] - b[i]) ** 2) / (i + 1)
    return all


# 计算向量的模
def cal_module(vec: tuple) -> float:
    return pow(vec[0] ** 2 + vec[1] ** 2, 0.5)


# 寻找最有可能的位移向量簇
def get_biggest_cluster(point_array: list, Q: int) -> list:
    """
    :param point_array: list[list]，点集
    :param Q: 表示每个点周围的八连通区域内（包括自己的位置，但不包括自己）还有多少个点，它才会与这些点纳入同一聚类
    :return: list[tuple]，最大簇的点集（去重后）
    """
    X = np.array(point_array)
    db = skc.DBSCAN(eps=1.5, min_samples=Q).fit(X)
    labels = db.labels_
    labels_copy = []
    for i in labels:
        if i != -1:
            labels_copy.append(i)
    max_val = max(labels_copy, key=labels_copy.count)
    res = X[labels == max_val]

    ans = []
    for i in res:
        ans.append(tuple(i))
    ans = list(set(ans))
    return ans


def show_data(data: list, Q: int):
    X = np.array(data)

    print("开始聚类")
    start = time.perf_counter()
    db = skc.DBSCAN(eps=1.5, min_samples=Q).fit(X)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
    end = time.perf_counter()
    print("聚类算法执行时间:", end - start, "s")
    labels = db.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

    print('每个样本的簇标号:')
    print(labels)

    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    print('分簇的数目: %d' % n_clusters_)
    # print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels))  # 轮廓系数评价聚类的好坏

    # label的值是一个与X长度相同，里面为每一个X对应的簇Index

    # label==i返回一个与X长度相同，值为True 或False的数组（若簇Index==i是True）

    # one_cluster就是当前簇对应的X数据

    labels_copy = []
    for i in labels:
        if i != -1:
            labels_copy.append(i)
    max_val = max(labels_copy, key=labels_copy.count)
    for i in range(n_clusters_):
        one_cluster = X[labels == i]
        plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
    plt.show()
    plt.close()
    one_cluster = X[labels == max_val]
    plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
    plt.show()
