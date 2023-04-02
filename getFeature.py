import random
import numpy as np
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics  # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图


def get_feature(matrix: np.ndarray, Q1: float, Q2: float):
    """
    :param matrix:
    :param Q1: Q1取[0,1]，代表两个copy-move图像间的最小距离，Q1取0代表距离最小为0，取1代表距离为整个图像的对角线，一般取0.2
    :param Q2: Q2取(0,+inf)，代表对判断两个特征向量相似性的严格程度，越小越严格，一般取10
    :return:{(x,y):(((x11,y11),(x12,y12)),((x21,y21),(x22,y22)),...),...}
            key (x,y)为位移向量,value (x11,y11),(x12,y12)为相同位移向量的块在原矩阵中的坐标,每两个块为一组
    """
    vec_arr = []
    n, m = matrix.shape[:2]
    for i in range(n):
        for j in range(m):
            vec_arr.append({"vec": get_zigzag(matrix[i][j]), "x": i, "y": j})
    vec_arr.sort(key=cmp)

    min_dis = pow(n ** 2 + m ** 2, 0.5) * Q1
    res = {}
    dis_vec_list = []
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

    useful_dis_vec_array = get_biggest_cluster(dis_vec_list)
    usefel_res = {}
    # show_data(dis_vec_list)
    for i in range(len(useful_dis_vec_array)):
        useful_dis_vec = useful_dis_vec_array[i]
        usefel_res[useful_dis_vec] = res[useful_dis_vec]
    return res


def get_zigzag(matrix: np.ndarray) -> list:
    """
    :param matrix:8*8的DCT变换后的小矩阵
    :return: z字形排列后得到的列表
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
    return res


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


# 放缩向量
def shrink_vec(vec: tuple, Q: float) -> tuple:
    x = int(vec[0] * Q)
    y = int(vec[1] * Q)
    if x < 0:
        x = -x
        y = -y
    elif x == 0 and y < 0:
        y = -y
    return (x, y)


def get_biggest_cluster(point_array: list) -> list:
    X = np.array(point_array)
    db = skc.DBSCAN(eps=1.5, min_samples=10).fit(X)
    labels = db.labels_
    labels_copy = []
    for i in labels:
        if i != -1:
            labels_copy.append(i)
    max_val = max(labels_copy, key=labels_copy.count)
    res = X[labels == max_val]

    ans=[]
    for i in res:
        ans.append(tuple(i))
    ans=list(set(ans))
    return ans


def show_data(data: list):
    X = np.array(data)

    db = skc.DBSCAN(eps=1.5, min_samples=10).fit(X)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
    labels = db.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

    print('每个样本的簇标号:')
    print(labels)

    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    print('分簇的数目: %d' % n_clusters_)
    print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels))  # 轮廓系数评价聚类的好坏

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
