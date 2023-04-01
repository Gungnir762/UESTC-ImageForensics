import random
import numpy as np


def get_feature(matrix: np.ndarray,Q1:float,Q2:float):
    """
    :param matrix:
    :param Q1: Q1取[0,1]，代表两个copy-move图像间的最小距离，Q1取0代表距离最小为0，取1代表距离为整个图像的对角线，一般取0.2
    :param Q2: Q2取(0,1]，代表位移向量的放缩程度，取1代表不放缩，取值越小代表位移向量放缩的系数越小。位移向量放缩后取整，相同的视为一组。一般取0.3~0.5
    :return:
    """
    vec_arr = []
    n, m = matrix.shape[:2]
    for i in range(n):
        for j in range(m):
            vec_arr.append({"vec": get_zigzag(matrix[i][j]), "x": i, "y": j})
    vec_arr.sort(key=cmp)
    # for i in vec_arr:
    #     print(i["vec"])

    min_dis = pow(n ** 2 + m ** 2, 0.5) * Q1
    res = {}
    for i in range(len(vec_arr))[1:]:
        dis_vec = cal_dis_vec(vec_arr[i - 1], vec_arr[i])
        if cal_module(dis_vec) < min_dis:
            continue
        dis_vec=shrink_vec(dis_vec,Q2)
        if not res.__contains__(dis_vec):
            k = ((vec_arr[i]["x"], vec_arr[i]["y"]), (vec_arr[i - 1]["x"], vec_arr[i - 1]["y"]))
            s = set([k])
            res[dis_vec] = s
        else:
            res[dis_vec].add(((vec_arr[i]["x"], vec_arr[i]["y"]), (vec_arr[i - 1]["x"], vec_arr[i - 1]["y"])))

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
    return (x, y)


# 计算向量的模
def cal_module(vec: tuple) -> float:
    return pow(vec[0] ** 2 + vec[1] ** 2, 0.5)

# 放缩向量
def shrink_vec(vec: tuple, Q: float) -> tuple:
    return (int(vec[0] * Q), int(vec[1] * Q))
