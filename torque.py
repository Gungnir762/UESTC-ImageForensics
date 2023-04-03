# 按矩判断相似度
import time
from collections import defaultdict

import numpy as np
import cv2
from divide import divide2block


# 求每一个分块的矩
def get_torque(img_block: np.ndarray) -> np.ndarray:
    """
    :param img_block: 以8*8矩阵为元素的二维矩阵
    :return: 以矩为元素的二维矩阵
    """
    r, c = img_block.shape
    torque_block = np.zeros((r, c), dtype=np.ndarray)
    for i in range(r):
        for j in range(c):
            torque_block[i][j] = np.sum(img_block[i][j])
    return torque_block


# 获取矩相同的块的坐标
def get_equal_torque_block(torque_block: np.ndarray) -> dict:
    """
    :param torque_block: 以原图像分块后的8*8矩阵的矩为元素的二维矩阵
    :return: {torque：[(x1, y1), (x2, y2), ...]}
    """
    equal_dict = defaultdict(list)
    for i, row in enumerate(torque_block):
        for j, torque in enumerate(row):
            equal_dict[torque].append((i, j))
    return equal_dict


def vector_quantization(equal_dict, threshold=2) -> dict:
    """
    :param equal_dict: {torque：[(x1, y1), (x2, y2), ...]}
    :return: {(x,y)：[((x11, y11), (x12, y12)), ((x21,y21),(x22,y22)),...]}
            key (x,y)为位移向量,value (x11,y11),(x12,y12)为相同位移向量的块在原矩阵中的坐标,每两个块为一组
    """
    vector_dict = defaultdict(list)
    for key in equal_dict:
        # 相同矩的块数小于2则不计入结果
        if len(equal_dict[key]) < threshold:
            continue
        for i, point1 in enumerate(equal_dict[key]):
            for point2 in equal_dict[key][i + 1:]:
                x = point1[0] - point2[0]
                y = point1[1] - point2[1]
                vector_dict[(x, y)].append((point1, point2))
    return vector_dict


# 得到黑白二值化图像
def get_img_bin(img, vector_dict, threshold=5) -> np.ndarray:
    """
    :param img: 原图像
    :param vector_dict:
    :param threshold: 位移向量的模小于threshold则不计入结果
    :return: 二值化图像
    """
    img_bin = np.ones(img.shape, dtype=np.uint8) * 255  # 255为白色
    for key in vector_dict.keys():
        if np.linalg.norm(key) < threshold:
            continue
        for vec in vector_dict[key]:
            # print(vec)
            for k in range(2):
                x = vec[k][0]
                y = vec[k][1]
                for ii in range(8):
                    for jj in range(8):
                        img_bin[x + ii][y + jj] = 0
    return img_bin


if __name__ == '__main__':
    start = time.time()
    image_path = './data/015_F.png'
    # 按灰度值读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_block = divide2block(img)
    torque_block = get_torque(img_block)
    equal_dict = get_equal_torque_block(torque_block)

    # print(torque_block)
    # print(type(torque_block[0][0]))

    # with open('./tmps/equal.csv', 'w') as f:
    #     for i in equal_dict:
    #         f.write(str(i) + " " + str(equal_dict[i]) + '\n')

    vector_dict = vector_quantization(equal_dict)
    with open('./tmps/vec.csv', 'w') as f:
        for i in vector_dict:
            f.write(str(i) + " " + str(vector_dict[i]) + '\n')

    # img_bin = get_img_bin(img, vector_dict, threshold=1)
    # # # 二值化
    # # retval, bin_img = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
    # # show_img = bin_img * 255
    print(f'耗时：{time.time() - start}')
    # tmp = np.hstack((img, img_bin))
    # cv2.imshow('img', tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
