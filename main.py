# 按DCT判断相似度
import time

import cv2
import numpy as np

from getFeature import get_feature
from divide import divide2block


# 计算矩阵的DCT变换
def DCT(matrix: np.ndarray) -> np.ndarray:
    matrix_float = np.float32(matrix)
    matrix_dct = cv2.dct(matrix_float)
    return matrix_dct


def get_dct_block(img_block: np.ndarray) -> np.ndarray:
    """
    :param img_block: 以8*8矩阵为元素的二维矩阵
    :return: 以DCT变换后的8*8矩阵为元素的二维矩阵
    """
    r, c = img_block.shape
    dct_block = np.zeros((r, c), dtype=np.ndarray)
    for i in range(r):
        for j in range(c):
            dct_block[i][j] = DCT(img_block[i][j])
    return dct_block


# 简单DCT量化
def quantify_dct_block(dct_block: np.ndarray, Q: float) -> np.ndarray:
    """
    :param dct_block: 以DCT变换后的8*8矩阵为元素的二维矩阵
    :return: 量化后的DCT变换后的8*8矩阵
    """
    # 亮度量化矩阵
    q_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 36, 55, 64, 81, 104, 113, 92],
                         [49, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]])
    q_matrix = q_matrix * Q
    r, c = dct_block.shape
    dct_block_int = np.zeros((r, c), dtype=np.ndarray)
    for i in range(r):
        for j in range(c):
            # dct_block_int[i][j] = np.int32(dct_block[i][j] * Q)
            dct_block_int[i][j] = np.int32(dct_block[i][j] / q_matrix)
    return dct_block_int


def get_img_masked_and_bin(img, relative_block_list):
    img_masked = img.copy()
    img_bin = np.ones(img.shape, dtype=np.uint8) * 255
    for point in relative_block_list:
        for i in range(8):
            for j in range(8):
                img_masked[point[0] + i][point[1] + j] = 0
                img_bin[point[0] + i][point[1] + j] = 0
    return img_masked, img_bin


if __name__ == '__main__':
    # image_path = './data/006_F.png'
    image_path = input("请输入图片文件路径：")
    start = time.time()
    # 按灰度值读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_block = divide2block(img)
    dct_block = get_dct_block(img_block)
    quantified_block = quantify_dct_block(dct_block, 0.01)

    relative_block_list = get_feature(quantified_block, 0.02, 0.5, 50, 1.5)
    if relative_block_list == None:
        print("未找到copy-move伪造块，进程退出")
        print(f'耗时：{time.time() - start}')
        exit(0)

    img_masked, img_bin = get_img_masked_and_bin(img, relative_block_list)

    print(f'耗时：{time.time() - start}')

    tmp = np.hstack((img, img_masked, img_bin))
    cv2.namedWindow("Image")
    cv2.imshow("Image", tmp)
    cv2.waitKey(0)
