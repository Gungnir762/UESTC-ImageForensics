import cv2
import numpy as np

# 计算矩阵的DCT变换
def get_dct(matrix: np.ndarray)->np.ndarray:
    matrix_float = np.float32(matrix)
    matrix_dct = cv2.dct(matrix_float)
    return matrix_dct

#获取特征值对
def get_feature(matrix):
    """
    :param matrix: 二维数组
    :return: map
    """
    pass


if __name__== '__main__':
    pass