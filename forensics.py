import time
import cv2
import io
import numpy as np
from getFeature import get_feature


# 按像素点分为(M-b+1)*(N-b+1)个8*8的矩阵块，本问题中为（505，505）
def divide2block(img: np.ndarray, step=8) -> np.ndarray:
    """
    :param img: 原图像
    :param step: 步长
    :return: 长，宽均为img.shape-(step,step)的矩阵，且每个元素为8*8的矩阵
    """

    def create_matrix(matrix: np.ndarray, row, col, step=8) -> np.ndarray:
        """
        :param matrix: 原矩阵
        :param row: 开始时分块8*8矩阵在原矩阵中的X坐标
        :param col: 开始时分块8*8矩阵在原矩阵中的Y坐标
        :param step: 步长
        :return: 创建好的8*8矩阵
        """
        temp = np.zeros((step, step), dtype=int)
        for i, m in zip(range(step), range(row, row + step)):
            for j, n in zip(range(step), range(col, col + step)):
                # print(matrix[m][n])
                temp[i][j] = matrix[m][n]
        return temp

    r, c = img.shape
    r_block = r - step + 1
    c_block = c - step + 1
    img_block = np.zeros((r_block, c_block), dtype=np.ndarray)

    # 按8*8的矩阵分块
    for i in range(r_block):
        for j in range(c_block):
            # print(i, j)
            img_block[i][j] = create_matrix(img, i, j, step)
            # print(img_block[i][j])
    return img_block


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


def get_img_bin_io(img_bio: bytes) -> [io.BytesIO]:
    start = time.time()
    # 按灰度值读取
    # img = cv2.imread(img_bio, cv2.IMREAD_GRAYSCALE)
    img_np = np.frombuffer(img_bio, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

    img_block = divide2block(img)
    dct_block = get_dct_block(img_block)
    quantified_block = quantify_dct_block(dct_block, 0.01)

    relative_block_list = get_feature(quantified_block, 0.02, 0.5, 50, 1.5)
    if relative_block_list == None:
        print("未找到copy-move伪造块，进程退出")
        print(f'耗时：{time.time() - start}')
        exit(0)

    img_bin = np.ones(img.shape, dtype=np.uint8) * 255
    for point in relative_block_list:
        for i in range(8):
            for j in range(8):
                img_bin[point[0] + i][point[1] + j] = 0

    print(f'耗时：{time.time() - start}')
    img_bytes = cv2.imencode(".png", img_bin)[1]
    bytes_io = io.BytesIO(img_bytes)

    # img_bytes = cv2.imencode(".png", img)[1]
    # bytes_io = io.BytesIO(img_bytes)
    return bytes_io


if __name__ == '__main__':
    image_path = './data/015_F.png'
    # image_path = input("请输入图片文件路径：")
    start = time.time()
    # 按灰度值读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_bytes = cv2.imencode(".png", img)[1]
    bytes_io = io.BytesIO(img_bytes)

    img_bio = get_img_bin_io(bytes_io)
    img_np = np.frombuffer(img_bio.getvalue(), dtype=np.uint8)
    img_bin = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

    tmp = np.hstack((img, img_bin))
    cv2.namedWindow("Image")
    cv2.imshow("Image", tmp)
    cv2.waitKey(0)
