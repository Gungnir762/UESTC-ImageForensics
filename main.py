import cv2
import numpy as np

from GetFeature import get_feature


# 计算矩阵的DCT变换
def DCT(matrix: np.ndarray) -> np.ndarray:
    matrix_float = np.float32(matrix)
    matrix_dct = cv2.dct(matrix_float)
    return matrix_dct


# 将原二维矩阵按8*8分块，返回分块后的矩阵
def divide2block(matrix: np.ndarray, step=8) -> np.ndarray:
    # 创建8*8的小矩阵
    def create_matrix(matrix: np.ndarray, row, col, step) -> np.ndarray:
        """
        :param matrix: 原矩阵
        :param row: 开始时处在分块8*8矩阵中的X坐标
        :param col: Y坐标
        :param step: 步长
        :return: 创建好的8*8矩阵
        """
        temp = np.zeros((step, step), dtype=int)
        for i, m in zip(range(step), range(step * row, step * row + step)):
            for j, n in zip(range(step), range(step * col, step * col + step)):
                # print(matrix[m][n])
                temp[i][j] = matrix[m][n]
        return temp

    # 获取矩阵的行数和列数
    row, col = matrix.shape
    # 计算分块后的矩阵的行数和列数
    row_block = row // step
    col_block = col // step
    # 初始化分块后的矩阵
    matrix_block = np.zeros((row_block, col_block), dtype=np.ndarray)
    # 按8*8的矩阵分块
    for i in range(0, row_block):
        for j in range(0, col_block):
            matrix_block[i][j] = create_matrix(matrix, i, j, step)
    return matrix_block


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
def quantify_dct_block(dct_block: np.ndarray, Q=5) -> np.ndarray:
    """
    :param dct_block: 以DCT变换后的8*8矩阵为元素的二维矩阵
    :param Q: 量化系数
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

    r, c = dct_block.shape
    dct_block_int = np.zeros((r, c), dtype=np.ndarray)
    for i in range(r):
        for j in range(c):
            # dct_block_int[i][j] = np.int32(dct_block[i][j] * Q)
            dct_block_int[i][j] = np.int32(dct_block[i][j] / q_matrix)
    return dct_block_int


# 得到黑白二值化图像
def get_binary_img(dct_block: np.ndarray, relative_block_list) -> np.ndarray:
    """
    :param dct_block: 量化后的DCT变换后的8*8矩阵
    :param relative_block_list:
    :return: 只有两种像素值（0或255）的二值化图像dd
    """
    r, c = dct_block.shape
    r0, c0 = dct_block[0][0].shape
    binary_img = np.zeros((r, c), dtype=np.ndarray)

    for key in relative_block_list:
        if len(relative_block_list[key]) < 47:
            continue
        for point_pair in relative_block_list[key]:
            binary_img[point_pair[0][0]][point_pair[0][1]] = np.ones((r0, c0), dtype=np.uint8) * 255
            binary_img[point_pair[1][0]][point_pair[1][1]] = np.ones((r0, c0), dtype=np.uint8) * 255

    return binary_img


if __name__ == '__main__':
    image_path = './data/015_F.png'
    # 按灰度值读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_block = divide2block(img)
    dct_block = get_dct_block(img_block)
    quantified_block = quantify_dct_block(dct_block, 5)

    relative_block_list = get_feature(quantified_block, 0.2, 0.2)

    # binary_img = get_binary_img(quantified_block, relative_block_list)
    # print(binary_img)
    # print(binary_img.shape)
    # print(binary_img[0][0].shape)

    # cv2.imshow("wname", binary_img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    # with open('tmps/tmp.csv', 'w') as f:
    #     for i in relative_block_list:
    #         f.write(str(i) + ' ' + str(len(relative_block_list[i])) + '\n')

    # # 输出测试
    # x, y = dct_block.shape
    # z, w = dct_block[0][0].shape
    # output_arr = []
    # tmp = []
    # for i in range(y):
    #     tmp.append(0)
    # for i in range(x):
    #     output_arr.append(tmp.copy())
    # for i in relative_block_list:
    #     if len(relative_block_list[i]) < 47:
    #         continue
    #     for j in relative_block_list[i]:
    #         output_arr[j[0][0]][j[0][1]] += 1
    #         output_arr[j[1][0]][j[1][1]] += 1
    # for i in range(x):
    #     for j in range(y):
    #         if output_arr[i][j] == 0:
    #             output_arr[i][j] = ' '
    #         else:
    #             output_arr[i][j] = '#'
    #
    # with open('tmps/result.txt', 'w') as f:
    #     for i in range(x):
    #         print(output_arr[i])
    #         f.write(str(output_arr[i]) + '\n')
    # 输出测试
    img_copy = img.copy()
    for i in relative_block_list:
        if len(relative_block_list[i]) < 48:
            continue
        for j in relative_block_list[i]:
            for k in range(2):
                x = j[k][0]
                y = j[k][1]
                for ii in range(8):
                    for jj in range(8):
                        img_copy[8 * x + ii][8 * y + jj] = 0

    cv2.namedWindow("Image")
    cv2.imshow("Image", img_copy)
    cv2.waitKey(0)
