import cv2
import numpy as np

from GetFeature import get_zigzag, get_feature  # get_feature, cal_dis_vec, cmp,


# 计算矩阵的DCT变换
def get_dct_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix_float = np.float32(matrix)
    matrix_dct = cv2.dct(matrix_float)
    return matrix_dct


# 将原二维矩阵按8*8的分块，返回分块后的矩阵
def divide2block(matrix: np.ndarray, step=8) -> np.ndarray:
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
    :param img_block: 以8*8矩阵为元素的矩阵
    :return: 以DCT变换后的8*8矩阵为元素的矩阵
    """
    r, c = img_block.shape
    dct_block = np.zeros((r, c), dtype=np.ndarray)
    for i in range(r):
        for j in range(c):
            dct_block[i][j] = get_dct_matrix(img_block[i][j])
    return dct_block


if __name__ == '__main__':
    image_path = './data/015_F.png'
    # 按灰度值读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_block = divide2block(img)
    block_dct = get_dct_block(img_block)

    # 简单DCT量化
    x, y = block_dct.shape
    z, w = block_dct[0][0].shape
    block_dct_int=np.empty((x,y,z,w),dtype=int,order='C')
    Q = 5  # Q越大越精确
    for xi in range(x):
        for yi in range(y):
            for zi in range(z):
                for wi in range(w):
                    block_dct_int[xi][yi][zi][wi] = int(block_dct[xi][yi][zi][wi] * Q)


    block_list = get_feature(block_dct_int,0.2,0.2)
    with open('./tmp.csv', 'w') as f:
        for i in block_list:
            f.write(str(i) + ' ' + str(len(block_list[i])) + '\n')

    # 输出测试
    arr = []
    tmp = []
    for i in range(y):
        tmp.append(0)
    for i in range(x):
        arr.append(tmp.copy())
    for i in block_list:
        if len(block_list[i])<47 :
            continue
        for j in block_list[i]:
            arr[j[0][0]][j[0][1]] += 1
            arr[j[1][0]][j[1][1]] += 1
    for i in range(x):
        for j in range(y):
            if arr[i][j]==0:
                arr[i][j]=' '
            else:
                arr[i][j]='#'
    for i in range(x):
        print(arr[i])
