import numpy as np
import cv2


# 按像素点分为(M-b+1)*(N-b+1)个8*8的矩阵块，本问题中为（505，505）
def divide2block(img: np.ndarray, step=8):
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


if __name__ == "__main__":
    img = cv2.imread('./data/001_F.png', cv2.IMREAD_GRAYSCALE)
    # with open('./tmps/001_F.csv', 'w') as f:
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             f.write(str(img[i][j]) + ',')
    #         f.write('\n')
    # print(img)

    img_block = divide2block(img)
    # print(img_block.shape)
    for i in range(img.shape[0] - 8, img.shape[0]):
        for j in range(img.shape[1] - 8, img.shape[1]):
            print(img[i][j], end=' ')
        print('\n')

    print(img_block[-1][-1])
    print(img_block[-1][-1].shape)
