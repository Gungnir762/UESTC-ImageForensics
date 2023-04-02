# 按矩判断相似度
import numpy as np
import cv2
from main import get_img_bin, get_img_mask

if __name__ == '__main__':
    image_path = './data/015_F.png'
    # 按灰度值读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
