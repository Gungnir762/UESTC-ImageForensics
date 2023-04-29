"""
author:zyr
function:创建一个默认的白色图片
notice:None
"""
import cv2
import numpy as np

if __name__ == '__main__':
    default_img = np.ones((512, 512), dtype=np.uint8)*255
    cv2.imwrite('./default.jpg', default_img)
