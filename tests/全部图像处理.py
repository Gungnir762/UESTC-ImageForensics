from main import divide2block
from torque import get_torque, get_equal_torque_block, vector_quantization, get_img_bin
import cv2
import numpy as np

if __name__ == "__main__":
    for i in range(1, 30):
        id = str(i).rjust(3, '0')
        image_path = '../data/' + id + '_F.png'
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_block = divide2block(img)
        torque_block = get_torque(img_block)
        equal_dict = get_equal_torque_block(torque_block)
        vector_dict = vector_quantization(equal_dict)
        img_bin = get_img_bin(img, vector_dict, threshold=1)
        cv2.imwrite('../output/' + id + '_F_bin.png', img_bin)
        print(rf'{image_path} done')
