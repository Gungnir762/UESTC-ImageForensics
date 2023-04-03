import os

import numpy as np

from divide import divide2block
from getFeature import get_feature
from main import get_dct_block, quantify_dct_block, get_img_masked_and_bin
import cv2


def main_solution(img):
    img_block = divide2block(img)
    dct_block = get_dct_block(img_block)
    quantified_block = quantify_dct_block(dct_block)

    relative_block_list = get_feature(quantified_block, 0.2, 10, 500)

    img_masked, img_bin = get_img_masked_and_bin(img, relative_block_list)
    return img_masked, img_bin


if __name__ == "__main__":
    for i in range(1, 30):
        id = str(i).rjust(3, '0')
        # id = '015'
        image_path = os.path.realpath('../data/' + id + '_F.png')
        print(rf'{image_path} processing')

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            img_masked, img_bin = main_solution(img)
        except Exception as e:
            print(rf'{image_path} error')
            print(e)
            continue
        tmp = np.hstack((img, img_masked, img_bin))

        cv2.imwrite('../output/' + id + '_F_compare.png', tmp)
        print(rf'{image_path} done')
