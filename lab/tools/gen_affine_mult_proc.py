import os
from pathlib import Path
from functools import partial, wraps
from multiprocessing import Pool

from tqdm import tqdm
from skimage import transform as trans
import numpy as np
import cv2

# TODO: make this runnable

IMG_LMK_LIST = '../IJBC/meta/ijbc_name_5pts_score.txt'
IMG_ROOT = '../IJBC/loose_crop'
OUT = '../IJBC/affine-112X96'
N_THREAD = 30


def read_img_lmk_list(list_path):
    with open(list_path) as f:
        content = f.readlines()

    for line in content:
        item = line.strip().split(' ')
        img_name = item[0]
        lmk = np.array([float(x) for x in item[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))

        yield img_name, lmk


SRC = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)
SRC[:, 0] += 8.0
IMG_SIZE = (112, 96)

p_out = Path(OUT)
if not p_out.exists():
    p_out.mkdir(exist_ok=True)

# for img_name, lmk in read_img_lmk_list(IMG_LMK_LIST):


def gen_affine(img_name_, lmk_):
    img = cv2.imread(os.path.join(IMG_ROOT, img_name_))
    tform = trans.SimilarityTransform()
    tform.estimate(lmk_, SRC)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, (IMG_SIZE[1], IMG_SIZE[0]), borderValue=0.0)

    cv2.imwrite(os.path.join(OUT, img_name_), img)

    print(img_name_)
    

if __name__ == "__main__":
    print("Loading image list ... ")
    img_lmk_list = list(read_img_lmk_list(IMG_LMK_LIST))

    pool = Pool(N_THREAD)
    pool.starmap(gen_affine, img_lmk_list)
    pool.close()
    pool.join()

    