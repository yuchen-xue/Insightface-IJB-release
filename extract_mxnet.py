"""
A naive way to extract features with mxnet models
"""
import timeit
import os
import numpy as np
import cv2
import tqdm
from recognition.embedding import Embedding


def get_image_feature(img_path_, img_list_path_, model_path_, gpu_id_):
    img_list = open(img_list_path_)
    embedding = Embedding(model_path_, 1, gpu_id_)
    files = img_list.readlines()
    img_feats_ = []
    for img_index, each_line in enumerate(tqdm.tqdm(files)):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path_, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_feats_.append(embedding.get(img, lmk))
    img_feats_ = np.array(img_feats_).astype(np.float32)
    return img_feats_


img_path = './IJBC/loose_crop'
img_list_path = './IJBC/meta/ijbc_name_5pts_score.txt'
model_path = './pretrained_models/r100-cosface-emore-epoch1-5/model'
gpu_id = [0]
start = timeit.default_timer()
img_feats = get_image_feature(img_path, img_list_path, model_path, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
np.save("cosface_feat.npy", img_feats)
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
