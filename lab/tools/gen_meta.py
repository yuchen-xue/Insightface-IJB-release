import numpy as np
from pathlib import Path


def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates_ = ijb_meta[:, 1].astype(np.int)
    medias_ = ijb_meta[:, 2].astype(np.int)
    return templates_, medias_


def read_template_pair_list(path):
    """
    origin: loadtxt(path, dtype=str)
    changed to: loadtxt(path, dtype=int)
    """
    pairs = np.loadtxt(path, dtype=int)
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label_ = pairs[:, 2].astype(np.int)
    return t1, t2, label_


if __name__ == '__main__':

    templates, medias = read_template_media_list(Path('../IJBC/meta/ijbc_face_tid_mid.txt'))
    np.save('ijbc_templates.npy', templates)
    np.save('ijbc_medias.npy', medias)

    p1, p2, label = read_template_pair_list(Path('../IJBC/meta/ijbc_template_pair_label.txt'))
    np.save('ijbc_p1.npy', p1)
    np.save('ijbc_p2.npy', p2)
    np.save('ijbc_labels.npy', label)
