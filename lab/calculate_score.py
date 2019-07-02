import numpy as np
import scipy.io as sio
import h5py

"""Generate similarity score array for one model"""


def normalize_feature(feats_):
    return feats_ / np.sqrt(np.sum(feats_ ** 2, -1, keepdims=True))


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))

    template_norm_feats = normalize_feature(template_feats)

    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1).flatten()
        score[s] = similarity_score
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def main():
    raw_feats = np.load('experiments/Arcface/VGG2-ResNet50-Arcface_img_feats.npy')

    # raw_feats = sio.loadmat('experiments/sphereface/sphereface-ijbc-affine-112X96.mat')['feature']

    # raw_feats = h5py.File('experiments/center-loss/center-loss-ijbc-affine-112X112.mat').get('feature')[()]

    # arr1 = h5py.File('experiments/center-loss/center-loss-ijbc-affine-112X96-20000.mat').get('feature')[()]
    # arr2 = h5py.File('experiments/center-loss/center-loss-ijbc-affine-112X96-40000.mat').get('feature2')[()]
    # raw_feats = np.concatenate((arr1, arr2))

    ijbc_templates = np.load('meta/ijbc_templates.npy')
    ijbc_medias = np.load('meta/ijbc_medias.npy')
    ijbc_p1 = np.load('meta/ijbc_p1.npy')
    ijbc_p2 = np.load('meta/ijbc_p2.npy')

    # normalise features to remove norm information
    # raw_feats = normalize_feature(raw_feats)
    template_norm_feats, unique_templates = image2template_feature(raw_feats, ijbc_templates, ijbc_medias)

    score = verification(template_norm_feats, unique_templates, ijbc_p1, ijbc_p2)
    np.save('experiments/VGG2-ResNet50-Arcface-no-norm-score.npy', score)


if __name__ == '__main__':
    main()
