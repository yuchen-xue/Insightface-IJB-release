import numpy as np
from pathlib import Path
from timeit import default_timer

import tqdm

# TODO: make this runnable

TRUNK_SIZE = 2000


def concat_feats(feat_list):
    for i, feat_path in enumerate(tqdm.tqdm(feat_paths)):
        feat = np.load(feat_path)
        output_feat[i, :] = feat


feat_root = '../../../Facenet-workspace/features-IJBC-affine-112X112'
root_path = Path(feat_root)

output_feat = np.zeros([100, 512], np.float32)

feat_paths = [str(root_path.joinpath(f'{i}.npy')) for i in range(100)]
start = default_timer()


for i, feat_path in enumerate(tqdm.tqdm(feat_paths)):
    feat = np.load(feat_path)
    output_feat[i, :] = feat
end = default_timer()

print(f"Time elapsed: {end - start}")

np.save('facenet-IJBC-affine-112X112.npy', output_feat)
