"""
Pack several IJBC features files into one single file
"""

import numpy as np
from pathlib import Path
from timeit import default_timer

import tqdm


feat_root = '../../../Facenet-workspace/features-IJBC-affine-112X112'
root_path = Path(feat_root)

output_feat = np.zeros([469375, 512], np.float32)

start = default_timer()
for i in tqdm.tqdm(range(469375)):
    feat_path = str(root_path.joinpath(f'{i}.npy'))
    feat = np.load(feat_path)
    output_feat[i, :] = feat
end = default_timer()

print(f"Time elapsed: {end - start}")

np.save('facenet-IJBC-affine-112X112.npy', output_feat)
