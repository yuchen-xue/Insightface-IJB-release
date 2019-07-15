from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# TODO: clean up

# arr_list = []
# arr_root = '../../Face-feature-extract/features_output/IJBC/sphereface'
# arr_root = '../../Face-feature-extract/features_output/IJBC/sphereface'
# arr_root = '../../Face-feature-extract/features_output/testing/IJBC/sphereface'
# arr_root = '../../Face-feature-extract/features_output/testing/IJBC/sphereface-no-norm-input'
# arr_root = '../../Face-feature-extract/features_output/testing/IJBC/sphereface-loose_crop-no-norm-input'
# arr_root = '../../Face-feature-extract/features_output/testing/IJBC/sphereface-loose_crop-flip'
# arr_root = '../../Face-feature-extract/features_output/testing/IJBC/sphereface-flip'
# for i in range(1, 1025):
#     arr_path = Path(f'{arr_root}/{i}.npy')
#     arr = np.load(str(arr_path)).flatten()
#     arr_list.append(arr)
# img_feats = np.array(arr_list)

# img_feats = np.load('sphereface-norm_feats.npy')
# img_feats = np.load('sphereface-norm_feats.npy')
# img_feats = np.load('VGG2-ResNet50-Arcface_img_feats.npy')
# img_feats = np.load('VGG2-ResNet50-Arcface_feats_norm.npy')
# img_feats = np.load('MS1MV2-ResNet100-feature-simple.npy')
# img_feats = np.load('MS1MV2-ResNet100-feature-norm.npy')
# img_feats = np.load('../../Face-feature-extract/features_output/IJBC/center-loss-concat.npy')
# img_feats = np.load('../../Face-feature-extract/features_output/IJBC/sphereface-concat.npy')

t = img_feats[:128, :1024]
dist = cdist(t, t)
plt.matshow(dist)
plt.title(f"{Path(arr_root).relative_to('../../Face-feature-extract/features_output/testing/IJBC')}")
# plt.title(f"VGG2-ResNet50-Arcface_img_feat")
plt.show()
