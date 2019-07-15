from pathlib import Path
from itertools import chain

IJB_root = Path('./IJBB/loose_crop')
target_dir = Path('./IJBB-samples/loose_crop')
target_dir.mkdir(exist_ok=True)

for i in chain(range(1, 50), range(6000, 7000)):
    target_file = IJB_root / f'{i}.jpg'
    des_file = target_dir / f'{i}.jpg'
    des_file.symlink_to(str(Path('..' / target_file)))

