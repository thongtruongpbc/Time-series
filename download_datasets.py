# Refs: https://github.com/JovianHQ/opendatasets
# gg drive for datasets: https://drive.usercontent.google.com/download?id=1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB&export=download&authuser=0
# can use: gdown --fuzzy "https://drive.google.com/file/d/1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB/view"

import gdown
import os

file_id = "1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB"

gdown.download(
    id=file_id, output="/home/cds/mnt/thongtx/datasets/dataset.zip", quiet=False
)

gdown.extractall("/home/cds/mnt/thongtx/datasets/dataset.zip")
os.remove("/home/cds/mnt/thongtx/datasets/dataset.zip")
