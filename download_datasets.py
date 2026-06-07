
import gdown
import os

file_id = "1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB"

gdown.download(
    id=file_id, output="./datasets/dataset.zip", quiet=False
)

gdown.extractall("./datasets/dataset.zip")
os.remove("./datasets/dataset.zip")
