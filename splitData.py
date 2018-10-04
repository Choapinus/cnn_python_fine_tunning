# how to use:
# python splitData.py --help
# or
# python splitData.py --dataset /path/to/dataset/ --n_images 1000 --train 0.9
# output: directory named "dataset" with data separated by classes with train/validation folders
# dataset original format:
# 
# dataset_folder
# ├── class1
# │      ├── im1.jpg
# │      ├── im2.jpg
# │      ├── ...
# │      └── imN.jpg
# ├── class2
# │      ├── im1.jpg
# │      ├── im2.jpg
# │      ├── ...
# │      └── imN.jpg
# ├── ...
# └── classN
#        ├── im1.jpg
#        ├── im2.jpg
#        ├── ...
#        └── imN.jpg
#
#
# new dataset format:
# dataset
#   ├── train
#   │   ├── class1
#   │   │   ├── 000001.jpg
#   │   │   ├── 000002.jpg
#   │   │   ├── 000004.jpg
#
#           ...
#
#   │   ├── classN
#   │   │   ├── 000001.jpg
#   │   │   ├── 000002.jpg
#   │   │   ├── 000004.jpg
#   |
#   ├── validation
#   │   ├── class1
#   │   │   ├── 000001.jpg
#   │   │   ├── 000002.jpg
#   │   │   ├── 000004.jpg
#
#           ...
#
#   │   ├── classN
#   │   │   ├── 000001.jpg
#   │   │   ├── 000002.jpg
#   │   │   ├── 000004.jpg




import os
import argparse
from shutil import copy, rmtree

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    required=True,
    help="path to dataset"
)
ap.add_argument(
    "-n",
    "--n_images",
    type=int,
    required=False,
    default=2000,
    help="number of images per class"
)
ap.add_argument(
    "-t",
    "--train",
    type=float,
    required=False,
    default=0.8,
    help="percent of train images. The different will be for validation"
)

args = vars(ap.parse_args())

if os.path.isdir("./dataset"):
	rmtree("./dataset")

dirs = os.listdir(args["dataset"])
os.mkdir("./dataset")
os.mkdir("./dataset/train")
os.mkdir("./dataset/validation")

for _dir in dirs:
	os.mkdir(os.path.join('./dataset/train', _dir))
	os.mkdir(os.path.join('./dataset/validation', _dir))

train_images = int(args["n_images"] * args["train"])
images_dir = {}

for _dir in dirs:
	images_dir[_dir] = os.listdir(os.path.join(args["dataset"], _dir))
	images_dir[_dir].sort()

# dataset/train/key_dict

for i in range(train_images):
	for key in images_dir:
		src = os.path.join(args["dataset"], key, images_dir[key][i])
		dst = os.path.join('./dataset/train', key, images_dir[key][i])
		copy(src, dst)

for i in range(train_images, args["n_images"]):
	for key in images_dir:
		src = os.path.join(args["dataset"], key, images_dir[key][i])
		dst = os.path.join('./dataset/validation', key, images_dir[key][i])
		copy(src, dst)