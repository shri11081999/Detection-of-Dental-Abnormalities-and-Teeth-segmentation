import glob
import json
import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import functools

if __name__ == "__main__":

    imgs_list = sorted(glob.glob("data/Radiographs/*.JPG"))
    segs_list = sorted(glob.glob("data/Segmentation/teeth_mask/*.jpg"))
    masks_list = sorted(glob.glob("data/Segmentation/maxillomandibular/*.jpg"))

    train_ids, val_ids = train_test_split(range(len(imgs_list)), test_size=0.3, shuffle=True, random_state=12345)
    valid_ids, test_ids = train_test_split(val_ids, test_size=0.333, shuffle=True, random_state=6789)
    printf = functools.partial(print, end="")
    out = {
        "class_names": ["background", "teeth"],
        "class_weights": {},
        "train": [],
        "valid": [],
        "test": []
    }
    for k in ["train", "valid", "test"]:
        for id in tqdm.tqdm(eval(f"{k}_ids")):
            out[k].append(
                {
                    "img": imgs_list[id],
                    "seg": segs_list[id],
                    "msk": masks_list[id],
                }
            )
    num_classes = len(out["class_names"])
    num_samples_class = np.zeros(num_classes)
    print(num_samples_class)
    count=0
    count2=0
    print()
    print()
    print()
    for data in tqdm.tqdm(out["test"], total=len(out["test"])):
        file = data["seg"]
        seg = np.asarray(Image.open(file).convert("1"), dtype="uint8")
        for c in range(num_classes):
            num_samples_class[c] += len(seg[seg==c])
    num_samples = seg.shape[0] * seg.shape[1] * len(out["test"])
    class_weights = num_samples / (num_classes * num_samples_class)
    print(num_samples," ",num_samples_class[0])
    print(num_samples," ",num_samples_class[1])
    print(class_weights)
