from PIL import Image
import numpy as np
import torch
import monai.transforms as T


def get_transorms(
    new_shape, 
    bright_range=None, 
    rotation_range=None, 
    scale_range=None, 
    num_classes=None, 
    to_tensor=True, 
    probs=[0.5, 0.5, 0.5]):

    bright_prob, rot_prob, scale_prob = probs
    transform_list = []

    transform_list.append(T.NormalizeIntensityd(
        keys="img", 
        nonzero=True, 
        channel_wise=True)
    )

    transform_list.append(T.Resized(
        keys=["img", "seg"], 
        spatial_size=new_shape, 
        mode=["bilinear", "nearest"])
    )
    
    if bright_range is not None:
        transform_list.append(T.RandAdjustContrastd(
            keys="img", 
            prob=bright_prob, 
            gamma=bright_range)
        )

    if rotation_range is not None:
        transform_list.append(T.RandRotated(
            keys=["img", "seg"], 
            prob=rot_prob, 
            range_x=rotation_range, 
            mode=["bilinear", "nearest"])
        )
    if scale_range is not None:
        transform_list.append(T.RandZoomd(
            keys=["img", "seg"], 
            prob=scale_prob, 
            min_zoom=scale_range[0], 
            max_zoom=scale_range[1], 
            mode=["bilinear", "nearest"])
        )
    if num_classes is not None:
        transform_list.append(T.AsDiscreted(keys="seg", to_onehot=num_classes))

    if to_tensor == True:
        transform_list.append(T.ToTensord(keys=["img", "seg"], dtype=torch.float32))
        transform_list.append(T.EnsureTyped(keys=["img", "seg"], dtype=torch.float32))

    return T.compose.Compose(transform_list)



new_shape = (256, 512)
bright_range = (0.8, 1.2)
rotation_range = (-np.pi/36, np.pi/36)
scale_range = (0.8, 1.2)
num_classes=2
train_transform = get_transorms(
        new_shape, 
        bright_range=bright_range, 
        rotation_range=rotation_range, 
        scale_range=scale_range, 
        num_classes=num_classes
    )
'''img=Image.open("./data/Radiographs/1.jpg")
img.show()
msk = Image.open("./data/Segmentation/maxillomandibular/1.jpg").convert("1")
msk.show()
msk = np.asarray(Image.open("./data/Segmentation/maxillomandibular/1.jpg").convert("1"), dtype="float32")
img = np.asarray(Image.open("./data/Radiographs/1.jpg").convert("L"), dtype="float32") * msk
data = Image.fromarray(img)
data.show()'''
msk = np.asarray(Image.open("./data/Segmentation/maxillomandibular/1.jpg").convert("1"), dtype="float32")
seg = np.asarray(Image.open("./data/Segmentation/teeth_mask/1.jpg").convert("1"), dtype="float32")
img = np.asarray(Image.open("./data/Radiographs/1.jpg").convert("L"), dtype="float32") * msk
data = {
            "img": np.expand_dims(img, axis=0),
            "seg": np.expand_dims(seg, axis=0)
        }

data = train_transform(data)

print(data["img"][0])
datanum=data["img"][0].numpy()
print(datanum)
fff=datanum[0][0]
for i in range(len(datanum)):
    for j in range(len(datanum[0])):
        if(datanum[i][j]==fff):
            datanum[i][j]=0
        else:
            datanum[i][j]=255
            print(datanum[i][j],end="")
datanum = np.asarray(datanum, dtype="float32")
print(datanum)

img=Image.open("./data/Radiographs/1.jpg").convert("L")
img.show()
voila=Image.fromarray(datanum)
voila.show()