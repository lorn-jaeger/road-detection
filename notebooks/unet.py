# %% [markdown]
# rendering placeholder

# %%
# Introduction
# In this notebook we use a UNet segmentation model for performing road segmentation on a custom TIFF dataset.

# Libraries 📚⬇
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

import segmentation_models_pytorch as smp
import tifffile

# ----------------------------------------------------------------------------
#   DATASET STRUCTURE
# ----------------------------------------------------------------------------
# Your TIFFs look like:
#    (H, W, 5):   [R, G, B, DEM, LABEL]
#
# The label is 0 for background and 1 for road.
#
# We'll split images & labels simply by reading TIFFs, slicing label,
# and not using separate directories for masks.
# ----------------------------------------------------------------------------

DATA_DIR = "data/tiffs/"   # adjust if needed

all_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".tiff")])

# Simple split (80/10/10)
n = len(all_files)
train_files = all_files[:int(0.8*n)]
val_files   = all_files[int(0.8*n):int(0.9*n)]
test_files  = all_files[int(0.9*n):]

# Class names and RGB for visualization (same as Mass Roads)
class_names = ['background', 'road']
class_rgb_values = [[0,0,0], [255,255,255]]

select_classes = class_names
select_class_rgb_values = np.array(class_rgb_values)

print("Classes:", class_names)
print("RGB:", class_rgb_values)

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx,(name,image) in enumerate(images.items()):
        plt.subplot(1,n_images,idx+1)
        plt.xticks([]); plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def one_hot_encode_binary(mask):
    """
    mask is HxW with values 0 or 1
    returns HxWx2 (background, road)
    """
    h,w = mask.shape
    result = np.zeros((h,w,2), dtype='float32')
    result[:,:,0] = (mask == 0)
    result[:,:,1] = (mask == 1)
    return result

def reverse_one_hot(mask):
    return np.argmax(mask, axis=-1)

def colour_code_segmentation(mask, label_values):
    return np.array(label_values)[mask.astype(int)]

# ----------------------------------------------------------------------------
# Dataset Class
# ----------------------------------------------------------------------------

class RoadsDataset(torch.utils.data.Dataset):

    def __init__(self, file_list, augmentation=None, preprocessing=None):
        self.files = file_list
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]

        # Read TIFF
        arr = tifffile.imread(path).astype('float32')   # (H,W,5)

        image = arr[:,:,:4]     # RGB + DEM
        label = arr[:,:,4]      # 0 or 1

        # Convert label → one-hot
        mask = one_hot_encode_binary(label)

        # Augment
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # Preprocess
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

# ----------------------------------------------------------------------------
# Visualize a Sample
# ----------------------------------------------------------------------------

dataset = RoadsDataset(train_files)
img, m = dataset[0]
viz_mask = colour_code_segmentation(reverse_one_hot(m), select_class_rgb_values)

visualize(
    original_image = img[:,:,:3],   # RGB only
    ground_truth_mask = viz_mask,
    one_hot_encoded_mask = reverse_one_hot(m)
)

# ----------------------------------------------------------------------------
# Augmentations
# ----------------------------------------------------------------------------

def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf([
            album.HorizontalFlip(p=1),
            album.VerticalFlip(p=1),
            album.RandomRotate90(p=1),
        ], p=0.75),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    return album.Compose([
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ])

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

# ----------------------------------------------------------------------------
# Visualize Augmented Samples
# ----------------------------------------------------------------------------

aug_dataset = RoadsDataset(
    train_files,
    augmentation=get_training_augmentation(),
)

sample_i = random.randint(0,len(aug_dataset)-1)
for _ in range(3):
    img, m = aug_dataset[sample_i]
    visualize(
        original_image = img[:,:,:3],
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(m), select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(m)
    )

# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
CLASSES = select_classes

model = smp.Unet(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = len(CLASSES),
    activation = ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = RoadsDataset(
    train_files,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

valid_dataset = RoadsDataset(
    val_files,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# ----------------------------------------------------------------------------
# Training Setup
# ----------------------------------------------------------------------------

TRAINING = True
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.00008)])

train_epoch = smp.utils.train.TrainEpoch(
    model, loss=loss, metrics=metrics,
    optimizer=optimizer, device=DEVICE, verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, loss=loss, metrics=metrics,
    device=DEVICE, verbose=True,
)

# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------

if TRAINING:
    best_iou_score = 0
    train_logs_list, valid_logs_list = [], []

    for i in range(EPOCHS):
        print(f"\nEpoch: {i}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        if valid_logs['iou_score'] > best_iou_score:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, "./best_model.pth")
            print("Model saved!")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

if os.path.exists("./best_model.pth"):
    best_model = torch.load("./best_model.pth", map_location=DEVICE)
    print("Loaded best model.")

test_dataset = RoadsDataset(
    test_files,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

test_loader = DataLoader(test_dataset)

test_vis = RoadsDataset(
    test_files,
    augmentation=get_validation_augmentation(),
)

def crop_image(image, target_dim=1500):
    size = image.shape[0]
    pad = (size - target_dim)//2
    if pad < 0: return image
    return image[pad:size-pad, pad:size-pad]

sample_preds = "sample_predictions/"
os.makedirs(sample_preds, exist_ok=True)

for idx in range(len(test_dataset)):
    img, gt = test_dataset[idx]
    img_vis = crop_image(test_vis[idx][0])

    x = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    pr = best_model(x).squeeze().detach().cpu().numpy()
    pr = np.transpose(pr, (1,2,0))

    pr_mask = reverse_one_hot(pr)
    gt_mask = reverse_one_hot(np.transpose(gt,(1,2,0)))

    pr_vis = crop_image(colour_code_segmentation(pr_mask, select_class_rgb_values))
    gt_vis = crop_image(colour_code_segmentation(gt_mask, select_class_rgb_values))

    cv2.imwrite(os.path.join(sample_preds, f"sample_{idx}.png"),
                np.hstack([img_vis[:,:,:3], gt_vis, pr_vis])[:,:,::-1])

    visualize(
        original_image=img_vis[:,:,:3],
        ground_truth_mask=gt_vis,
        predicted_mask=pr_vis
    )

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

test_epoch = smp.utils.train.ValidEpoch(
    model, loss=loss, metrics=metrics,
    device=DEVICE, verbose=True,
)

logs = test_epoch.run(test_loader)
print("\nEvaluation on Test Data:")
print(f"Mean IoU Score: {logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {logs['dice_loss']:.4f}")

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

train_df = pd.DataFrame(train_logs_list)
valid_df = pd.DataFrame(valid_logs_list)

plt.figure(figsize=(20,8))
plt.plot(train_df.index, train_df.iou_score, lw=3, label="Train")
plt.plot(valid_df.index, valid_df.iou_score, lw=3, label="Valid")
plt.title("IoU Score")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_df.index, train_df.dice_loss, lw=3, label="Train")
plt.plot(valid_df.index, valid_df.dice_loss, lw=3, label="Valid")
plt.title("Dice Loss")
plt.grid()
plt.legend()
plt.show()

# %%
# Introduction
# In this notebook we use a UNet segmentation model for performing road segmentation on a custom TIFF dataset.

# Libraries 📚⬇
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

from PIL import Image
import segmentation_models_pytorch as smp

# ----------------------------------------------------------------------------
#   DATASET STRUCTURE
# ----------------------------------------------------------------------------
# Your TIFFs:     (H, W, 5)
#      [R, G, B, DEM, LABEL]
#
# LABEL is 0 or 1
# ----------------------------------------------------------------------------

DATA_DIR = "data/tiffs/"

all_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".tiff")])

# 80/10/10 split
n = len(all_files)
train_files = all_files[:int(0.8*n)]
val_files   = all_files[int(0.8*n):int(0.9*n)]
test_files  = all_files[int(0.9*n):]

# Class names + RGB values (Mass Roads palette)
class_names = ['background', 'road']
class_rgb_values = [[0,0,0], [255,255,255]]

select_classes = class_names
select_class_rgb_values = np.array(class_rgb_values)

print("Classes:", class_names)
print("RGB:", class_rgb_values)

# ----------------------------------------------------------------------------
# Helper: Load multi-band TIFF via Pillow
# ----------------------------------------------------------------------------
def load_multiband_tiff(path):
    """
    Safe loader: handles multi-page TIFFs, stacking them into (H,W,C)
    """
    img = Image.open(path)
    bands = []
    try:
        i = 0
        while True:
            img.seek(i)
            arr = np.array(img)
            bands.append(arr)
            i += 1
    except EOFError:
        pass

    arr = np.stack(bands, axis=-1).astype('float32')
    return arr


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx,(name,image) in enumerate(images.items()):
        plt.subplot(1,n_images,idx+1)
        plt.xticks([]); plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def one_hot_encode_binary(mask):
    h,w = mask.shape
    out = np.zeros((h,w,2), dtype='float32')
    out[:,:,0] = (mask == 0)
    out[:,:,1] = (mask == 1)
    return out

def reverse_one_hot(mask):
    return np.argmax(mask, axis=-1)

def colour_code_segmentation(mask, label_values):
    return np.array(label_values)[mask.astype(int)]


# ----------------------------------------------------------------------------
# Dataset Class
# ----------------------------------------------------------------------------
class RoadsDataset(torch.utils.data.Dataset):

    def __init__(self, file_list, augmentation=None, preprocessing=None):
        self.files = file_list
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]

        # Load TIFF using Pillow
        arr = load_multiband_tiff(path)   # (H,W,5)

        image = arr[:,:,:4]      # RGB + DEM
        label = arr[:,:,4]       # 0/1 mask

        mask = one_hot_encode_binary(label)

        if self.augmentation:
            s = self.augmentation(image=image, mask=mask)
            image, mask = s["image"], s["mask"]

        if self.preprocessing:
            s = self.preprocessing(image=image, mask=mask)
            image, mask = s["image"], s["mask"]

        return image, mask


# ----------------------------------------------------------------------------
# Visualize Sample
# ----------------------------------------------------------------------------
dataset = RoadsDataset(train_files)
img, m = dataset[0]

viz_mask = colour_code_segmentation(reverse_one_hot(m), select_class_rgb_values)

visualize(
    original_image = img[:,:,:3],
    ground_truth_mask = viz_mask,
    one_hot_encoded_mask = reverse_one_hot(m)
)


# ----------------------------------------------------------------------------
# Augmentations
# ----------------------------------------------------------------------------
def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf([
            album.HorizontalFlip(p=1),
            album.VerticalFlip(p=1),
            album.RandomRotate90(p=1),
        ], p=0.75),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    return album.Compose([
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ])

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _t = []
    if preprocessing_fn:
        _t.append(album.Lambda(image=preprocessing_fn))
    _t.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_t)


# ----------------------------------------------------------------------------
# Visualize Augmented Samples
# ----------------------------------------------------------------------------
aug_dataset = RoadsDataset(
    train_files,
    augmentation=get_training_augmentation(),
)

sample_i = random.randint(0,len(aug_dataset)-1)
for _ in range(3):
    img, m = aug_dataset[sample_i]
    visualize(
        original_image = img[:,:,:3],
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(m), select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(m)
    )


# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
CLASSES = select_classes

model = smp.Unet(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = len(CLASSES),
    activation = ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = RoadsDataset(
    train_files,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

valid_dataset = RoadsDataset(
    val_files,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


# ----------------------------------------------------------------------------
# Training Setup
# ----------------------------------------------------------------------------
TRAINING = True
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.00008)])


train_epoch = smp.utils.train.TrainEpoch(
    model, loss=loss, metrics=metrics,
    optimizer=optimizer, device=DEVICE, verbose=True
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, loss=loss, metrics=metrics,
    device=DEVICE, verbose=True
)


# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------
if TRAINING:
    best_iou_score = 0
    train_logs_list, valid_logs_list = [], []

    for i in range(EPOCHS):
        print(f"\nEpoch: {i}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        if valid_logs['iou_score'] > best_iou_score:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, "./best_model.pth")
            print("Model saved!")


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if os.path.exists("./best_model.pth"):
    best_model = torch.load("./best_model.pth", map_location=DEVICE)
    print("Loaded best model.")

test_dataset = RoadsDataset(
    test_files,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

test_loader = DataLoader(test_dataset)

test_vis = RoadsDataset(
    test_files,
    augmentation=get_validation_augmentation(),
)


def crop_image(image, target_dim=1500):
    size = image.shape[0]
    pad = (size - target_dim)//2
    if pad < 0:
        return image
    return image[pad:size-pad, pad:size-pad]


sample_preds = "sample_predictions/"
os.makedirs(sample_preds, exist_ok=True)

for idx in range(len(test_dataset)):
    img, gt = test_dataset[idx]
    img_vis = crop_image(test_vis[idx][0])

    x = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    pr = best_model(x).squeeze().detach().cpu().numpy()
    pr = np.transpose(pr, (1,2,0))

    pr_mask = reverse_one_hot(pr)
    gt_mask = reverse_one_hot(np.transpose(gt,(1,2,0)))

    pr_vis = crop_image(colour_code_segmentation(pr_mask, select_class_rgb_values))
    gt_vis = crop_image(colour_code_segmentation(gt_mask, select_class_rgb_values))

    cv2.imwrite(os.path.join(sample_preds, f"sample_{idx}.png"),
                np.hstack([img_vis[:,:,:3], gt_vis, pr_vis])[:,:,::-1])

    visualize(
        original_image=img_vis[:,:,:3],
        ground_truth_mask=gt_vis,
        predicted_mask=pr_vis
    )


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
test_epoch = smp.utils.train.ValidEpoch(
    model, loss=loss, metrics=metrics,
    device=DEVICE, verbose=True
)

logs = test_epoch.run(test_loader)
print("\nEvaluation on Test Data:")
print(f"Mean IoU Score: {logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {logs['dice_loss']:.4f}")


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
train_df = pd.DataFrame(train_logs_list)
valid_df = pd.DataFrame(valid_logs_list)

plt.figure(figsize=(20,8))
plt.plot(train_df.index, train_df.iou_score, lw=3, label="Train")
plt.plot(valid_df.index, valid_df.iou_score, lw=3, label="Valid")
plt.title("IoU Score")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_df.index, train_df.dice_loss, lw=3, label="Train")
plt.plot(valid_df.index, valid_df.dice_loss, lw=3, label="Valid")
plt.title("Dice Loss")
plt.grid()
plt.legend()
plt.show()


