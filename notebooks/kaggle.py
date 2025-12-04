# %% [markdown]
# ## Introduction
# 
# ### In this notebook we use [UNet](https://arxiv.org/abs/1505.04597) segmentation model for performing road segmentation on [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf).

# %% [markdown]
# ### Libraries 📚⬇

# %%
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
import tifffile

# %%
! uv pip install -q -U segmentation-models-pytorch albumentations tifffile > /dev/null
import segmentation_models_pytorch as smp

# %% [markdown]
# ### Defining train / val / test directories 📁

# %%
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(NOTEBOOK_DIR, 'data', 'tiffs')

all_files = sorted(
    [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(('.tif', '.tiff'))
    ]
)

if not all_files:
    raise RuntimeError(f"No TIFF files found in {DATA_DIR}.")

num_files = len(all_files)
train_end = int(0.8 * num_files)
valid_end = int(0.9 * num_files)

train_files = all_files[:train_end]
valid_files = all_files[train_end:valid_end]
test_files = all_files[valid_end:]

print(f"Found {num_files} TIFF files.")
print(f"Using {len(train_files)} for training, {len(valid_files)} for validation, and {len(test_files)} for testing.")

# %%
class_names = ['background', 'road']
class_rgb_values = [[0, 0, 0], [255, 255, 255]]
class_values = [0, 1]  # pixel values present in the mask channel

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

# %% [markdown]
# #### Shortlist specific classes to segment

# %%
# Useful to shortlist specific classes in datasets with large number of classes
select_classes = class_names
select_class_rgb_values = np.array(class_rgb_values)
select_class_values = class_values

print('Selected classes and their corresponding RGB values in labels:')
print('Class Names: ', select_classes)
print('Class RGB values: ', select_class_rgb_values.tolist())

# %% [markdown]
# ### Helper functions for viz. & one-hot encoding/decoding

# %%
# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes.
    # Arguments
        label: 2D (HxW) array segmentation image label or 3D RGB mask
        label_values: list of scalar pixel values or RGB triplets
        
    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    label = np.asarray(label)
    for colour in label_values:
        colour = np.array(colour)
        if label.ndim == 3 and colour.size > 1:
            equality = np.equal(label, colour)
            class_map = np.all(equality, axis=-1)
        else:
            class_map = np.equal(label, colour).astype('bool')
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype('float32')

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# %%
class RoadsDataset(torch.utils.data.Dataset):

    """Custom TIFF Dataset. Read stacked TIFFs, apply augmentation and preprocessing transformations.
    
    Args:
        file_paths (list[str]): collection of TIFF file paths
        class_values (list[int]): possible values present in the mask channel
        augmentation (albumentations.Compose): data transformation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    """
    
    def __init__(
            self, 
            file_paths, 
            class_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.file_paths = file_paths
        self.class_values = class_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read stacked TIFF
        data = tifffile.imread(self.file_paths[i]).astype('float32')
        image = data[:, :, :3].astype('uint8')
        mask = data[:, :, -1]
        mask = (mask > 127).astype('uint8')
        
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_values)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.file_paths)

# %% [markdown]
# #### Visualize Sample Image and Mask 📈

# %%
dataset = RoadsDataset(train_files, class_values=select_class_values)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[random_idx]

visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask)
)

# %% [markdown]
# ### Defining Augmentations 🙃

# %%
def get_training_augmentation():
    train_transform = [    
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

# %% [markdown]
# #### Visualize Augmented Images & Masks

# %%
augmented_dataset = RoadsDataset(
    train_files, 
    class_values=select_class_values,
    augmentation=get_training_augmentation(),
)

random_idx = random.randint(0, len(augmented_dataset)-1)

# Different augmentations on a random image/mask pair (256*256 crop)
for i in range(3):
    image, mask = augmented_dataset[random_idx]
    visualize(
        original_image = image,
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask)
    )

# %% [markdown]
# ## Training UNet

# %% [markdown]
# <h3><center>UNet Model Architecture</center></h3>
# <img src="https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png" width="750" height="750"/>
# <h4><center><a href="https://arxiv.org/abs/1505.04597">Image Courtesy: UNet [Ronneberger et al.]</a></center></h4>

# %% [markdown]
# ### Model Definition

# %%
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = select_classes
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# %% [markdown]
# #### Get Train / Val DataLoaders

# %%
# Data loader configuration
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 1
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Get train and val dataset instances
train_dataset = RoadsDataset(
    train_files, 
    class_values=select_class_values,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

valid_dataset = RoadsDataset(
    valid_files, 
    class_values=select_class_values,
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)

# Get train and val data loaders
train_loader = DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
valid_loader = DataLoader(
    valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


def run_epoch(model, loader, device, loss_fn, metric_fn=None, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            if metric_fn is not None:
                batch_iou = metric_fn(outputs, masks)
            else:
                batch_iou = compute_iou(outputs, masks)
        total_iou += batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou

    num_batches = len(loader)
    return {
        'dice_loss': total_loss / num_batches,
        'iou_score': total_iou / num_batches,
    }

# %% [markdown]
# #### Set Hyperparams

# %%
# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 5

# Set device: `cuda` or `cpu`
def select_device(min_free_gb=1.5):
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            print(f"CUDA detected with {free_gb:.2f} / {total_gb:.2f} GB free.")
            if free_gb >= min_free_gb:
                return torch.device("cuda")
            else:
                print(f"Only {free_gb:.2f} GB free on GPU. Falling back to CPU.")
        except Exception as exc:
            print(f"Unable to query CUDA memory ({exc}). Falling back to CPU.")
    return torch.device("cpu")


DEVICE = select_device()
model = model.to(DEVICE)

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

# define loss function
loss_fn = smp.losses.DiceLoss(mode='multilabel', from_logits=False)

# define metrics
def compute_iou(preds, targets, smooth=1e-7):
    """Compute IoU for the road class (index 1) using argmax predictions."""
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    pred_road = (preds == 1)
    target_road = (targets == 1)
    intersection = (pred_road & target_road).float().sum()
    union = (pred_road | target_road).float().sum()
    return (intersection + smooth) / (union + smooth)

# define optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.00008),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists('../input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth'):
    model = torch.load('../input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth', map_location=DEVICE)

# %% [markdown]
# ### Training UNet

# %%
%%time

if TRAINING:

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = run_epoch(model, train_loader, DEVICE, loss_fn, optimizer=optimizer)
        valid_logs = run_epoch(model, valid_loader, DEVICE, loss_fn, optimizer=None)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

# %% [markdown]
# ### Prediction on Test Data

# %%
# load best saved model checkpoint from the current run
if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location=DEVICE)
    best_model = best_model.to(DEVICE)
    print('Loaded UNet model from this run.')

# load best saved model checkpoint from previous commit (if present)
elif os.path.exists('../input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth'):
    best_model = torch.load('../input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth', map_location=DEVICE)
    best_model = best_model.to(DEVICE)
    print('Loaded UNet model from a previous commit.')

# %%
# create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
test_dataset = RoadsDataset(
    test_files, 
    class_values=select_class_values,
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# test dataset for visualization (without preprocessing transformations)
test_dataset_vis = RoadsDataset(
    test_files, 
    class_values=select_class_values,
    augmentation=get_validation_augmentation(),
)

# get a random test image/mask index
random_idx = random.randint(0, len(test_dataset_vis)-1)
image, mask = test_dataset_vis[random_idx]

visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask)
)

# Notice the images / masks are 1536*1536 because of 18px padding on all sides. 
# This is to ensure the input image dimensions to UNet model are a multiple of 2 (to account for pooling & transpose conv. operations).

# %%
# Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims=[1500,1500,3]):
   
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    if padding<0:
        return image

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

# %%
sample_preds_folder = 'sample_predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)

# %%
for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask,(1,2,0))
    # Get prediction channel corresponding to road
    pred_road_heatmap = pred_mask[:,:,select_classes.index('road')]
    pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
    # Convert gt_mask from `CHW` format to `HWC` format
    gt_mask = np.transpose(gt_mask,(1,2,0))
    gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
    
    visualize(
        original_image = image_vis,
        ground_truth_mask = gt_mask,
        predicted_mask = pred_mask,
        predicted_road_heatmap = pred_road_heatmap
    )

# %% [markdown]
# ### Model Evaluation on Test Dataset

# %%
valid_logs = run_epoch(model, test_dataloader, DEVICE, loss_fn, optimizer=None)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

# %% [markdown]
# ### Plot Dice Loss & IoU Metric for Train vs. Val

# %%
train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
train_logs_df.T

# %%
plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=21)
plt.ylabel('IoU Score', fontsize=21)
plt.title('IoU Score Plot', fontsize=21)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('iou_score_plot.png')
plt.show()

# %%
plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=21)
plt.ylabel('Dice Loss', fontsize=21)
plt.title('Dice Loss Plot', fontsize=21)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('dice_loss_plot.png')
plt.show()
