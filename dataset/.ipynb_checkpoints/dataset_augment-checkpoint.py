from albumentations.augmentations.functional import image_compression
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import glob
import PIL
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


# if you want to run dataset.py, please import these stuff
# import augment
# from data_io.process_functions import random_crop, image_read, random_mask
# instead of
from dataset.data_io.process_functions import random_crop, image_read, random_mask

def normalize(img):
    img = img.astype(np.float32)
    img = img /127.5
    img = img - 1
    return img

class Cell(Dataset):
    def __init__(self, img_path, label_path, img_size=512, transform = None) -> None:
        super().__init__()
        self.imgs = glob.glob(img_path)
        self.imgs.sort()
        self.labels = glob.glob(label_path)
        self.labels.sort()
        self.img_size = img_size
        self.transform = transform
        
        assert len(self.imgs) > 0, "Can't find data;"
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img  = image_read(self.imgs[index])
        img = cv2.resize(img, (512,512))
        label = image_read(self.labels[index])[..., 0]
        label = cv2.resize(label, (512,512))
        
        if self.transform:
            aug_func = self.transform
            augmenter = aug_func(image=img, mask=label)
            img, label = augmenter["image"], augmenter["mask"]
            label = 1 - label/255.
            label = (label > 0.1).astype(np.float32)
            # label = label/255.
            # label = (label < 0.9).astype(np.float32)
            img, label = random_mask(img, label, num_mask=25, min_size=10, max_size=256)
            
        else:
            label = 1 - label/255.
            label = (label > 0.1).astype(np.float32)
            pass
        
        # label = label/255.
        label = 1 - label
        img = normalize(img)
        img = np.swapaxes(img, 2, 1)
        img = np.swapaxes(img, 1, 0)
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        
        return img, label

def visualize_augmentations(dataset, idx=0, samples=12, cols=6):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize))])
    rows = samples // cols * 2
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        img, msk = dataset[idx]
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        ax.ravel()[2*i].imshow(msk*255)
        ax.ravel()[2*i].set_axis_off()
        ax.ravel()[2*i+1].imshow(img)
        ax.ravel()[2*i+1].set_axis_off()
    plt.tight_layout()
    plt.show()

def get_dataset(name, img_path, mask_path, batch_size=1, shuffle=True, transform = None):
    # dataset = globals()[name](img_path, mask_path)
    dataset = Cell(img_path = img_path, label_path = mask_path, transform = transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=False,
        num_workers=6
#         num_workers = 1 # modified by xxy because of the lack of memory
    )
    return dataloader

if __name__ == '__main__':
    data_augment = augment.strong_aug()
    
    train_data = get_dataset(name='Cell',
                                img_path='E:/U-RISC_dataset/Simple_Track_Image/train/*.png',
                                mask_path='E:/U-RISC_dataset/Simple_Track_Label/train/*.png',
                                batch_size=8,
                                shuffle=False);
    
    train_data_augment = get_dataset(name='Cell',
                                img_path='E:/U-RISC_dataset/Simple_Track_Image/train/*.png',
                                mask_path='E:/U-RISC_dataset/Simple_Track_Label/train/*.png',
                                batch_size=8,
                                shuffle=False,
                                transform = data_augment);

    visualize_augmentations(dataset = train_data_augment.dataset)
    # for i,  (img, mask) in enumerate(train_data):
    #     print(img.shape, mask.shape)     # # [8, 3, 1024, 1024]  [8, 1024, 1024]     
    #     print(mask[0])
    #     print(img)
    #     print(i)