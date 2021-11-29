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

# if you want to run dataset.py, please import augment
# import augment

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
        img  = PIL.Image.open(self.imgs[index]).convert('RGB')
        label = PIL.Image.open(self.labels[index]).convert('1')
        
        # img = np.array(img)
        # label = np.array(label) # converted to "True"(?), I don't know if its will cause some trouble
        
        # img = img.permute(1, 2, 0)
        # label = label.permute(1, 2, 0)
        
        if self.transform:
            img = np.array(img)
            img = self.transform(image = img)["image"]
        else:
            img = self.resize(self.to_tensor(img))
            label = self.resize(self.to_tensor(label))[0]
        
        return img, label

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def get_dataset(name, img_path, mask_path, batch_size=1, shuffle=True, transform = None):
    dataset = globals()[name](img_path, mask_path)
    # dataset = Cell(img_path = img_path, label_path = mask_path, transform = transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=False,
        # num_workers=8
        num_workers = 1 # modified by xxy because of the lack of memory
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