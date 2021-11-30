import torch
from torch.utils.data import DataLoader, Dataset
import glob
import PIL
import numpy as np
import os
import torchvision.transforms as transforms

class Cell(Dataset):
    def __init__(self, img_path, label_path, img_size=512) -> None:
        super().__init__()
        self.imgs = glob.glob(img_path)
        self.imgs.sort()
        self.labels = glob.glob(label_path)
        self.labels.sort()
        self.img_size = img_size
        
        assert len(self.imgs) > 0, "Can't find data;"
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img  = PIL.Image.open(self.imgs[index]).convert('RGB')
        img = self.resize(self.to_tensor(img))
        label = PIL.Image.open(self.labels[index]).convert('1')
        label = self.resize(self.to_tensor(label))[0]

        # img = img.permute(1, 2, 0)
        # label = label.permute(1, 2, 0)
        return img, label


def get_dataset(name, img_path, mask_path, batch_size=1, shuffle=True):
    dataset = globals()[name](img_path, mask_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=False,
        num_workers=4
    )
    return dataloader


if __name__ == '__main__':
    train_data = get_dataset(name='Cell',
                                img_path='E:/U-RISC_dataset/Simple_Track_Image/train/*.png',
                                mask_path='E:/U-RISC_dataset/Simple_Track_Label/train/*.png',
                                batch_size=8,
                                shuffle=False)

    for i,  (img, mask) in enumerate(train_data):
        print(img.shape, mask.shape)     # # [8, 3, 1024, 1024]  [8, 1024, 1024]     
        print(mask[0])