import argparse
import logging
import sys
from pathlib import Path
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from utils.data_loading import BasicDataset, CarvanaDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate
from unet import UNet
from dataset.dataset import *
import unet
from utils.dice_score import dice_loss
from evaluate import evaluate

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Cell')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument('--config', is_config_file=True, default='configs/cell.txt',  # change
                        help='config file path')
    
    # parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-4,
    #                     help='Learning rate', dest='lr')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    
    parser.add_argument("--batch_size", type=int, default=1, # change
                        help='num of total epoches')
    parser.add_argument("--nepoch", type=int, default=401,  # change
                        help='num of total epoches')
    
    parser.add_argument("--i_val",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=5000, 
                        help='frequency of weight ckpt saving')
    
    parser.add_argument("--i_schedular", type=int, default=3000, 
                        )
    
    parser.add_argument("--img_train_datadir", type=str, default='/data1/lttt/Simple_Track_Image/train/*.png')
    parser.add_argument("--label_train_datadir", type=str, default='/data1/lttt/Simple_Track_Label/train/*.png') 
    parser.add_argument("--img_val_datadir", type=str, default='/data1/lttt/Simple_Track_Image/val/*.png')
    parser.add_argument("--label_val_datadir", type=str, default='/data1/lttt/Simple_Track_Label/val/*.png')


    return parser


def train_net():
    parser = config_parser()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.to(device)
    
    train_dataset = get_dataset(name=args.name,
                                img_path=args.img_train_datadir,
                                mask_path=args.label_train_datadir,
                                batch_size=args.batch_size,
                                shuffle=True)

    val_dataset = get_dataset(name=args.name,
                                img_path=args.img_val_datadir,
                                mask_path=args.label_val_datadir,
                                batch_size=1,
                                shuffle=False)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    pbar = tqdm(total=args.nepoch * len(train_dataset))

    val_data_iter = iter(val_dataset)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_dir = os.makedirs(os.path.join('logs', 'events', TIMESTAMP))
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.nepoch):
        for i,  (img, mask) in enumerate(train_dataset):
            # print(img.shape, mask.shape)     # # [8, 3, 1024, 1024]  [8, 1024, 1024]     
            assert img.shape[1] == net.n_channels
            images = img.to(device=device, dtype=torch.float32)
            true_masks = mask.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=args.amp):
                masks_pred = net(images)
                # # masks_pred: batch_size x 2 x H x W (dtype:float32) ; true_masks: batch_size x H x W (dtype:torch.long)
                loss = criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                    multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            writer.add_scalar('Loss/train', loss.item(),global_step)
            # log learning rate
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            
            pbar.update(1)

            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            # if global_step % args.i_schedular == 0 or global_step == 10:
            #     val_score = evaluate(net, val_dataset, device)
            #     scheduler.step(val_score)
            if global_step % args.i_val == 0:
                try:
                    img, mask = next(val_data_iter)
                except StopIteration:
                    val_data_iter = iter(val_dataset)
                    img, mask = next(val_data_iter)
                images = img.to(device=device, dtype=torch.float32)
                true_masks = mask.to(device=device, dtype=torch.long)
                net.eval()
                with torch.no_grad():
                    masks_pred = net(images)
                    # # masks_pred: batch_size x 2 x H x W (dtype:float32) ; true_masks: batch_size x H x W (dtype:torch.long)
                    loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
                pred_zero_one = torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu() ## [0]: get first one in batch_sizse
                writer.add_image('pred_mask/val', torch.cat((pred_zero_one, true_masks[0].float().cpu()), dim=-1), global_step, dataformats='HW')
                writer.add_scalar('Loss/val', loss.item(), global_step)
                net.train()

            if global_step % args.i_print == 0:
                tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss.item()} ")
            if global_step % args.i_weight == 0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'unet_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            global_step += 1
            


if __name__ == '__main__':

    train_net()
