from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow
from dataload import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=1, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=50, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=1, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=8, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")#1e-4
parser.add_argument("--img_size", default=256, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
#parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

def sample_data_and_noise(path, batch_size, image_size):

    dataset = MyDataset(path)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=False, batch_size=batch_size, num_workers=8
            )
            loader = iter(loader)
            yield next(loader)

if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    model_single = Glow(
        3, args.n_flow, 1, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    path="dataset/samples_1000x256x256x3.npz"
    dataset=iter(sample_data_and_noise(path, 1, 256))

    model.load_state_dict(torch.load("checkpoint/model_180001.pt"))
    print(torch.cuda.memory_allocated())
    model.eval()
    all_noise = []
    n_bins = 2.0 ** args.n_bits
    for k in range(100): 
        image,_=next(dataset)
        image=image.permute(0, 3, 1, 2).float()
        print(image.shape)
        image=image.to(torch.device("cuda"))
        print(torch.cuda.memory_allocated(),1)
        z_outs=model.module.reverse(image)
        del image
        print(torch.cuda.memory_allocated())
        print(z_outs.shape)
        all_noise.append((z_outs*80).detach().cpu().numpy())
        del z_outs
       
    arr = np.concatenate(all_noise, axis=0)
    np.savez("noise/data", all_noise)

#必须加--affine，不然模型参数不对
#CUDA_VISIBLE_DEVICES=7 python sam.py --affine