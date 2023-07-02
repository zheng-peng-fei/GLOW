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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 按照 PCI_BUS_ID 排序
os.environ["CUDA_VISIBLE_DEVICES"] = "6"         # 设置第一块GPU可见


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=2, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=256, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
#parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)

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



def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer):
    #dataset = iter(sample_data(args.path, args.batch, args.img_size))
    path="dataset/samples_1000x256x256x3.npz"
    dataset=iter(sample_data_and_noise(path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, noise = next(dataset)
            image=image.permute(0, 3, 1, 2)
            image = image.to(device)
            print(torch.cuda.memory_allocated())
            noise=noise/80
            noise= noise.to(device)
            #image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, z_outs = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, z_outs = model(image + torch.rand_like(image) / n_bins)

            b_size,channel,height,width=z_outs[0].shape
            channel=channel//2
            height=height*2
            width=width*2
            #16,3,64,64
            noise1=torch.cat([z_outs[3].reshape(b_size,channel*8,height//8,width//8),z_outs[2]],dim=2)
            noise2=torch.cat([noise1.reshape(b_size,channel*4,height//4,width//4),z_outs[1]],dim=2)
            noise3=torch.cat([noise2.reshape(b_size,channel*2,height//2,width//2),z_outs[0]],dim=2)
            noise3=noise3.reshape(b_size,channel,height,width)
            print(noise3.shape)
            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            print(noise3[0][0][0][2])
            print(noise[0][0][0][2])

            tensor=(noise-noise3)**2
            distance=tensor.mean(dim=list(range(1, len(tensor.shape))))
            model.zero_grad()

            loss=torch.mean(distance)
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    #model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)