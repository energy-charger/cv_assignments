import argparse

import numpy as np


import math
import torch
import torchvision.transforms as transforms
from PIL import Image


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(255 / np.sqrt(mse))


parser = argparse.ArgumentParser(description='SRCNN run parameters')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

img = Image.open(args.image).convert('YCbCr')
imgori = img.copy()
img = img.resize((int(img.size[0] / args.zoom_factor), int(img.size[1] / args.zoom_factor)),
                 Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
img = img.resize((int(img.size[0] * args.zoom_factor), int(img.size[1] * args.zoom_factor)),
                 Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
img2 = img.convert('RGB')
img2.save(f"BICUBIC_{args.image}")

y, cb, cr = img.split()

img_to_tensor = transforms.ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
print(device)
model = torch.load(args.model).to(device)
input = input.to(device)

out = model(input)
out = out.cpu()
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert(
    'RGB')  # we merge the output of our network with the upscaled Cb and Cr from before
# before converting the result in RGB
imgori = np.array(imgori)
imgfin = np.array(out_img)
print(psnr(imgori, imgfin))
out_img.save(f"zoomed_{args.image}")
