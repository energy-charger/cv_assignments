# encoding: utf-8
"""
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(255 / np.sqrt(mse))


# import torch, gc
#
# gc.collect()
# torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', default='woman.png', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

CUDA = torch.cuda.is_available()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if CUDA:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
imgori = image.copy()
image = image.resize((int(image.size[0] / opt.upscale_factor), int(image.size[1] / opt.upscale_factor)), Image.BICUBIC)
image = image.resize((int(image.size[0] * opt.upscale_factor), int(image.size[1] * opt.upscale_factor)), Image.BICUBIC)
image.save(f"BICUBIC_{opt.image_name}")

with torch.no_grad():
    image = Variable(ToTensor()(image))
if CUDA:
    image = image.cuda()
image = image[:3].unsqueeze(0)

start = time.perf_counter()
out = model(image)
elapsed = (time.perf_counter() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
imgfin = np.array(out_img)
imgori = np.array(imgori)
print(psnr(imgori, imgfin))
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
