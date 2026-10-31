
import os
import logging
from glob import glob
from argparse import ArgumentParser
import shutil
import re

parser = ArgumentParser("metrics")
parser.add_argument("--path", "-s", required=True, type=str)
parser.add_argument("--iteration", "-i", required=False, type=str, default='10000')
args = parser.parse_args()
PSNR = 0
SSIM = 0
LPIPS = 0


metric_path = os.path.join(args.path, 'metrics_mean.txt')
if os.path.exists(metric_path):
    os.remove(metric_path)

dir_lst = glob(args.path + '/*')
for d in dir_lst:
    with open (os.path.join(d, 'metrics_{}.txt'.format(args.iteration)), 'r') as f:
        l = f.readline()
        psnr = re.sub(r'[^0-9.]', '', l)
        print(d, psnr)
        PSNR += float(psnr)
        l = f.readline()
        ssim = re.sub(r'[^0-9.]', '', l)
        SSIM += float(ssim)
        l = f.readline()
        lpips = re.sub(r'[^0-9.]', '', l)
        LPIPS += float(lpips)

PSNR /= len(dir_lst)
SSIM /= len(dir_lst)
LPIPS /= len(dir_lst)

with open(metric_path, 'w') as f:
    f.write('PSNR : {}\n'.format(PSNR))
    f.write('SSIM : {}\n'.format(SSIM))
    f.write('LPIPS : {}\n'.format(LPIPS))
    
print(PSNR)
print(SSIM)
print(LPIPS)