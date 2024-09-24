import argparse
import h5py
import numpy as np
from PIL import Image as Image
from osgeo import gdal
from utils import calc_rmse
import torch


def Make_example(args):
    h5_file = h5py.File(args.output_path,'w')

    DEM = args.DEM

    lr_patchs = []
    hr_patchs = []
    hr_DEM = Image.open(DEM)
    hr_DEM = np.array(hr_DEM).astype(np.float32)
    num = 0

    for i in range(0,hr_DEM.shape[0] - args.patch_size + 1,args.stride):
        if num >= args.num:
            break
        for j in range(0,hr_DEM.shape[1] - args.patch_size + 1,args.stride):
            hr_patchs.append(hr_DEM[i:i + args.patch_size,j:j + args.patch_size])
            num = num + 1
            if num >= args.num:
                break
    for image in hr_patchs:
        lr_patchs.append(image[::args.scale, ::args.scale])

    hr_patchs = np.array(hr_patchs)
    lr_patchs = np.array(lr_patchs)

    h5_file.create_dataset('lr',data = lr_patchs)
    h5_file.create_dataset('hr',data = hr_patchs)
    h5_file.close()

if __name__ == '__main__':
    gdal.UseExceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEM',type=str,required=True)
    parser.add_argument('--output-path',type=str,required=True)
    parser.add_argument('--num',type=int,default=4000)
    parser.add_argument('--patch-size',type=int,default=192)
    parser.add_argument('--stride',type=int,default=14)
    parser.add_argument('--scale',type=int,default=2)
    args = parser.parse_args()

    Make_example(args)