import argparse
import h5py
import numpy as np
from PIL import Image as Image
from osgeo import gdal


def hr_to_lr(pic,scale):
    pic = pic.resize((pic.width // scale,pic.height // scale),resample=Image.BICUBIC)
    pic = pic.resize((pic.width * scale,pic.height * scale),resample=Image.BICUBIC)
    return pic

def ToImage(pic,num):
    band = pic.GetRasterBand(num)
    array = band.ReadAsArray()
    image = Image.fromarray(array)
    return image

def train(args):
    h5_file = h5py.File(args.output_path,'w')
    RS = args.RS
    DEM = args.DEM

    lr_patchs_1 = []
    lr_patchs_2 = []
    lr_patchs_3 = []
    lr_patchs_4 = []

    hr_patchs_1 = []
    hr_patchs_2 = []
    hr_patchs_3 = []
    hr_patchs_4 = []

    hr_RS = gdal.open(RS)
    hr_DEM = Image.open(DEM)

    red_RS = ToImage(hr_RS,1)
    blue_RS = ToImage(hr_RS,2)
    green_RS = ToImage(hr_RS,3)

    hr_width = (red_RS.width // args.scale) * args.scale
    hr_height = (red_RS.height // args.scale) * args.scale

    red_RS = red_RS.resize((hr_width,hr_height),resample=Image.BICUBIC)
    blue_RS = blue_RS.resize((hr_width,hr_height),resample=Image.BICUBIC)
    green_RS = green_RS.resize((hr_width,hr_height),resample=Image.BICUBIC)
    hr_DEM = hr_DEM.resize((hr_width,hr_height),resample=Image.BICUBIC)
    lr_red_RS = hr_to_lr(red_RS,args.scale)
    lr_blue_RS = hr_to_lr(blue_RS,args.scale)
    lr_green_RS = hr_to_lr(green_RS,args.scale)
    lr_DEM = hr_to_lr(hr_DEM,args.scale)
    
    for i in range(0,red_RS.shape[0] - args.patch_size + 1,args.stride):
        if num >= 5000:
            break
        for j in range(0,red_RS.shape[1] - args.patch_size + 1,args.stride):
            lr_patchs_1.append(lr_red_RS[i:i + args.patch_size,j:j + args.patch_size])
            lr_patchs_2.append(lr_blue_RS[i:i + args.patch_size,j:j + args.patch_size])
            lr_patchs_3.append(lr_green_RS[i:i + args.patch_size,j:j + args.patch_size])
            lr_patchs_4.append(lr_DEM[i:i + args.patch_size,j:j + args.patch_size])
            hr_patchs_1.append(red_RS[i:i + args.patch_size,j:j + args.patch_size])
            hr_patchs_2.append(blue_RS[i:i + args.patch_size,j:j + args.patch_size])
            hr_patchs_3.append(green_RS[i:i + args.patch_size,j:j + args.patch_size])
            hr_patchs_4.append(hr_DEM[i:i + args.patch_size,j:j + args.patch_size])
            num = num + 1
            if num >= 5000:
                break
    
    lr_patchs_1 = np.array(lr_patchs_1)
    hr_patchs_1 = np.array(hr_patchs_1)

    h5_file.create_dataset('lr1',data = lr_patchs_1)
    h5_file.create_dataset('lr2',data = lr_patchs_2)
    h5_file.create_dataset('lr3',data = lr_patchs_3)
    h5_file.create_dataset('lr4',data = lr_patchs_4)
    h5_file.create_dataset('hr1',data = hr_patchs_1)
    h5_file.create_dataset('hr2',data = hr_patchs_2)
    h5_file.create_dataset('hr3',data = hr_patchs_3)
    h5_file.create_dataset('hr4',data = hr_patchs_4)
    h5_file.close()

def eval(args):
    h5_file = h5py.File(args.output_path,'w')

    RS = args.RS
    DEM = args.DEM

    lr_group_1 = h5_file.create_group('lr1')
    lr_group_2 = h5_file.create_group('lr2')
    lr_group_3 = h5_file.create_group('lr3')
    lr_group_4 = h5_file.create_group('lr4')

    hr_group_1 = h5_file.create_group('hr1')
    hr_group_2 = h5_file.create_group('hr2')
    hr_group_3 = h5_file.create_group('hr3')
    hr_group_4 = h5_file.create_group('hr4')

    hr_RS = gdal.open(RS)
    hr_DEM = Image.open(DEM)

    red_RS = ToImage(hr_RS,1)
    blue_RS = ToImage(hr_RS,2)
    green_RS = ToImage(hr_RS,3)

    hr_width = (red_RS.width // args.scale) * args.scale
    hr_height = (red_RS.height // args.scale) * args.scale

    red_RS = red_RS.resize((hr_width,hr_height),resample=Image.BICUBIC)
    blue_RS = blue_RS.resize((hr_width,hr_height),resample=Image.BICUBIC)
    green_RS = green_RS.resize((hr_width,hr_height),resample=Image.BICUBIC)
    hr_DEM = hr_DEM.resize((hr_width,hr_height),resample=Image.BICUBIC)
    lr_red_RS = hr_to_lr(red_RS,args.scale)
    lr_blue_RS = hr_to_lr(blue_RS,args.scale)
    lr_green_RS = hr_to_lr(green_RS,args.scale)
    lr_DEM = hr_to_lr(hr_DEM,args.scale)

    hr_group_1.create_dataset(str(1),data=red_RS[:300,:300])
    hr_group_2.create_dataset(str(1),data=blue_RS[:300,:300])
    hr_group_3.create_dataset(str(1),data=green_RS[:300,:300])
    hr_group_4.create_dataset(str(1),data=hr_DEM[:300,:300])

    lr_group_1.create_dataset(str(1),data=lr_red_RS[:300,:300])
    lr_group_2.create_dataset(str(1),data=lr_blue_RS[:300,:300])
    lr_group_3.create_dataset(str(1),data=lr_green_RS[:300,:300])
    lr_group_4.create_dataset(str(1),data=lr_DEM[:300,:300])
    
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--RS',type=str,required=True)
    parser.add_argument('--DEM',type=str,required=True)
    parser.add_argument('--output-path',type=str,required=True)
    parser.add_argument('--patch-size',type=int,default=36)
    parser.add_argument('--stride',type=int,default=14)
    parser.add_argument('--scale',type=int,default=2)
    parser.add_argument('--eval',action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)