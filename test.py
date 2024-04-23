import time
from utils import str2bool, count_params
import os
import argparse
from glob import glob
import warnings
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from models.new_unet_ca_spp import Unet_CA
import imageio

from metrics import dice_coef
from hausdorff import hausdorff_distance

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
def parse_args():
    parser = argparse.ArgumentParser()

    # Basic Information
    parser.add_argument('--user', default='yk', type=str)

    parser.add_argument('--experiment', default='new_unet_ca_spp', type=str)

    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

    # DataSet Information
    parser.add_argument('--root', default='path to training set', type=str)

    parser.add_argument('--testImage_paths_root', default='/21TB/yk/UNet2D/without_two/testImage/*', type=str)

    parser.add_argument('--testMask_paths_root', default='/21TB/yk/UNet2D/without_two/testMask/*', type=str)

    parser.add_argument('--output_path', default='/21TB/yk/brats_trans_attention/output_model/Brats2018_unet_ca_spp_1GPU_woDS/', type=str)

    parser.add_argument('--output_gt_path', default='/21TB/yk/brats_trans_attention/output_model/GT/',
                        type=str)

    parser.add_argument('--model_paths_root',
                        default='/21TB/yk/brats_trans_attention/models_pth/Brats2018_unet_ca_spp_1GPU_woDS/model_epoch_334.pth',
                        type=str)


    parser.add_argument('--mode', default='test', type=str)

    parser.add_argument('--dataset', default='Dataset', type=str)

    parser.add_argument('--model_name', default='new_unet_ca_spp', type=str)

    # Training Information

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--name', default=None,
                        help='model name')

    parser.add_argument('--deepsupervision', default=False, type=str2bool)

    parser.add_argument('-m', default='Calculate',
                        help='GetPicture or Calculate')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()


    model_path = val_args.model_paths_root

    if not os.path.exists(val_args.output_path):
        os.makedirs(val_args.output_path)

    print('Config -----')
    for arg in vars(val_args):
        print('%s: %s' %(arg, getattr(val_args, arg)))
    print('------------')


    # create model
    print("=> creating model %s" %val_args.model_name)
    model = Unet_CA()

    model = model.cuda()

    val_img_paths = glob(val_args.testImage_paths_root)
    val_mask_paths = glob(val_args.testMask_paths_root)

    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    val_dataset = Dataset(val_args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if val_args.m == "GetPicture":

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.cuda()

                    if val_args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[val_args.batch_size*i:val_args.batch_size*(i+1)]

                    for i in range(output.shape[0]):

                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i,0,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,1,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,2,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                        imsave(val_args.output_path + rgbName, rgbPic)

            torch.cuda.empty_cache()

        print("Done!")

    if val_args.m == "Calculate":

        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []

        maskPath = glob(val_args.output_gt_path + "*.png")
        pbPath = glob(val_args.output_path + "*.png")
        if len(maskPath) == 0:
            print("请先生成图片!")
            return

        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
            dice = dice_coef(wtpbregion, wtmaskregion)
            wt_dices.append(dice)

            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)

            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)

            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)

            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)

            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)


        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")


if __name__ == '__main__':
    main()
