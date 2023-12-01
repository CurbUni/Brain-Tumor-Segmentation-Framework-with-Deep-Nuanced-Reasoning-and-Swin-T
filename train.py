# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 20003 train.py

from glob import glob
import argparse
import os
import random
import logging
import numpy as np
import time
from dataset import Dataset
import torch
import torch.optim
from models import model
import torch.distributed as dist
from losses import BCEDiceLoss
from metrics import iou_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import pandas as pd

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='yk', type=str)

parser.add_argument('--experiment', default='new_swin_transformer_one', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='BraTS,'
                            'training on train.txt!',
                    type=str)

parser.add_argument('--root', default='path to training set', type=str)

parser.add_argument('--img_paths_root', default=r'/45TB/yk/data/data_prepare_2021/trainImage/*', type=str)

parser.add_argument('--mask_paths_root', default=r'/45TB/yk/data/data_prepare_2021/trainMask/*', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--dataset', default='Dataset', type=str)

parser.add_argument('--model_name', default='new_swin_transformer_one', type=str)

parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='sigmoid_dice', type=str)

parser.add_argument('--num_class', default=1, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='4,5,6,7', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=96, type=int, help='1 GPU == 24')

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=200, type=int)

parser.add_argument('--save_freq', default=5, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

parser.add_argument('--early-stop', default=50, type=int, metavar='N', help='early stopping (default: 20)')

args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor


def train(train_loader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    ious = AverageMeter()

    num_gpu = (len(args.gpu)+1) // 2

    model.train()

    for i, data in enumerate(train_loader):

        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

        input, target = data

        input = input.cuda()
        target = target.cuda()

        output = model(input)

        loss = criterion(output, target)
        reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
        iou = iou_score(output, target)

        losses.update(reduce_loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.local_rank == 0:
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f} iou: {:.5f}'.format(epoch, i, reduce_loss, iou))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    num_gpu = (len(args.gpu) + 1) // 2

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input, target = data

            input = input.cuda()
            target = target.cuda()

            output = model(input)

            loss = criterion(output, target)
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            iou = iou_score(output, target)

            losses.update(reduce_loss.item(), input.size(0))
            ious.update(iou, input.size(0))

            if args.local_rank == 0:
                logging.info('loss: {:.5f}  iou: {:.5f} '.format(reduce_loss, iou))

        log = OrderedDict([
            ('loss', losses.avg),
            ('iou', ious.avg),
        ])

        return log


def main():

    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)

        log_file = log_dir + '.txt'
        open(log_file, 'w')
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    model = model()

    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = BCEDiceLoss().cuda()
    # criterion = getattr(criterions, args.criterion)

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint_2021',
                                  args.experiment + args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if args.load:
        pretrained_dict = torch.load(
            '/45TB/yk/codes/brats_trans_attention/models_pth/upernet_swin_base_patch4_window7_512x512.pth')
        for k, v in pretrained_dict.items():
            model_dict = v

        model.load_state_dict(model_dict, strict=False)
        
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.date)))

        logging.info('Successfully loading checkpoint and training!')
    else:
        logging.info('re-training!!!')

    Resume = False
    if Resume:
        path_checkpoint = '/45TB/yk/codes/brats_trans_attention/models_pth/Brats2018_newformer_1GPU_2021_woDS/model_epoch_0_25.pth'
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    img_paths = glob(args.img_paths_root)
    mask_paths = glob(args.mask_paths_root)
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))

    train_dataset = Dataset(args, train_img_paths, train_mask_paths)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    num_gpu = (len(args.gpu) + 1) // 2

    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset, sampler=val_sampler, batch_size=args.batch_size // num_gpu,
                            drop_last=False, num_workers=args.num_workers, pin_memory=True)

    logging.info('Samples for train = {}'.format(len(train_loader)))
    logging.info('Samples for valid = {}'.format(len(val_loader)))

    start_time = time.time()
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])
    best_iou = 0
    trigger = 0

    for epoch in range(args.end_epoch):

        start_epoch = time.time()

        logging.info('----------------------------------------This is a Train----------------------------------')
        torch.set_grad_enabled(True)
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        logging.info('----------------------------------------This is a Valid----------------------------------')
        torch.set_grad_enabled(False)
        val_log = validate(val_loader, model, criterion)
        torch.cuda.empty_cache()

        end_epoch = time.time()

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log_csv_path = os.path.join(checkpoint_dir, 'log.csv')
        log.to_csv(log_csv_path, index=False)

        trigger += 1

        if args.local_rank == 0:
            if val_log['iou'] > best_iou:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)
                best_iou = val_log['iou']
                print("=> saved best model")
                trigger = 0


        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

        if args.local_rank == 0:
            # ---------判断是否早停-----------#
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    end_time = time.time()
                    total_time = (end_time - start_time) / 3600
                    logging.info('The total training time is {:.2f} hours'.format(total_time))
                    logging.info('---------------------------The training process finished!--------------------------')
                    print("=> early stopping")
                    break

            torch.cuda.empty_cache()

    if args.local_rank == 0:
        # writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)