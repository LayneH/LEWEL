# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import builtins
from logging import root
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import backbone as backbone_models
from models import get_lewel_model
from utils import utils, lr_schedule, LARS, get_norm, dist_init
import data.transforms as data_transforms
from engine import ss_validate
from data.base_dataset import get_dataset

backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default="in1k",
                    help='name of dataset', choices=['in1k', 'in100', 'im_folder', 'in1k_idx'])
parser.add_argument('--data-root', default="",
                    help='root of dataset folder')
parser.add_argument('--arch', metavar='ARCH', default='LEWEL',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 64)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--save-dir', default="ckpts",
                    help='checkpoint directory')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=10, type=int,
                    metavar='N', help='checkpoint save frequency (default: 10)')
parser.add_argument('--eval-freq', default=5, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=23456, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--port', default=5389, type=int,
                    help='communication port for distributed training. ')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# ssl specific configs:
parser.add_argument('--proj-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--enc-m', default=0.996, type=float,
                    help='momentum of updating key encoder (default: 0.996)')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for network (default: None)')
parser.add_argument('--num-neck-mlp', default=2, type=int,
                    help='number of neck mlp (default: 2)')
parser.add_argument('--hid-dim', default=4096, type=int,
                    help='hidden dimension of mlp (default: 4096)')
parser.add_argument('--amp', action='store_true',
                    help='use automatic mixed precision training')

# options for LEWEL
parser.add_argument('--lewel-l2-norm', action='store_true',
                    help='use l2-norm before applying softmax on attention map')
parser.add_argument('--lewel-scale', default=1., type=float,
                    help='Scale factor of attention map (default: 1.)')
parser.add_argument('--lewel-num-heads', default=8, type=int,
                    help='Number of heads in lewel (default: 8)')
parser.add_argument('--lewel-loss-weight', default=0.5, type=float,
                    help='loss weight for aligned branch (default: 0.5)')

# options for KNN search
parser.add_argument('--num-nn', default=20, type=int,
                    help='Number of nearest neighbors (default: 20)')
parser.add_argument('--nn-mem-percent', type=float, default=0.1,
                    help='number of percentage mem datan for KNN evaluation')
parser.add_argument('--nn-query-percent', type=float, default=0.5,
                    help='number of percentage query datan for KNN evaluation')


best_acc1 = 0


def main(args):
    global best_acc1
    args.gpu = args.local_rank

    # create model
    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_lewel_model(args.arch)
    norm_layer = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        dim=args.proj_dim,
        m=args.enc_m,
        hid_dim=args.hid_dim,
        norm_layer=norm_layer,
        num_neck_mlp=args.num_neck_mlp,
        scale=args.lewel_scale,
        l2_norm=args.lewel_l2_norm,
        num_heads=args.lewel_num_heads,
        loss_weight=args.lewel_loss_weight,
    )
    print(model)
    print(args)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            state_dict = torch.load(args.pretrained, map_location="cpu")['state_dict']
            # rename state_dict keys
            for k in list(state_dict.keys()):
                new_key = k.replace("module.", "")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pretrained model from '{}'".format(args.pretrained))
            if len(msg.missing_keys) > 0:
                print("missing keys: {}".format(msg.missing_keys))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys: {}".format(msg.unexpected_keys))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))


    model.cuda()
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # define optimizer
    params = collect_params(model, exclude_bias_and_bn=True, sync_bn='EMAN' in args.arch)
    optimizer = LARS(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            if 'best_acc1' in checkpoint:
                best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            else:
                print("no scaler checkpoint")
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    transform1, transform2 = data_transforms.get_byol_tranforms()
    train_dataset = get_dataset(
        args.dataset,
        mode='train',
        transform=data_transforms.TwoCropsTransform(transform1, transform2),
        data_root=args.data_root)
    print("train_dataset:\n{}".format(train_dataset))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        persistent_workers=True)

    val_loader_base = torch.utils.data.DataLoader(
        get_dataset(
            args.dataset,
            mode='val',
            transform=data_transforms.get_transforms("DefaultVal"),
            data_root=args.data_root,
            percent=args.nn_mem_percent
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers//2, pin_memory=True,
        persistent_workers=True)

    val_loader_query = torch.utils.data.DataLoader(
        get_dataset(
            args.dataset,
            mode='val',
            transform=data_transforms.get_transforms("DefaultVal"),
            data_root=args.data_root,
            percent=args.nn_query_percent,
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers//2, pin_memory=True,
        persistent_workers=True)

    if args.evaluate:
        ss_validate(val_loader_base, val_loader_query, model, args)
        return

    best_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if epoch >= args.warmup_epoch:
            lr_schedule.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, epoch, args)

        is_best = False
        if (epoch + 1) % args.eval_freq == 0:
            acc1 = ss_validate(val_loader_base, val_loader_query, model, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.local_rank % args.world_size == 0):
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict(),
            }, is_best=is_best, epoch=epoch, args=args)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))


def train(train_loader, model, optimizer, scaler, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    curr_lr = utils.InstantMeter('LR', '')
    curr_mom = utils.InstantMeter('MOM', '')
    progress = utils.ProgressMeter(
        len(train_loader),
        [curr_lr, curr_mom, batch_time, data_time, losses],
        prefix="Epoch: [{}/{}]\t".format(epoch, args.epochs))

    # iter info
    batch_iter = len(train_loader)
    max_iter = float(batch_iter * args.epochs)

    # switch to train mode
    model.train()
    if "EMAN" in args.arch:
        print("setting the key model to eval mode when using EMAN")
        if hasattr(model, 'module'):
            model.module.target_net.eval()
        else:
            model.target_net.eval()

    end = time.time()
    for i, (images, _, idx) in enumerate(train_loader):
        # update model momentum
        curr_iter = float(epoch * batch_iter + i)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            idx = idx.cuda(args.gpu, non_blocking=True)

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * batch_iter
            curr_step = epoch * batch_iter + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        if scaler is None:
            # compute loss
            loss = model(im_v1=images[0], im_v2=images[1], idx=idx)

            # measure accuracy and record loss
            losses.update(loss.item(), images[0].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:   # AMP
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(im_v1=images[0], im_v2=images[1], idx=idx)
            # measure accuracy and record loss
            losses.update(loss.item(), images[0].size(0))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if hasattr(model, 'module'):
            model.module.momentum_update(curr_iter, max_iter)
            curr_mom.update(model.module.curr_m)
        else:
            model.momentum_update(curr_iter, max_iter)
            curr_mom.update(model.curr_m)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def collect_params(model, exclude_bias_and_bn=True, sync_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    weight_param_list, bn_and_bias_param_list = [], []
    weight_param_names, bn_and_bias_param_names = [], []
    for name, param in model.named_parameters():
        if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name or (sync_bn and 'mlp.1' in name)):
            bn_and_bias_param_list.append(param)
            bn_and_bias_param_names.append(name)
        else:
            weight_param_list.append(param)
            weight_param_names.append(name)
    print("weight params:\n{}".format('\n'.join(weight_param_names)))
    print("bn and bias params:\n{}".format('\n'.join(bn_and_bias_param_names)))
    param_list = [{'params': bn_and_bias_param_list, 'weight_decay': 0., 'lars_exclude': True},
                  {'params': weight_param_list}]
    return param_list


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.distributed = True
    opt.multiprocessing_distributed = True

    _, opt.local_rank, opt.world_size = dist_init(opt.port)
    cudnn.benchmark = True

    # suppress printing if not master
    if dist.get_rank() != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    main(opt)
