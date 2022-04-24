# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import backbone as customized_models
import data.transforms as data_transforms
from utils import utils, get_norm, dist_init
from engine import validate
from data.base_dataset import get_dataset

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default="in1k",
                    help='name of dataset', choices=['in1k', 'in100', 'im_folder', 'in1k_idx'])
parser.add_argument('--data-root', default="",
                    help='root of dataset folder')
parser.add_argument('--trainindex', default=None, type=str, metavar='PATH',
                    help='path to train annotation (default: None)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', default=[30, 60], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-classifier', default=-1, type=float,
                    metavar='LR', help='initial learning rate for classifier')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--port', default=5389, type=int,
                    help='communication port for distributed training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# online_net.backbone for BYOL
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
parser.add_argument('--save-dir', default="ckpts",
                    help='checkpoint directory')

best_acc1 = 0


def main(args):
    global best_acc1
    args.gpu = args.local_rank

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    norm = get_norm(args.norm)
    model = models.__dict__[args.arch](num_classes=args.cls, norm_layer=norm)
    print(model)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                new_key = k.replace("module.", "")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
            model_num_cls = state_dict['fc.weight'].shape[0]
            if model_num_cls != args.cls:
                # if num_cls don't match, remove the last layer
                del state_dict['fc.weight']
                del state_dict['fc.bias']
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, \
                    "missing keys:\n{}".format('\n'.join(msg.missing_keys))
            else:
                model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))
    elif args.self_pretrained:
        if os.path.isfile(args.self_pretrained):
            print("=> loading checkpoint '{}'".format(args.self_pretrained))
            checkpoint = torch.load(args.self_pretrained, map_location="cpu")

            if 'state_dict' in checkpoint:
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                model_prefix = 'module.' + args.model_prefix
                for k in list(state_dict.keys()):
                    # retain only student model up to before the embedding layer
                    if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                        # remove prefix
                        new_key = k.replace(model_prefix + '.', "")
                        state_dict[new_key] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
            else:
                state_dict = checkpoint  

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            if 'epoch' in checkpoint:
                print("=> loaded pre-trained model '{}' (epoch {})".format(args.self_pretrained, checkpoint['epoch']))
            else:
                print("=> loaded pre-trained model '{}' ".format(args.self_pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.self_pretrained))

    model.cuda()
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # define optimizers for different parameter groups
    if args.lr != args.lr_classifier:
        print("using separate parameter groups with different learning rates!")
        feature_params, classifier_params = [], []
        feature_names, classifier_names = [], []
        for name, param in model.named_parameters():
            if 'fc.' in name:
                classifier_params += [param]
                classifier_names += [name]
            else:
                feature_params += [param]
                feature_names += [name]
        print("classifier params: {} using lr {}".format(classifier_names, args.lr_classifier))
        print("the rest params using lr {}".format(args.lr))
        optimizer = torch.optim.SGD([
            {'params': feature_params},
            {'params': classifier_params, 'lr': args.lr_classifier}
        ], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
        # Data loading code
    train_transforms = data_transforms.get_transforms("DefaultTrain")
    train_dataset = get_dataset('in1k_idx', mode='train', data_root=args.data_root,
                        idx=args.trainindex, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        persistent_workers=True)

    val_transforms = data_transforms.get_transforms("DefaultVal")
    val_loader = torch.utils.data.DataLoader(
        get_dataset(args.dataset, mode='val', 
        data_root=args.data_root, transform=val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch + 1) % args.eval_freq == 0:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % args.world_size == 0):
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    if len(optimizer.param_groups) > 1:
        curr_lr = [param_group['lr'] for param_group in optimizer.param_groups]
    else:
        curr_lr = optimizer.param_groups[0]['lr']
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t".format(epoch, args.epochs, curr_lr))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    num_groups = len(optimizer.param_groups)
    assert 1 <= num_groups <= 2
    lrs = []
    if num_groups == 1:
        lrs += [args.lr]
    elif num_groups == 2:
        lrs += [args.lr, args.lr_classifier]
    assert len(lrs) == num_groups
    for group_id, param_group in enumerate(optimizer.param_groups):
        lr = lrs[group_id]
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        param_group['lr'] = lr


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