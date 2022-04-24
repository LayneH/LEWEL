# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import builtins
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
parser.add_argument('--dataset', default='in1k',
                    help='name of dataset')
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
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--save-dir', default="ckpts",
                    help='checkpoint directory')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

parser.add_argument('--pretrained', default='', type=str,
                    help='path to self-supervised pretrained checkpoint')
# online_net.backbone for BYOL
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')

best_acc1 = 0


def main(args):
    global best_acc1
    args.gpu = args.local_rank

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    norm = get_norm(args.norm)
    model = models.__dict__[args.arch](num_classes=args.cls, norm_layer=norm)
    print(model)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained) and 'checkpoint' in args.pretrained:
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            model_prefix = 'module.' + args.model_prefix
            # print(state_dict.keys())
            for k in list(state_dict.keys()):
                # retain only student model up to before the embedding layer
                if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                    # remove prefix
                    new_key = k.replace(model_prefix + '.', "")
                    state_dict[new_key] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            print(state_dict.keys())
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            if len(msg.missing_keys) > 0:
                print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        
        elif os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            state_dict = torch.load(args.pretrained, map_location="cpu")
            print(state_dict.keys())
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            if len(msg.missing_keys) > 0:
                print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model.cuda()
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
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
    train_transforms = data_transforms.get_transforms("DefaultTrain")
    train_dataset = get_dataset(
        args.dataset, mode='train', transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        persistent_workers=True)

    val_transforms = data_transforms.get_transforms("DefaultVal")
    val_loader = torch.utils.data.DataLoader(
        get_dataset(
            args.dataset, mode='val', transform=val_transforms),
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
                and args.local_rank % args.world_size == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % args.world_size == 0):
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained, args)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t".format(epoch, args.epochs, curr_lr))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

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


def sanity_check(state_dict, pretrained_weights, args):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    model_prefix = 'module.' + args.model_prefix + '.'
    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = model_prefix + k[len('module.'):] \
            if k.startswith('module.') else model_prefix + k
        # for BYOL model
        k_pre = k_pre.replace('backbone.fc.', 'neck.mlp.')

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
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
