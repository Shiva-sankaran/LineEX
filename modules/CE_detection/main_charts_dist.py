import argparse
from pickle import TRUE
import random
import time
import os
from pathlib import Path
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--save_intvl', default=5, type=int)
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='charts')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--coco_eval', default=False, type=bool)

    # distributed training parameters
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--num_gpus', default=3, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    rank = args.local_rank
    # rank = int(os.environ['LOCAL_RANK'])
    if rank == 0:
        versions = os.listdir(args.output_dir + '/logs')
        versions = [int(v) for v in versions]
        versions.sort()
        version = versions[-1]
        if args.resume and not args.resume.startswith('https'):
            writer = SummaryWriter(args.output_dir + '/logs/' + str(version))
        else:
            os.mkdir(args.output_dir + '/logs/' + str(int(version) + 1))
            writer = SummaryWriter(args.output_dir + '/logs/' + str(int(version) + 1))

    # args.distributed = False
    if args.distributed:
        device = torch.device('cuda:%d' % rank)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend = 'nccl', init_method='env://', 
                                world_size = args.num_gpus, rank = rank)
        # torch.cuda.set_per_process_memory_fraction(0.7, 2)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # args.device = "cuda:3"
    # device = "cuda:3"
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, 
        mode='min', patience=3, threshold=1e-4, threshold_mode='abs', verbose=True)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
            if 'scheduler' in checkpoint:
                lr_scheduler_plat.load_state_dict(checkpoint['lr_scheduler_plat'])
            elif 'lr_scheduler_plat' in checkpoint and 'lr_scheduler_step' in checkpoint:
                lr_scheduler_step.load_state_dict(checkpoint['lr_scheduler_step'])
                lr_scheduler_plat.load_state_dict(checkpoint['lr_scheduler_plat'])
            print()
            print("Learning rate set to: {}".format(optimizer.param_groups[0]['lr']))
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded model at Epoch {}".format(int(args.start_epoch)))

    if args.eval:
        val_loss, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.coco_eval)
        val_loss /= len(data_loader_val)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    if args.resume:
        iter = args.start_epoch*len(data_loader_train) + 1
        print(iter)
    else:
        iter = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        epoch_loss, iter = train_one_epoch(
            model, criterion, data_loader_train, optimizer, iter, device, epoch,
            args.clip_max_norm, writer if rank == 0 else None, args)
        epoch_loss /= len(data_loader_train)
        if os.environ['LOCAL_RANK']:
            print("Iter = {} (Epoch = {}, {}), Loss = {}".format(iter, epoch, device, epoch_loss))

        lr_scheduler_step.step()
        lr_scheduler_plat.step(epoch_loss)
        if args.output_dir:
            # checkpoint_paths = [output_dir / 'ckpt/checkpoint_latest.pth']
            checkpoint_paths = [args.output_dir+'/ckpt/'+args.save_dir+'/checkpoint_latest.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_intvl == 0:
                # checkpoint_paths.append(output_dir / f'ckpt/checkpoint{epoch:04}.pth')
                checkpoint_paths.append(args.output_dir+'/ckpt/'+args.save_dir+'/checkpoint'+str(epoch+1)+'.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler_plat': lr_scheduler_plat.state_dict(),
                    'lr_scheduler_step': lr_scheduler_step.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        val_loss, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, 
            args.output_dir, args.coco_eval)
        if rank == 0:
            val_loss /= len(data_loader_val)
            writer.add_scalar('val_loss', val_loss, iter)
            print("Validating at Iter = {}, {}), Val Loss = {}".format(iter, device, val_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# COMMAND TO RUN:
# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 main_charts_dist.py 
# --coco_path /home/md.hassan/charts/data/data/synth_lines/data/st_lines --batch_size 14 
# --dataset_file charts --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
