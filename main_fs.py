"""
COSeg Training script.

"""

import os
import time
import random
import numpy as np
import argparse
import shutil
import copy
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from functools import partial
from PointSAM_pred import PointSAM as pointsam
from util import config
from util.s3dis_fs import S3DIS_FS, S3DIS_FS_TEST, S3DIS_FSForVIS
from util.scannet_v2_fs import Scannetv2_FS, Scannetv2_FS_TEST
from util.common_util import (
    AverageMeter,
    find_free_port,
)
from util.data_util import (
    collate_fn_limit_fs,
    collate_fn_limit_fs_train,
)
from util import transform
from util.logger import get_logger

from util.lr import MultiStepWithWarmup, PolyLR
from util.common_util import load_pretrain_checkpoint, evaluate_metric
from model.coseg_2way5shot import COSeg
import wandb
from point_sam.build_model import build_point_sam
import torch.nn.functional as F


def get_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch Point Cloud Semantic Segmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/s3dis_COSeg_fs.yaml",
        help="config file",
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="config/scannetv2_COSeg_fs.yaml",
    #     help="config file",
    # )
    parser.add_argument(
        "opts",
        help="see config/s3dis_COSeg_fs.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../pretrained_model/model.safetensors",
        help="point sam pretrained weight path",
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed
        and args.rank % args.ngpus_per_node == 0
    )


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(x) for x in args.train_gpu
    )
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size

        mp.spawn(
            main_worker,
            nprocs=args.ngpus_per_node,
            args=(args.ngpus_per_node, args),
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        torch.cuda.set_device(gpu)
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        if args.vis:
            wandb.init(
                project="COSeg",
                name=os.path.basename(args.save_path),
                config=args,
            )

    # get model
    model = COSeg(args)

    sam = pointsam(args).eval()

    # set optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        defaults = {}
        defaults["lr"] = args.base_lr

        params = []
        memo = set()
        for param in model.encoder.parameters():
            param.requires_grad = False
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(
                recurse=False
            ):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)

                params.append({"params": [value], **hyperparams})

        optimizer = torch.optim.AdamW(
            params, lr=args.base_lr, weight_decay=args.weight_decay
        )

    if main_process():
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info(model)
        logger.info(
            "#Model parameters: {}".format(
                sum([x.nelement() for x in model.parameters()])
            )
        )
        # logger.info("=>creaing point sam ...")
        # logger.info(sam)
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        args.workers = int(
            (args.workers + ngpus_per_node - 1) / ngpus_per_node
        )
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = model.cuda()

    if args.pretrain_backbone:
        # load pretrained backbone
        model = load_pretrain_checkpoint(model, args.pretrain_backbone, gpu)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            pretrained_dict = checkpoint["state_dict"]
            if not isinstance(
                model, torch.nn.parallel.DistributedDataParallel
            ):
                pretrained_dict = {
                    k.replace("module.", ""): v
                    for k, v in pretrained_dict.items()
                }

            model.load_state_dict(pretrained_dict)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler_state_dict = checkpoint["scheduler"]
            best_iou = checkpoint["best_iou"]
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume)
                )

    val_transform = None
    if args.data_name == "s3dis":
        if args.forvis:
            val_data = S3DIS_FSForVIS(
                split="test",
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                target_class=args.target_class,
            )
        else:
            val_data = S3DIS_FS_TEST(
                split=args.eval_split,
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                num_episode_per_comb=args.num_episode_per_comb,
            )
        valid_calsses = list(val_data.classes)

    elif args.data_name == "scannetv2":
        val_data = Scannetv2_FS_TEST(
            split=args.eval_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=val_transform,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
            num_episode_per_comb=args.num_episode_per_comb,
        )
        valid_calsses = list(val_data.classes)
    else:
        raise ValueError(
            "The dataset {} is not supported.".format(args.data_name)
        )

    if not args.forvis:
        # main process firstly call, since it will construct the dataset if not exist
        # and avoid conflicts from other processes
        if main_process():
            logger.info(
                "The main process prepares test data while other processes wait..."
            )
            val_data.prepare_test_data()

        if args.distributed:
            dist.barrier()
            val_data.prepare_test_data()

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=partial(
            collate_fn_limit_fs, include_scene_names=args.forvis
        ),
    )

    if args.test:
        #model = load_pretrain_checkpoint(model, args.pretrain_backbone, gpu)
        validate(val_loader, model, valid_calsses, sam)
        if main_process():
            writer.close()
            logger.info("==>Test done!")
        return

    if args.data_name == "s3dis":
        train_transform = None
        if args.aug:
            jitter_sigma = args.get("jitter_sigma", 0.01)
            jitter_clip = args.get("jitter_clip", 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info(
                    "jitter_sigma: {}, jitter_clip: {}".format(
                        jitter_sigma, jitter_clip
                    )
                )
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(
                        along_z=args.get("rotate_along_z", True)
                    ),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomJitter(
                        sigma=jitter_sigma, clip=jitter_clip
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )
        train_data = S3DIS_FS(
            split="train",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
        )
        train_calsses = list(train_data.classes)
    elif args.data_name == "scannetv2":
        train_transform = None
        if args.aug:
            if main_process():
                logger.info("use Augmentation")
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(
                        along_z=args.get("rotate_along_z", True)
                    ),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )

        train_data = Scannetv2_FS(
            split="train",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
        )
        train_calsses = list(train_data.classes)
    else:
        raise ValueError(
            "The dataset {} is not supported.".format(args.data_name)
        )

    if main_process():
        logger.info("Train Classes: {}".format(train_calsses))
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data
        )
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn_limit_fs_train,
    )

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == "step"
        if main_process():
            logger.info(
                "scheduler: MultiStepWithWarmup. scheduler_update: {}".format(
                    args.scheduler_update
                )
            )
        iter_per_epoch = len(train_loader)
        milestones = [
            int(args.epochs * 0.6) * iter_per_epoch,
            int(args.epochs * 0.8) * iter_per_epoch,
        ]
        scheduler = MultiStepWithWarmup(
            optimizer,
            milestones=milestones,
            gamma=0.1,
            warmup=args.warmup,
            warmup_iters=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
        )
    elif args.scheduler == "MultiStep":
        assert args.scheduler_update == "epoch"
        milestones = (
            [int(x) for x in args.milestones.split(",")]
            if hasattr(args, "milestones")
            else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        )
        gamma = args.gamma if hasattr(args, "gamma") else 0.1
        if main_process():
            logger.info(
                "scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(
                    args.scheduler_update, milestones, gamma
                )
            )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif args.scheduler == "Poly":
        if main_process():
            logger.info(
                "scheduler: Poly. scheduler_update: {}".format(
                    args.scheduler_update
                )
            )
        if args.scheduler_update == "epoch":
            scheduler = PolyLR(
                optimizer, max_iter=args.epochs, power=args.power
            )
        elif args.scheduler_update == "step":
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(
                optimizer,
                max_iter=args.epochs * iter_per_epoch,
                power=args.power,
            )
        else:
            raise ValueError(
                "No such scheduler update {}".format(args.scheduler_update)
            )
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader,
            model,
            optimizer,
            epoch,
            scaler,
            scheduler,
            train_calsses,
            sam,
        )
        if args.scheduler_update == "epoch":
            scheduler.step()
        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar("loss_train", loss_train, epoch_log)
            writer.add_scalar("mIoU_train", mIoU_train, epoch_log)
            writer.add_scalar("mAcc_train", mAcc_train, epoch_log)
            writer.add_scalar("allAcc_train", allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, valid_calsses, sam
            )
            if main_process():
                writer.add_scalar("loss_val", loss_val, epoch_log)
                writer.add_scalar("mIoU_val", mIoU_val, epoch_log)
                writer.add_scalar("mAcc_val", mAcc_val, epoch_log)
                writer.add_scalar("allAcc_val", allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + "/model/model_last.pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save({"state_dict": model.state_dict(),
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_iou': best_iou}, filename)
            if is_best:
                logger.info("Is best")
                shutil.copyfile(
                    filename, args.save_path + "/model/model_best.pth"
                )

    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def train(
    train_loader, model, optimizer, epoch, scaler, scheduler, train_calsses, sam
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    sam = sam.cuda().eval()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (
        support_x,
        support_base_y,
        support_y,
        support_offset,
        query_x,
        query_base_y,
        query_y,
        query_offset,
        sampled_classes,
        support_block_names,
        query_block_names,
    ) in enumerate(train_loader):
        data_time.update(time.time() - end)

        query_y = query_y.cuda(non_blocking=True)

        use_amp = args.use_amp

        support_proposals, query_proposals = None, None
        from util import forward_mask
        # with torch.no_grad():
        #     support_proposals, query_proposals = forward_mask.forward_mask_func(
        #         args, sam, query_x.cuda(), support_x.cuda(), support_offset, query_offset
        #     )
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            output2, loss, query_pred, query_pred_low, output, yz = model(
                support_offset,
                support_x,
                support_y,
                query_offset,
                query_x,
                query_y,
                epoch,
                support_base_y=support_base_y,
                query_base_y=query_base_y,
                sampled_classes=sampled_classes,
                support_proposals=support_proposals,
                query_proposals=query_proposals,
            )

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == "step":
            scheduler.step()

        output = output.max(1)[1].squeeze(0)  # output: 1, c, pts
        n = query_y.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = query_y.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = evaluate_metric(
            output, query_y, sampled_classes, train_calsses, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(
                union
            ), dist.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (
            sum(target_meter.val) + 1e-10
        )
        miou = np.mean(intersection_meter.val / (union_meter.val + 1e-10))
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(
            int(t_h), int(t_m), int(t_s)
        )

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
                lr = max(lr)
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Loss {loss_meter.val:.4f} "
                "Lr: {lr} "
                "Accuracy {accuracy:.4f} "
                "miou {miou:.4f}.".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    lr=lr,
                    accuracy=accuracy,
                    miou=miou,
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", loss_meter.val, current_iter)
            writer.add_scalar(
                "mIoU_train_batch",
                np.mean(intersection / (union + 1e-10)),
                current_iter,
            )
            writer.add_scalar(
                "mAcc_train_batch",
                np.mean(intersection / (target + 1e-10)),
                current_iter,
            )
            writer.add_scalar("allAcc_train_batch", accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            "Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                epoch + 1, args.epochs, mIoU, mAcc, allAcc
            )
        )
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, valid_calsses, sam):
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    target_meter1 = AverageMeter()

    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    target_meter2 = AverageMeter()

    intersection_meter3 = AverageMeter()
    union_meter3 = AverageMeter()
    target_meter3 = AverageMeter()

    lo = AverageMeter()
   

    count_all = 0
    if args.forvis:
        target_class = val_loader.dataset.target_class
        pred_path = os.path.join(args.vis_save_path, target_class)
        os.makedirs(pred_path, exist_ok=True)

    torch.cuda.empty_cache()
    model.eval()
    sam = sam.cuda().eval()
    end = time.time()
    for i, batch in enumerate(val_loader):
        if args.forvis:
            (
                support_x,
                support_y,
                support_offset,
                query_x,
                query_y,
                query_offset,
                sampled_classes,
                scene_names,
            ) = batch
        else:
            (
                support_x,
                support_y,
                support_offset,
                query_x,
                query_y,
                query_offset,
                sampled_classes,
                support_proposals,
                query_proposals,
            ) = batch

        data_time.update(time.time() - end)

        query_y = query_y.cuda(non_blocking=True)

        with torch.no_grad():
            sam.panduan(query_x, support_x, query_y, support_y, query_offset, support_offset)
                
        with torch.no_grad():
            low, loss, query_y_low, query_pred_low, output, count1 = model(
                support_offset,
                support_x,
                support_y,
                query_offset,
                query_x,
                query_y,
                5,
                sampled_classes=sampled_classes,
                support_proposals = support_proposals,
                query_proposals = query_proposals,
            )
        output_logits = F.softmax(output.squeeze(), dim = 0) # 3 x N
        #query_pred =query_pred.max(1)[1]
        output = output.max(1)[1].squeeze(0)  # output: 1, c, pts 每个pts标签
        #count1 = torch.tensor(0).cuda()
        n = query_y.size(0)
        if n >= 512:
            device = output.device
            query_x = query_x.cuda(non_blocking=True)
            mask = sam(query_x[:,:3].contiguous(), query_x[:,3:6].contiguous(), pred=output_logits, pred_gt = output , offest = query_offset, query_pred_low = query_pred_low)
            alpha = 0.5
            output2 = alpha * output_logits + (1 - alpha) * mask
            output2 = output2.max(0)[1]
        else:
            output2 = output
        if args.multiprocessing_distributed:
            loss *= n
            count = query_y.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection1, union1, target1 = evaluate_metric(
            output, query_y, sampled_classes, valid_calsses, args.ignore_label
        )

        intersection2, union2, target2 = evaluate_metric(
           low.int().cuda(), query_y_low.cuda(), sampled_classes, valid_calsses, args.ignore_label
        )

        intersection3, union3, target3 = evaluate_metric(
            output2, query_y, sampled_classes, valid_calsses, args.ignore_label
        )
       

        if args.forvis:
            query_name = scene_names[0]
            support_name = scene_names[1]
            save_dir = os.path.join(pred_path, f"{query_name}_{support_name}")
            os.makedirs(save_dir, exist_ok=True)
            np.save(
                os.path.join(save_dir, "query.npy"),
                query_x.cpu().numpy(), # 
            )
            np.save(
                os.path.join(save_dir, "querylb.npy"),
                query_y.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "sup.npy"),
                support_x.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "suplb.npy"),
                support_y.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "pred.npy"),
                output.cpu().numpy(),
            )
            torch.cuda.empty_cache()

        if args.multiprocessing_distributed:
            dist.all_reduce(intersection1), dist.all_reduce(
                union1
            ), dist.all_reduce(target1)
        intersection1, union1, target1 = (
            intersection1.cpu().numpy(),
            union1.cpu().numpy(),
            target1.cpu().numpy(),
        )


        if args.multiprocessing_distributed:
            dist.all_reduce(intersection2), dist.all_reduce(
                union2
            ), dist.all_reduce(target2)
        intersection2, union2, target2 = (
            intersection2.cpu().numpy(),
            union2.cpu().numpy(),
            target2.cpu().numpy(),
        )

        if args.multiprocessing_distributed:
            dist.all_reduce(intersection3), dist.all_reduce(
                union3
            ), dist.all_reduce(target3)
        intersection3, union3, target3 = (
            intersection3.cpu().numpy(),
            union3.cpu().numpy(),
            target3.cpu().numpy(),
        )

        #count1 = torch.tensor(1).cuda()

        if args.multiprocessing_distributed:
            dist.all_reduce(count1)
           
        count1 = count1.cpu().numpy()


        intersection_meter1.update(intersection1), union_meter1.update(
            union1
        ), target_meter1.update(target1)

        intersection_meter2.update(intersection2), union_meter2.update(
            union2
        ), target_meter2.update(target2)

        intersection_meter3.update(intersection3), union_meter3.update(
            union3
        ), target_meter3.update(target3)

        accuracy1 = sum(intersection_meter1.val) / (
            sum(target_meter1.val) + 1e-10
        )

        accuracy2 = sum(intersection_meter2.val) / (
            sum(target_meter2.val) + 1e-10
        )

        accuracy3 = sum(intersection_meter3.val) / (
            sum(target_meter3.val) + 1e-10
        )

        lo.update(count1)


        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                "Accuracy {accuracy:.4f}.".format(
                    i + 1,
                    len(val_loader),
                    data_time=data_time,
                    batch_time=batch_time,
                    loss_meter=loss_meter,
                    accuracy=accuracy1,
                )
            )
        torch.cuda.empty_cache()

    iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)
    accuracy_class1 = intersection_meter1.sum / (target_meter1.sum + 1e-10)
    mIoU1 = np.mean(iou_class1)
    mAcc1 = np.mean(accuracy_class1)
    allAcc1 = sum(intersection_meter1.sum) / (sum(target_meter1.sum) + 1e-10)

    iou_class2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
    accuracy_class2 = intersection_meter2.sum / (target_meter2.sum + 1e-10)
    mIoU2 = np.mean(iou_class2)
    mAcc2 = np.mean(accuracy_class2)
    allAcc2 = sum(intersection_meter2.sum) / (sum(target_meter2.sum) + 1e-10)

    iou_class3 = intersection_meter3.sum / (union_meter3.sum + 1e-10)
    accuracy_class3 = intersection_meter3.sum / (target_meter3.sum + 1e-10)
    mIoU3 = np.mean(iou_class3)
    mAcc3 = np.mean(accuracy_class3)
    allAcc3 = sum(intersection_meter3.sum) / (sum(target_meter3.sum) + 1e-10)
    if main_process():
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU1, mAcc1, allAcc1
            )
        )
        for i in range(len(valid_calsses)):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    valid_calsses[i], iou_class1[i], accuracy_class1[i]
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Before Refine Evaluation <<<<<<<<<<<<<<<<<")

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU2, mAcc2, allAcc2
            )
        )
        for i in range(len(valid_calsses)):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    valid_calsses[i], iou_class2[i], accuracy_class2[i]
                )
            )
            
        logger.info("<<<<<<<<<<<<<<<<< End Middle Evaluation <<<<<<<<<<<<<<<<<")

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU3, mAcc3, allAcc3
            )
        )
        for i in range(len(valid_calsses)):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    valid_calsses[i], iou_class3[i], accuracy_class3[i]
                )
            )
        
        logger.info("<<<<<<<<<<<<<<<<< End After Refine Evaluation <<<<<<<<<<<<<<<<<")

    
    print(lo.sum)


    return loss_meter.avg, mIoU1, mAcc1, allAcc1


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
