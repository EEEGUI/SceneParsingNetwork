import os
import yaml
import time
import shutil
import torch
import random

from tqdm import tqdm
from models import icnet, spnet
from torch import optim
from loss import Loss
from utils import get_logger
from schedulers import PolynomialLR
from torch.utils import data
from torchvision.transforms import Compose
from augmentations import *
from cityscapes_loader import cityscapesLoader
from tensorboardX import SummaryWriter
from metrics import SegmentationScore, DepthEstimateScore, averageMeter

networks = {'icnet': icnet.icnet, 'spnet': spnet.spnet}


def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device(cfg['device'])

    # Setup Metrics
    seg_scores = SegmentationScore()
    depth_scores = DepthEstimateScore()

    augmentations = Compose([
                             RandomRotate(cfg['training']['argumentation']['random_rotate']),
                             RandomCrop(cfg['training']['img_size']),
                             RandomHorizonFlip(cfg['training']['argumentation']['random_hflip']),
                            ])

    traindata = cityscapesLoader(cfg['data']['path'],
                                 img_size=cfg['training']['img_size'],
                                 split=cfg['data']['train_split'],
                                 is_transform=True,
                                 augmentations=augmentations)

    valdata = cityscapesLoader(cfg['data']['path'],
                               img_size=cfg['training']['img_size'],
                               split=cfg['data']['val_split'],
                               is_transform=True)

    trainloader = data.DataLoader(traindata, batch_size=cfg['training']['batch_size'])
    valloader = data.DataLoader(valdata, batch_size=cfg['training']['batch_size'])

    # Setup Model
    model = networks[cfg['arch']](**cfg['model'])

    model.to(device)
    loss_fn = Loss(**cfg['training']['loss']).to(device)
    # Setup optimizer, lr_scheduler and loss function
    optimizer = optim.SGD(model.parameters(), **cfg['training']['optimizer'])
    # TODO
    # optimizer_loss = optim.SGD(loss_fn.parameters(), **cfg['training']['optimizer_loss'])

    scheduler = PolynomialLR(optimizer, max_iter=cfg['training']['train_iters'], **cfg['training']['schedule'])
    # TODO
    # scheduler_loss = ConstantLR(optimizer_loss)

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_miou = -100.0
    best_abs_rel = float('inf')
    i = start_iter
    flag = True
    optimizer.zero_grad()
    while i <= cfg["training"]["train_iters"] and flag:
        for sample in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            # TODO
            # scheduler_loss.step()
            model.train()
            images = sample['image'].to(device)
            labels = sample['label'].to(device)
            depths = sample['depth'].to(device)

            # TODO
            # optimizer_loss.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, depths, labels) / cfg['training']['accu_steps']
            loss.backward()
            if i % cfg['training']['accu_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            # TODO
            # optimizer_loss.step()

            time_meter.update(time.time() - start_ts)

            if (i) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i,
                    cfg["training"]["train_iters"],
                    loss.item()*cfg['training']['accu_steps'],
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item()*cfg['training']['accu_steps'], i)
                writer.add_scalar('param/delta1', loss_fn.delta1, i)
                writer.add_scalar('param/delta2', loss_fn.delta2, i)
                writer.add_scalar('param/learning-rate', scheduler.get_lr()[0], i)
                time_meter.reset()

            if i % cfg["training"]["val_interval"] == 0 or i == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    for i_val, sample in tqdm(enumerate(valloader)):
                        images_val = sample['image'].to(device)
                        labels_val = sample['label'].to(device)
                        depths_val = sample['depth'].to(device)
                        outputs = model(images_val)
                        val_loss = loss_fn(outputs, depths_val, labels_val)

                        depth_scores.update(depths_val.cpu().numpy(), outputs[-1][0].data.cpu().numpy())
                        seg_scores.update(labels_val.cpu().numpy(), outputs[-1][1].data.max(1)[1].cpu().numpy())

                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                seg_score, class_iou = seg_scores.get_scores()
                for k, v in seg_score.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("seg_val_metrics/{}".format(k), v, i)

                for k, v in class_iou.items():
                    # logger.info("{}: {}".format(k, v))
                    writer.add_scalar("seg_val_metrics/cls_{}".format(k), v, i)

                depth_score = depth_scores.get_scores()
                for k, v in depth_score.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("depth_val_metrics/{}".format(k), v, i)

                val_loss_meter.reset()
                seg_scores.reset()
                depth_scores.reset()

                # if seg_score["Mean IoU : \t"] >= best_miou and depth_score['abs_rel'] <= best_abs_rel:
                if seg_score["Mean IoU : \t"] >= best_miou:
                    best_iou = seg_score["Mean IoU : \t"]
                    best_abs_rel = depth_score['abs_rel']
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                        "best_abs_rel": best_abs_rel
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pth".format(cfg['arch'], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

                state = {
                    "epoch": i + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_{}_model.pth".format(i, cfg['arch'], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)

            if i == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    config_file = "config/spnet-cityscapes.yml"
    with open(config_file) as fp:
        cfg = yaml.safe_load(fp)

    # run_id = random.randint(1, 100000)
    run_id = 607
    logdir = os.path.join("runs", os.path.basename(config_file)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(config_file, logdir)

    logger = get_logger(logdir, config_file.split('/')[-1].split('-')[0])
    logger.info("Let the games begin")

    train(cfg, writer, logger)
