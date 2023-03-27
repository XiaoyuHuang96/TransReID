from utils.logger import setup_logger
from datasets import make_dataloader, make_replay_dataloader
from model import make_model, make_small_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss, FCDistLoss
from processor import do_train, do_train_transformer, do_train_transformer_distill
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid_tranformer", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    replay_num_classes = 0
    if(cfg.SOLVER.USE_REPLAY_CLASS):
        replay_train_loader, replay_train_loader_normal, replay_val_loader, replay_num_query, replay_num_classes, replay_camera_num, replay_view_num = make_replay_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes+replay_num_classes, camera_num=camera_num, view_num = view_num)

    if(len(cfg.MODEL.WEIGHT) > 0):
        model.load_param_distill_replay(cfg.MODEL.WEIGHT)


    small_model = make_model(cfg, num_class=num_classes+replay_num_classes, camera_num=camera_num, view_num = view_num)
    if(len(cfg.SMALL.WEIGHT) > 0):
        small_model.load_param_distill_replay(cfg.SMALL.WEIGHT)

    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    # model.load_param_distill_transformer(cfg.MODEL.WEIGHT)

    # small_model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    # small_model.load_param(cfg.SMALL.WEIGHT)

    print("define loss function..")
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    ref_loss_func = FCDistLoss()
    print("define optimizer ..")
    # ref_optimizer, ref_optimizer_center = make_optimizer(cfg, model, center_criterion)

    optimizer, optimizer_center = make_optimizer(cfg, small_model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)
    print("start training transformer model..")

    if(cfg.SOLVER.USE_TRAIN_LOSS):
        do_train_transformer(
            cfg,
            model,
            small_model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            # ref_optimizer,
            # ref_optimizer_center,
            scheduler,
            loss_func,
            ref_loss_func,
            num_query, args.local_rank
        )
    else:
        do_train_transformer_distill(
            cfg,
            model,
            small_model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            # ref_optimizer,
            # ref_optimizer_center,
            scheduler,
            loss_func,
            ref_loss_func,
            num_query, args.local_rank
        )