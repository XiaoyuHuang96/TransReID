from utils.logger import setup_logger
from datasets import make_dataloader, make_replay_dataloader, getDatasetClassNum
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
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

    logger = setup_logger("transreid", output_dir, if_train=True)
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

    viper_num_classes = getDatasetClassNum(root_dir=cfg.DATASETS.ROOT_DIR, datasetName='viper')
    market_num_classes = getDatasetClassNum(root_dir=cfg.DATASETS.ROOT_DIR, datasetName='market1501')
    cuhk_sysu_num_classes = getDatasetClassNum(root_dir=cfg.DATASETS.ROOT_DIR, datasetName='cuhk_sysu')
    msmt_num_classes = getDatasetClassNum(root_dir=cfg.DATASETS.ROOT_DIR, datasetName='msmt17')

    replay_num_classes = 0
    add_num = 0
    old_replay_dataset = None
    old_replay_dataset_path = os.path.join(cfg.OUTPUT_DIR, 'old_replay_data_phase{}.pth'.format(cfg.SOLVER.TRAINING_PHASE-1))
    if(cfg.SOLVER.TRAINING_PHASE == 2):
        replay_num_classes += viper_num_classes
    elif(cfg.SOLVER.TRAINING_PHASE == 3):
        replay_num_classes += viper_num_classes + market_num_classes
        add_num = viper_num_classes
        if(os.exists(old_replay_dataset_path)):
            old_replay_dataset = torch.load(old_replay_dataset_path)
    elif(cfg.SOLVER.TRAINING_PHASE == 4):
        replay_num_classes += viper_num_classes + market_num_classes + cuhk_sysu_num_classes
        add_num = viper_num_classes + market_num_classes
        if(os.exists(old_replay_dataset_path)):
            old_replay_dataset = torch.load(old_replay_dataset_path)

    model = make_model(cfg, num_class=num_classes+replay_num_classes, camera_num=camera_num, view_num = view_num)

    if(len(cfg.MODEL.WEIGHT) > 0):
        model.load_param_distill_replay(cfg.MODEL.WEIGHT)

    replay_dataloader, replay_dataset = select_replay_samples(model, training_phase=cfg.SOLVER.TRAINING_PHASE,\
                                                   add_num=add_num, old_datas=old_replay_dataset)
    
    torch.save(replay_dataset,
                os.path.join(cfg.OUTPUT_DIR, 'old_replay_data_phase{}.pth'.format(cfg.SOLVER.TRAINING_PHASE)))

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank, 
        replay_dataloader,
    )
