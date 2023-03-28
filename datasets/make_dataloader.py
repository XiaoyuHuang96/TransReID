import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import collections
import numpy as np
import copy
import os
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .preprocessing import Preprocessor, Preprocessor28
from .sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler, RandomIdentitySamplerNoBatch
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .cuhk_sysu import CUHK_SYSU
from .viper import VIPeR
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'cuhk_sysu': CUHK_SYSU,
    'viper': VIPeR,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def train_collate_fn_replay(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, fnames, pids, camids, viewids  = zip(*batch)
    # print("fnames", fnames)
    # fnames = torch.tensor(fnames, dtype=torch.int64)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), fnames, pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=False,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num


def make_replay_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.REPLAY_NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=False,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num


def getDatasetClassNum(root_dir, datasetName):
    dataset = __factory[datasetName](root=root_dir)
    num_classes = dataset.num_train_pids
    return num_classes


def extract_features(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    views_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            features = model(imgs)
            for fname, feature, pid, cid, view in zip(fnames, features, pids, cids, domains):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
                views_all.append(view)
    model.train()
    return features_all, labels_all, fnames_all, camids_all, views_all

def select_replay_samples(cfg, model, training_phase=0, add_num=0, old_datas=None, select_samples=2):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.REPLAY_NAMES](root=cfg.DATASETS.ROOT_DIR)

    # train_set = ImageDataset(dataset.train, train_transforms)
    # train_set_normal = ImageDataset(dataset.train, val_transforms)
    # num_classes = dataset.num_train_pids
    # cam_num = dataset.num_train_cams
    # view_num = dataset.num_train_vids


    replay_data = []
    # normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # transformer = T.Compose([
    #     T.Resize((256, 128), interpolation=3),
    #     T.ToTensor(),
    #     normalizer
    # ])

    # train_transformer = T.Compose([
    #     T.Resize((256, 128), interpolation=3),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((256, 128)),
    #     T.ToTensor(),
    #     normalizer,
    #     T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    # ])

    # train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
    #                           batch_size=128, num_workers=4, shuffle=True, pin_memory=False, drop_last=False)
    train_loader = DataLoader(
        Preprocessor(dataset.train, root=dataset.dataset_dir,transform=train_transforms), batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, pin_memory=False, drop_last=False #, collate_fn=train_collate_fn_replay
    )
    print("extract features...")
    features_all, labels_all, fnames_all, camids_all, views_all = extract_features(model, train_loader)

    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)
    pid2views = collections.defaultdict(list)
    pid2idx = collections.defaultdict(list)

    for feature, pid, fname, cid, view in zip(features_all, labels_all, fnames_all, camids_all, views_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)
        pid2views[pid].append(view)

    labels_all = list(set(labels_all))
    # print("labels_all", max(labels_all), len(labels_all))
    # print("pid2features.keys()", max(pid2features.keys()), len(pid2features.keys()))
    for i, pid in enumerate(sorted(pid2features.keys())):
        pid2idx[pid] = i

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=True)
    # print("len(class_centers)", len(class_centers))
    for pid in select_pids:
        # print("pid", pid)
        pidx = pid2idx[pid]
        # print("pidx", pidx)
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pidx]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        # data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
        #                      batch_size=128,num_workers=8, sampler=RandomIdentitySampler(replay_data, select_samples),
        #                      pin_memory=False, drop_last=True)
        data_loader_replay = DataLoader(
            Preprocessor(replay_data, root=dataset.dataset_dir,transform=train_transforms), batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySamplerNoBatch(replay_data, select_samples),
            num_workers=num_workers, pin_memory=False, drop_last=True #, collate_fn=train_collate_fn_replay
        )
    else:
        replay_data.extend(old_datas)
        # data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
        #                      batch_size=128,num_workers=8,
        #                      sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
        #                      pin_memory=False, drop_last=True)
        data_loader_replay = DataLoader(
            Preprocessor28(replay_data, root=dataset.dataset_dir,transform=train_transforms), batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
            num_workers=num_workers, pin_memory=False, drop_last=True #, collate_fn=train_collate_fn_replay
        )

    return data_loader_replay, replay_data

def getReplayDataLoaderByPath(cfg, select_samples=2):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    replay_dataset = torch.load(os.path.join(cfg.SOLVER.REPLAY_DATASET_PATH, 'old_replay_data_phase{}.pth'.format(cfg.SOLVER.TRAINING_PHASE)))
    replay_dataloader = DataLoader(
        Preprocessor(replay_dataset, root=dataset.dataset_dir,transform=train_transforms), batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=MultiDomainRandomIdentitySampler(replay_dataset, select_samples),
        num_workers=num_workers, pin_memory=False, drop_last=True #, collate_fn=train_collate_fn_replay
    )

    return replay_dataloader, replay_dataset


def getReplayDataLoaderByPath28(cfg, select_samples=2):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    replay_dataset = torch.load(os.path.join(cfg.SOLVER.REPLAY_DATASET_PATH, 'old_replay_data_phase{}.pth'.format(cfg.SOLVER.TRAINING_PHASE)))
    replay_dataloader = DataLoader(
        Preprocessor28(replay_dataset, root=dataset.dataset_dir,transform=train_transforms), batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySamplerNoBatch(replay_dataset, select_samples),
        num_workers=num_workers, pin_memory=False, drop_last=True #, collate_fn=train_collate_fn_replay
    )

    return replay_dataloader, replay_dataset