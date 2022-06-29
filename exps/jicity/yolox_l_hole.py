# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp

# only need edit here
VOC_CLASSES = (
    "hole",
    "no_hole"
)
# ROOT_PATH = "/lzh/datasets/JiCity/ZShole20210802/"
# ROOT_PATH = "/lzh/datasets/JiCity/ZSDXJFhole20211230train/"
ROOT_PATH = "/home/data/1196"
NAME = "ZShole"
IMG_SUFFIX=".jpg"

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = len(VOC_CLASSES)
        self.depth = 1.0
        self.width = 1.0        
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.output_dir = '/project/train/models/'        

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            EVVOC,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = EVVOC(
            img_size=self.input_size,
            train=True,
            voc_classes=VOC_CLASSES,
            root_path=ROOT_PATH,
            name=NAME,
            img_suffix=IMG_SUFFIX,
            preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob)
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import EVVOC, ValTransform

        valdataset = EVVOC(
            img_size=self.test_size,
            train=False,
            voc_classes=VOC_CLASSES,
            root_path=ROOT_PATH,
            name=NAME,
            img_suffix=IMG_SUFFIX,
            preproc=ValTransform(legacy=legacy)
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
