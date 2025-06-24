import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_soft, collate_fn_, decode_mask
from datasets.tools_val import Resize
from IPython import embed
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, cfg, root_dir, transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform

        if training:
            segment_root = os.path.join(root_dir, "coco_anno/SegmentationClass/train2014_detectron2")
        else:
            segment_root = os.path.join(root_dir, "coco_anno/SegmentationClass/val2014_detectron2")
        # root_dir = './data/coco2014'
        # segment_root = './data/coco2014/coco_anno/SegmentationClass'

        all_anns = [os.path.join(segment_root, f) for f in os.listdir(segment_root) if f.endswith('.png')]
        all_anns = sorted(all_anns)


        train_list_gt = open(os.path.join(root_dir, "coco_name/train.txt")).read().splitlines()
        train_list = []
        for train_gt_name in train_list_gt:
            train_gt_name = os.path.join(self.root_dir, "coco_anno/SegmentationClass/train2014_detectron2/" + train_gt_name + '.png')

            train_list.append(train_gt_name)
        # len(train_list) = 82081, ./data/coco2014/coco_anno/SegmentationClass/train2014/COCO_train2014_000000007603.png',


        eval_list_gt = open(os.path.join(root_dir, "coco_name/val.txt")).read().splitlines()
        eval_list = []
        for eval_gt_name in eval_list_gt:
            eval_gt_name = os.path.join(self.root_dir, "coco_anno/SegmentationClass/val2014_detectron2/" + eval_gt_name + '.png')
            # eval_gt_name = './data/coco2014/coco_anno/SegmentationClass/val2014/COCO_val2014_000000000042.png'
            eval_list.append(eval_gt_name)
        # len(eval_list) = 40137, './data/coco2014/coco_anno/SegmentationClass/val2014/COCO_val2014_000000014278.png',


        if training:
            random.shuffle(train_list)
            image_ids = train_list

        else:
            random.shuffle(eval_list)
            image_ids = eval_list

        self.image_ids = image_ids

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.image_ids)

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        # embed(header='======PascalVOC.py:97======')
        return label_mask

    def __getitem__(self, idx):

        anno_path = self.image_ids[idx]
        # './data/coco2014/coco_anno/SegmentationClass/val2014/COCO_val2014_000000014278.png',
        if self.if_self_training:
            image_path = anno_path.replace("coco_anno/SegmentationClass/train2014_detectron2", "train2014").replace(".png", ".jpg")
        else:
            image_path = anno_path.replace("coco_anno/SegmentationClass/val2014_detectron2", "val2014").replace(".png", ".jpg")


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = np.array(Image.open(anno_path))

        categories = np.unique(gt_mask)
        categories = [item for item in categories if item != 255]
        categories = [item for item in categories if item != 80]

        if self.if_self_training:
            image_weak, image_strong = soft_transform(image)
            image = self.transform(image)

            if self.transform:
                image_weak = self.transform(image_weak)
                image_strong = self.transform.transform_image(image_strong)

            return image_weak, image_strong, image, gt_mask, image_path, categories
        else:
            if self.transform:
                image = self.transform(image)

            return image, gt_mask, image_path, categories


def load_datasets_soft(cfg, img_size):
    # in this way
   
    transform = ResizeAndPad(img_size)
    transform_val = Resize(img_size)
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.coco.root_dir,
        transform=transform_val,
    )
    soft_train = COCODataset(
        cfg,
        root_dir=cfg.datasets.coco.root_dir,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader

