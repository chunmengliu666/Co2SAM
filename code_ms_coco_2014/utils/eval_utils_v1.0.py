import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
from groundingdino.util.inference import Model as dino_model
from IPython import embed
import numpy as np
import torch.nn.functional as F
from PIL import Image
import json
import torchvision.transforms as transforms
import GroundingDINO.groundingdino.datasets.transforms as T

palette = [ 128, 0, 0, 
           0, 128, 0, 
           128, 128, 0, 
           0, 0, 128, 
           128, 0, 128, 
           0, 128, 128, 
           128, 128, 128,
           64, 0, 0, 
           192, 0, 0, 
           64, 128, 0, 
           192, 128, 0, 
           64, 0, 128, 
           192, 0, 128, 
           64, 128, 128, 
           192, 128, 128,
           0, 64, 0, 
           128, 64, 0, 
           0, 192, 0, 
           128, 192, 0, 
           0, 64, 128,
           0, 0, 0]

dino_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'background']

pmod_folder = './output/pmod'
score_folder = './output/score'

if not os.path.exists(pmod_folder):
        os.makedirs(pmod_folder)

if not os.path.exists(score_folder):
        os.makedirs(score_folder)


def _fast_hist(label_true, label_pred, n_class):

    mask = (label_true >= 0) & (label_true < n_class)

    if len(label_true) == len(label_pred) and len(label_pred) == len(mask):
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
    else:
        embed(header='=====metric.py:81======') 
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts


def validate(fabric: L.Fabric, cfg: Box, args, model: Model, dino_model1, val_dataloader: DataLoader, name: str, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    print('======validation======')

    preds_list, gts_list = [], []
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            if iter % 20 == 0:
                print('eval_iter:', iter)

            image, gt_mask, image_path, class_label = data
            dino_images = (image.squeeze(0).permute(1, 2, 0)*255).cpu().numpy().astype(np.uint8)

            detections = dino_model1.predict_with_classes(dino_images, dino_class_names, cfg.box_threshold, cfg.text_threshold)
            boxes = torch.tensor(detections.xyxy)

            logits = torch.tensor(detections.confidence)
            class_ids = detections.class_id
            class_ids = torch.tensor(class_ids.astype(np.float32))

            class_ids.nan_to_num(args.background_class)

            class_ids = class_ids.int()
            class_ids[class_ids < 0] = args.background_class

            transformed_boxes = model.apply_boxes_torch1(boxes, dino_images.shape[:2])

            
            r = torch.zeros((args.num_classes + 1, *dino_images.shape[:2]))

            mask_size = image.shape[-2:]
            height, width = gt_mask[0].shape

            if len(transformed_boxes) > 0:
                prompts = (transformed_boxes.cuda().to(dtype=torch.float64),)

                dino_images_resize = torch.tensor(dino_images).unsqueeze(0).permute(0, 3, 1, 2).cuda()

                dino_images_resize = model.preprocess(dino_images_resize)

                _, pred_masks, _, _ = model(dino_images_resize, prompts,dino_images.shape[:2])

                masks = pred_masks[0] * logits[:, None, None].cuda()

                for class_id in class_ids.unique():
                    r[class_id] = masks[class_ids == class_id].max(dim=0)[0]

                r[args.background_class, r.max(dim=0).values <= 0] = 1.
                
                r = sem_seg_postprocess(r, mask_size, height, width)
                r = r[:-1]
                r_mask = r.numpy()
                pseudo_label = np.argmax(r_mask, axis=0).astype(np.uint8)
            else:
                r = sem_seg_postprocess(r, mask_size, height, width)
                r = r[:-1]
                r_mask = r.numpy()

                pseudo_label = np.argmax(r_mask, axis=0)
                pseudo_label.fill(args.background_class) 
                pseudo_label = pseudo_label.astype(np.uint8)

            out = Image.fromarray(pseudo_label, mode='P')
            predict = torch.from_numpy(pseudo_label).unsqueeze(0).numpy()

            out.putpalette(palette)

            out_name = pmod_folder + '/' + image_path[0][-15:-4] + '.png'
            out.save(out_name)

            preds_list += list(predict)
            gts_list += list(gt_mask[0][np.newaxis, :])


    score = scores(gts_list, preds_list, n_class=args.num_classes)


    with open(score_folder + '_epoch' + str(epoch) + '.json', "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)

    return score['Mean IoU']




def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result