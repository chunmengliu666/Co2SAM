import os
import time
import torch
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset
from huggingface_hub import hf_hub_download
import argparse

from model import Model
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
from IPython import embed
from groundingdino.util.inference import Model as dino_model
#from grounded_sam_demo import get_grounding_output, load_model, load_image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["WORLD_SIZE"] = "1"

dino_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'background']

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    template_model: Model,
    dino_model1: dino_model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):

    """The SAM training loop."""
    

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    contra_loss = ContraLoss()
    max_iou = 0.


    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        template_losses = AverageMeter()
        contra_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)


        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)

            images_weak, images_strong, images, gt_mask, image_path, class_label = data
            # images_weak = [1, 3, 1024, 1024], minmax = [0, 1]
            # image_strong = [1, 3, 1024, 1024], minmax = [0, 1]
            # images = [1, 3, 768, 1024], minmax = [0, 1]
            # class_label, without background class, 0 is plane not background

            dino_images = (torch.from_numpy(images).squeeze(0).permute(1, 2, 0)*255).numpy().astype(np.uint8)
            # dino_images = [1024, 1024, 3], numpy

            prompts = torch.tensor([])
            for i, label_id in enumerate(class_label[0]):
                
                text_prompts = [dino_class_names[label_id]]

                detections = dino_model1.predict_with_classes(dino_images, text_prompts, cfg.box_threshold, cfg.text_threshold)
                boxes = torch.tensor(detections.xyxy)
                # boxes = [4, 4]
               
                prompts = torch.cat((prompts, boxes), dim=0)


            if len(prompts) > 0:
                prompts = (prompts.cuda().to(dtype=torch.float64),)

                batch_size = images_weak.size(0)


                with torch.no_grad():
                    template_image_embeds,  template_masks,  template_iou_predictions,  template_res_masks = template_model(images_weak, prompts, images_strong.shape[-2:])
                    # template_image_embeds = [1, 256, 64, 64], template_masks[0] = [4, 1024, 1024], template_res_masks = [4, 1, 256, 256]
                    # template_res_masks means low resolution masks from template model


                soft_image_embeds, soft_masks, soft_iou_predictions, soft_res_masks = model(images_weak, prompts, images_strong.shape[-2:])    # teacher
                # soft_image_embeds = [1, 256, 64, 64], soft_masks[0] = [4, 1024, 1024], soft_res_masks[0] = [4, 1, 256, 256]
                pred_image_embeds, pred_masks, iou_predictions, pred_res_masks = model(images_strong, prompts, images_strong.shape[-2:])   # student
                # pred_image_embeds = [1, 256, 64, 64], pred_masks[0] = [4, 1024, 1024], pred_res_masks[0] = [4, 1, 256, 256]

                num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
                loss_focal = torch.tensor(0., device=fabric.device)
                loss_dice = torch.tensor(0., device=fabric.device)
                loss_iou = torch.tensor(0., device=fabric.device)
                loss_template = torch.tensor(0., device=fabric.device)
                loss_contra = torch.tensor(0., device=fabric.device)

                for i, (pred_mask, soft_mask, template_mask, iou_prediction) in enumerate(zip(pred_masks, soft_masks, template_masks, iou_predictions)):
                    pred_mask_entropy = F.softmax(pred_mask, dim=0)
                    entropy = -torch.sum(pred_mask_entropy * torch.log(pred_mask_entropy + 1e-10), dim=0)
                    entropy_loss = entropy.mean()

                    template_mask = (template_mask > 0.).float()
                    loss_contra += contra_loss(soft_image_embeds[i], template_image_embeds[i], soft_res_masks[i].clone().detach(), template_res_masks[i].clone().detach())

                    loss_template += (0.5 * dice_loss(pred_mask, template_mask) + 0.5 * dice_loss(soft_mask, template_mask))

                    soft_mask = (soft_mask > 0.).float()
                    loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
                    loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
                    batch_iou = calc_iou(pred_mask, soft_mask)
                    loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks


                loss_total = 30. * loss_focal + loss_dice + loss_iou + loss_template + loss_contra + entropy_loss
                fabric.backward(loss_total)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                batch_time.update(time.time() - end)
                end = time.time()

                focal_losses.update(loss_focal.item(), batch_size)
                dice_losses.update(loss_dice.item(), batch_size)
                iou_losses.update(loss_iou.item(), batch_size)
                template_losses.update(loss_template.item(), batch_size)
                contra_losses.update(loss_contra.item(), batch_size)
                total_losses.update(loss_total.item(), batch_size)

                fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Template Loss [{template_losses.val:.4f} ({template_losses.avg:.4f})]'
                         f' | Contrast Loss [{contra_losses.val:.4f} ({contra_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

                loss_logger = {"Focal Loss": focal_losses.avg, "Dice Loss": dice_losses.avg,
                "IoU Loss": iou_losses.avg, "Template Loss": template_losses.avg,
                "Contrast Loss": contra_losses.avg, "Total Loss": total_losses.avg}
                fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)
                torch.cuda.empty_cache()

        if epoch % cfg.eval_interval == 0:
            iou = validate(fabric, cfg, args, model, dino_model1, val_dataloader, cfg.name, epoch)
            if iou > max_iou:
                state = {"model": model, "optimizer": optimizer}
                fabric.save(os.path.join(cfg.out_dir, "save", "last-ckpt.pth"), state)
                max_iou = iou


def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box, args) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model)

    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])

    template_model = copy_model(model)

    # init grounding DINO
    repo_id = "ShilongLiu/GroundingDINO"
    filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cache_model = hf_hub_download(repo_id=repo_id, filename=filename)
    dino_model1 = dino_model(model_config_path=cache_config_file,
                                   model_checkpoint_path=cache_model)

    validate(fabric, cfg, args, template_model, dino_model1, val_data, name=cfg.name, epoch=0)
    train_sam(cfg, fabric, model, template_model, dino_model1, optimizer, scheduler, train_data, val_data)

    del model, template_model, train_data, val_data

    
def pre_allocate_memory(percentage=0.9):
   total_memory = torch.cuda.get_device_properties(0).total_memory
   memory_to_allocate = int(total_memory * percentage)
   num_floats = memory_to_allocate // 4  
   dummy_tensor = torch.empty(num_floats, dtype=torch.float32, device='cuda')
   return dummy_tensor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--background_class", type=int, default=20)
    parser.add_argument("--ignore_label", type=int, default=255)
    
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    dummy_tensor = pre_allocate_memory(0.5)

    main(cfg, args)
    torch.cuda.empty_cache()
