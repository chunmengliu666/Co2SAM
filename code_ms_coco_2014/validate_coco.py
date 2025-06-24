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
from utils.eval_utils_coco import AverageMeter, calc_iou, validate, get_prompts
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
from IPython import embed
from groundingdino.util.inference import Model as dino_model
#from grounded_sam_demo import get_grounding_output, load_model, load_image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def main(cfg, args):
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

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    _, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    val_data = fabric._setup_dataloader(val_data)
    model = fabric.setup(model)

    full_checkpoint = fabric.load(args.save_checkpoint)

    model.load_state_dict(full_checkpoint["model"])

    # Init grounding DINO
    repo_id = "ShilongLiu/GroundingDINO"
    filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cache_model = hf_hub_download(repo_id=repo_id, filename=filename)
    dino_model1 = dino_model(model_config_path=cache_config_file, model_checkpoint_path=cache_model)

    validate(fabric, cfg, args, model, dino_model1, val_data, name=cfg.name, epoch=args.epoch)

def pre_allocate_memory(percentage=0.9):
   total_memory = torch.cuda.get_device_properties(0).total_memory
   memory_to_allocate = int(total_memory * percentage)
   num_floats = memory_to_allocate // 4  
   dummy_tensor = torch.empty(num_floats, dtype=torch.float32, device='cuda')
   # print(f"Pre-allocated {dummy_tensor.numel() * 4 / (1024 ** 2)} MB of GPU memory")
   return dummy_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=81)
    parser.add_argument("--background_class", type=int, default=80)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--save_checkpoint", type=str, default='checkpoints/last-ckpt_7942.pth')
    

    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    # dummy_tensor = pre_allocate_memory(0.5)

    main(cfg, args)
    torch.cuda.empty_cache()
