import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam_lora import LoRA_Sam
from typing import Tuple
from copy import deepcopy

from IPython import embed


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_embeddings = None
        self.target_length = 1024
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_h_4b8939.pth")
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def setup(self):
        checkpoint = self.get_checkpoint(self.cfg.model.type)
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)

        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.finetune()

    def finetune(self):
        LoRA_Sam(self.model, 4)
        # self.set_norm_layer()
        # self.set_evp_adaptor_layer()
        # self.set_prompt_layer()

    def set_norm_layer(self):
        for name, param in self.model.image_encoder.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def set_evp_adaptor_layer(self):
        for param in self.model.image_encoder.prompt_generator.parameters():
            param.requires_grad = True

    def set_prompt_layer(self):
        self.model.image_encoder.Prompt_Tokens.requires_grad = True

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts, orig_image_size):

        image_embeddings = self.encode(images)
        H, W = orig_image_size

        pred_masks, ious, res_masks = self.decode((H, W), prompts)

        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        self.image_embeddings = self.model.image_encoder(images)
        return self.image_embeddings 

    def decode(self, image_shape, prompts):

        image_embeddings = self.image_embeddings
        if image_embeddings == None:
            raise "No image embeddings"

        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):

            if isinstance(prompt, torch.Tensor):

                prompt = prompt.to(device=embedding.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=prompt,
                masks=None,
            )
            elif isinstance(prompt, tuple):

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=prompt,
                boxes=None,
                masks=None,
            )


            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            

            if self.model.training:
                masks = F.interpolate(
                    low_res_masks,
                    image_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                masks = self.model.postprocess_masks(low_res_masks, image_shape, image_shape)

            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)

        return pred_masks, ious, res_masks
    
    @staticmethod
    def get_preprocess_shape1(oldh, oldw, long_side_length):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords_torch1(self, coords, original_size):
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size

        oldh1 = int(original_size[0])
        new_h, new_w = self.get_preprocess_shape1(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

        
    def apply_boxes_torch1(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ):
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch1(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""

        # Normalize colors
        x = (x - self.pixel_mean.cuda()) / self.pixel_std.cuda()

        # Pad
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w

        x = F.pad(x, (0, padw, 0, padh))
        return x