from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "6",
    "box_threshold": 0.3,
    "text_threshold": 0.25,
    "batch_size": 1,
    "val_batchsize": 1,
    "num_workers": 4,
    "num_epochs": 50,
    "max_nums": 50,
    "num_points": 5,
    "resume": False,
    "dataset": "PascalVOC",
    "visual": False,
    "load_type": "soft",
    "prompt": "box",
    "out_dir": "output/PascalVOC/",
    "name": "base",
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    }
}

cfg = Box(base_config)
cfg.merge_update(config)

