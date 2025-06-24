set -e
set -x

# for ms coco dataset

# export CUDA_HOME=/usr/local/cuda-11
export TOKENIZERS_PARALLELISM=false


# for training
python adaptation_coco.py

#for validation
python validate_coco.py --save_checkpoint ./output/COCO/save/ckpt_2.pth --epoch 2

# instrall GroundingDINO
# echo $CUDA_HOME
# git clone https://github.com/IDEA-Research/GroundingDINO.git
# cd GroundingDINO/
# pip install -e .
