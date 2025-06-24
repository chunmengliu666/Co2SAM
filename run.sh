set -e
set -x

# export CUDA_HOME=/usr/local/cuda-11
export TOKENIZERS_PARALLELISM=false

# for training
 python adaptation.py

# for validation
 python validate.py


# install GroundingDINO
# echo $CUDA_HOME
# git clone https://github.com/IDEA-Research/GroundingDINO.git
# cd GroundingDINO/
# pip install -e .
