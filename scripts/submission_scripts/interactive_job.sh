source .venv/bin/activate
export HF_HOME=/home/rcherukuri/hf
export TRANSFORMERS_CACHE=/home/rcherukuri/hf/transformers
export HF_DATASETS_CACHE=/home/rcherukuri/hf/datasets
export HF_HUB_CACHE=/home/rcherukuri/hf/hub

export HOME=/home/rcherukuri
export PATH="/lustre/home/rcherukuri/lsa-quamba-fork/.venv/bin:/home/rcherukuri/.local/bin:/home/rcherukuri/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin:${PATH}"
module load cudnn/9.10.2
module load cuda/12.1
which nvcc
nvcc --version
nvidia-smi
export CUDA_HOME=/is/software/nvidia/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"