#!/usr/bin/env bash

set -x

log_out=$1
ratio=$2
gpu_rank=$3
lr=$4
drop_path=$5
layer_decay=$6
master_port=$7

# 调试变量传递
echo "log_out: ${log_out}"
echo "ratio: ${ratio}"
echo "gpu_rank: ${gpu_rank}"
echo "lr: ${lr}"
echo "drop_path: ${drop_path}"
ehco "layer_decay: ${layer_decay}"


export MASTER_PORT=${master_port}


OUTPUT_DIR="/cluster/home3/zhaoyutian/results/MRI_narcotize_video/video_mae/${log_out}"
DATA_PATH="/cluster/home3/zhaoyutian/datasets/MRI_narcotize_video/video_structure_original/4_1_no_test"
DATA_ROOT='/cluster/home3/zhaoyutian/datasets/MRI_narcotize_video/video_structure_original/frames'
MODEL_PATH='/cluster/home3/zhaoyutian/code/VideoMAEv2/vit_s_k710_dl_from_giant.pth'

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:10}  # Other training args

# batch_size can be adjusted according to the graphics card
# Please refer to `run_mae_pretraining.py` for the meaning of the following hyperreferences
CUDA_VISIBLE_DEVICES=${gpu_rank} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr=127.0.0.1 \
        /cluster/home3/zhaoyutian/code/VideoMAEv2/run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set MRI \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 48 \
        --input_size 224 \
        --short_side_size 316 \
        --save_ckpt_freq 20 \
        --num_frames 12 \
        --num_sample 1 \
        --num_workers 4 \
        --opt adamw \
        --lr ${lr} \
        --drop_path ${drop_path} \
        --clip_grad 0 \
        --layer_decay ${layer_decay} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --warmup_epochs 15 \
        --epochs 150 \
        --test_num_segment 1 \
        --test_num_crop 1 \
        --dist_eval \
        --no_pin_mem \
        --no_save_ckpt \
        --enable_deepspeed \
        --reprob 0 \
        --mixup 0 \
        --cutmix 0 \
        --smoothing 0 \
        --enable_wandb \
        --log_info ${log_out} \
        --gpu_num ${GPUS_PER_NODE} \
        --ratio ${ratio} \
        ${PY_ARGS}