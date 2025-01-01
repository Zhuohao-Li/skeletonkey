#!/bin/bash

#SBATCH --job-name=vim_1          # 作业名称
#SBATCH --account=PAS2490		    # Project ID
#SBATCH --output=/users/PAS2490/marcusshen/Vim/output_logs_vim/vim_1.log         # 输出日志文件
#SBATCH --error=/users/PAS2490/marcusshen/Vim/output_logs_vim/vim_1_error.log           # 错误日志文件
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=4	        # GPU per node
#SBATCH --mem=80G                   # 内存限制
#SBATCH --time=04:00:00             # 作业运行时间限制

# 运行命令或脚本 wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
# source $HOME/miniconda3/bin/activate /users/PAS2490/marcusshen/miniconda3/envs/vim
# module load cuda 
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \--nproc_per_node=4 --use_env main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 25 --data-set CIFAR --data-path ./datasets/cifar-100-python --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp

# vim_tiny cifar100
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,7 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --use_env main.py \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 128 \
#     --drop-path 0.0 \
#     --weight-decay 0.1 \
#     --num_workers 25 \
#     --data-set CIFAR \
#     --data-path ./datasets/cifar-100-python \
#     --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --no_amp

# vim tiny, cifar10
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,7 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --use_env main.py \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 128 \
#     --drop-path 0.0 \
#     --weight-decay 0.1 \
#     --num_workers 25 \
#     --data-set CIFAR10 \
#     --data-path ./datasets/cifar-10-python \
#     --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_cifar10 \
#     --finetune ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth \
#     --no_amp

# vim_small, cifar100
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --use_env main.py \
#     --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 128 \
#     --drop-path 0.0 \
#     --weight-decay 0.1 \
#     --num_workers 25 \
#     --data-set CIFAR \
#     --data-path ./datasets/cifar-100-python \
#     --output_dir ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --no_amp

# vim_small, cifar10
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --use_env main.py \
#     --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 128 \
#     --drop-path 0.0 \
#     --weight-decay 0.1 \
#     --num_workers 25 \
#     --data-set CIFAR10 \
#     --data-path ./datasets/cifar-10-python \
#     --output_dir ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_cifar10 \
#     --finetune ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth \
#     --no_amp\


# vim_tiny, tinyimagenet
# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 128 \
#     --drop-path 0.0 \
#     --weight-decay 0.1 \
#     --num_workers 25 \
#     --data-set IMNET \
#     --data-path ./datasets/tiny-imagenet-200 \
#     --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_tinyimagenet \
#     --finetune ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth \
#     --epochs 300 \
#     --no_amp

# vim small, tinyimagenet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 256 \
    --lr 5e-5 \
    --min-lr 1e-5 \
    --warmup-lr 1e-5 \
    --drop-path 0.0 \
    --weight-decay 1e-8 \
    --num_workers 25 \
    --data-set TINYIMAGENET \
    --data-path ./datasets/tiny-imagenet-200 \
    --output_dir ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_tinyimagenet \
    --finetune ./output/vim_s_midclstok_80p5acc.pth \
    --epochs 30 \
    --no_amp